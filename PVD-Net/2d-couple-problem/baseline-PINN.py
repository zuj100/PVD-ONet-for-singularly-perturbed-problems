import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


# ======================================================================
# reproducibility / device
# ======================================================================
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)


# ======================================================================
# problem setting
# Keep the same main hyperparameters as the PVD-Net code.
# ======================================================================
eps = 1.0e-3
xi0 = 20.0
net_width = 100
lr = 1.0e-4
epochs_adam = 100000

kx = np.pi / 2.0


# ======================================================================
# exact solution / manufactured source
# ======================================================================
def u_exact_np(x, y):
    ex = np.exp(-2.0 * x / eps)
    ey = np.exp(-3.0 * y / eps)
    return np.cos(kx * x) * (1.0 - ex) * (1.0 - y ** 3) * (1.0 - ey)


def f_exact_np(x, y):
    """
    Stable evaluation of the full manufactured source f(x,y).
    """
    ex = np.exp(-2.0 * x / eps)
    ey = np.exp(-3.0 * y / eps)

    p = np.cos(kx * x)
    s = np.sin(kx * x)
    q = 1.0 - y ** 3

    ax = 1.0 - ex
    by = 1.0 - ey

    xx = p * ax
    yy = q * by

    # -eps*X''-(x+2)X'
    fx = ((eps * kx ** 2 * p + (x + 2.0) * kx * s) * ax
          + 4.0 * kx * s * ex
          - (2.0 * x / eps) * p * ex)

    # -eps*Y''-(y^3+3)Y'
    fy = ((6.0 * eps * y + 3.0 * (y ** 3 + 3.0) * y ** 2) * by
          + 18.0 * y ** 2 * ey
          - (3.0 * y ** 3 / eps) * q * ey)

    return yy * fx + xx * fy + xx * yy


def f_exact_torch(x, y):
    """
    Torch version of the full manufactured source.
    This is the source used by the PINN residual.
    """
    ex = torch.exp(-2.0 * x / eps)
    ey = torch.exp(-3.0 * y / eps)

    p = torch.cos(kx * x)
    s = torch.sin(kx * x)
    q = 1.0 - y ** 3

    ax = 1.0 - ex
    by = 1.0 - ey

    xx = p * ax
    yy = q * by

    fx = ((eps * kx ** 2 * p + (x + 2.0) * kx * s) * ax
          + 4.0 * kx * s * ex
          - (2.0 * x / eps) * p * ex)

    fy = ((6.0 * eps * y + 3.0 * (y ** 3 + 3.0) * y ** 2) * by
          + 18.0 * y ** 2 * ey
          - (3.0 * y ** 3 / eps) * q * ey)

    return yy * fx + xx * fy + xx * yy


# ======================================================================
# single PINN
# Same width, depth and activation as each Net in the PVD-Net code.
# ======================================================================
class Net(nn.Module):
    def __init__(self, width=net_width):
        super(Net, self).__init__()

        layers = [nn.Linear(2, width), nn.SiLU()]

        for _ in range(4):
            layers += [nn.Linear(width, width), nn.SiLU()]

        layers += [nn.Linear(width, 1)]

        self.main = nn.Sequential(*layers)

    def forward(self, x, y):
        return self.main(torch.cat([x, y], dim=1))


net = Net().to(device)


def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


net.apply(init_normal)

n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('trainable parameters:', n_params)


# ======================================================================
# tensor helper
# ======================================================================
def t(a):
    return torch.tensor(
        np.asarray(a, dtype=np.float64),
        dtype=torch.float32,
        device=device
    ).reshape(-1, 1)


# ======================================================================
# collocation points
#
# Same PDE-point budget as the PVD-Net:
#   n_o = 800
#   n_i = 500
#   n_i = 500
#   n_c = 400
#
# But ALL points are physical (x,y) points and are fed to ONE network.
# ======================================================================
n_o, n_i, n_c = 800, 500, 400
n_bc_o = 200

# ----------------------------------------------------------------------
# 1. bulk points
# ----------------------------------------------------------------------
X_O = t(np.random.uniform(0.0, 1.0, n_o))
Y_O = t(np.random.uniform(0.0, 1.0, n_o))

# ----------------------------------------------------------------------
# 2. x-boundary-layer points
# Original PVD-Net uses xi in [0, xi0].
# Map them back by x = eps * xi.
# ----------------------------------------------------------------------
XI_X = np.random.uniform(0.0, xi0, n_i)
Y_X_np = np.random.uniform(0.0, 1.0, n_i)

X_X = t(eps * XI_X)
Y_X = t(Y_X_np)

# ----------------------------------------------------------------------
# 3. y-boundary-layer points
# y = eps * eta
# ----------------------------------------------------------------------
X_Y_np = np.random.uniform(0.0, 1.0, n_i)
ETA_Y = np.random.uniform(0.0, xi0, n_i)

X_Y = t(X_Y_np)
Y_Y = t(eps * ETA_Y)

# ----------------------------------------------------------------------
# 4. corner points
# x = eps*xi, y = eps*eta
# ----------------------------------------------------------------------
XI_C = np.random.uniform(0.0, xi0, n_c)
ETA_C = np.random.uniform(0.0, xi0, n_c)

X_C = t(eps * XI_C)
Y_C = t(eps * ETA_C)

# Concatenate all physical PDE points.
X_PDE = torch.cat([X_O, X_X, X_Y, X_C], dim=0)
Y_PDE = torch.cat([Y_O, Y_X, Y_Y, Y_C], dim=0)

print('number of PDE points:', X_PDE.shape[0])


# ======================================================================
# boundary points
# Standard PINN imposes the original Dirichlet condition on all 4 walls.
# ======================================================================
Y_B0 = t(np.random.uniform(0.0, 1.0, n_bc_o))
Y_B1 = t(np.random.uniform(0.0, 1.0, n_bc_o))

X_B0 = t(np.random.uniform(0.0, 1.0, n_bc_o))
X_B1 = t(np.random.uniform(0.0, 1.0, n_bc_o))



def grad(u, v):
    return torch.autograd.grad(
        u,
        v,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        only_inputs=True
    )[0]


# ======================================================================
# PINN losses
# ======================================================================
def loss_pde():
    """
    Full original PDE residual:

        -eps*(u_xx+u_yy)
        -(x+2)u_x
        -(y^3+3)u_y
        +u
        -f(x,y) = 0.
    """
    x = X_PDE.clone().detach().requires_grad_(True)
    y = Y_PDE.clone().detach().requires_grad_(True)

    u = net(x, y)

    u_x = grad(u, x)
    u_y = grad(u, y)

    u_xx = grad(u_x, x)
    u_yy = grad(u_y, y)

    f = f_exact_torch(x, y)

    r = (-eps * (u_xx + u_yy)
         - (x + 2.0) * u_x
         - (y ** 3 + 3.0) * u_y
         + u
         - f)

    return (r ** 2).mean()


def loss_bc():
    """
    Homogeneous Dirichlet condition on all four physical boundaries:

        u(0,y)=0,
        u(1,y)=0,
        u(x,0)=0,
        u(x,1)=0.
    """
    zero_y0 = torch.zeros_like(Y_B0)
    one_y1 = torch.ones_like(Y_B1)

    zero_x0 = torch.zeros_like(X_B0)
    one_x1 = torch.ones_like(X_B1)

    l_x0 = (net(zero_y0, Y_B0) ** 2).mean()
    l_x1 = (net(one_y1, Y_B1) ** 2).mean()

    l_y0 = (net(X_B0, zero_x0) ** 2).mean()
    l_y1 = (net(X_B1, one_x1) ** 2).mean()

    return l_x0 + l_x1 + l_y0 + l_y1


def loss_components():
    return {
        'pde': loss_pde(),
        'bc': loss_bc(),
    }


def total_loss():
    L = loss_components()

    # Standard PINN baseline:
    # original PDE residual + original boundary-condition loss.
    return L['pde'] + L['bc']


# ======================================================================
# output folders
# ======================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, 'PINN-baseline')

os.makedirs(os.path.join(out_dir, 'plots'), exist_ok=True)
os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)

best_path = os.path.join(out_dir, 'model', 'best_PINN.pth')


def save_best():
    torch.save(net.state_dict(), best_path)


# ======================================================================
# training
# ======================================================================
print(f'Adam epochs = {epochs_adam}, output = {out_dir}')

tic = time.time()

best = float('inf')
loss_hist = []
pde_hist = []
bc_hist = []

optimizer = optim.Adam(
    net.parameters(),
    lr=lr,
    betas=(0.9, 0.99),
    eps=1e-15
)

for epoch in range(epochs_adam):
    optimizer.zero_grad()

    L_comp = loss_components()
    L = L_comp['pde'] + L_comp['bc']

    L.backward()
    optimizer.step()

    v = L.item()

    loss_hist.append(v)
    pde_hist.append(L_comp['pde'].item())
    bc_hist.append(L_comp['bc'].item())

    if v < best:
        best = v
        save_best()

    if epoch % 10000 == 0:
        print(
            f'[Adam] Epoch {epoch:6d}  '
            f'total {v:.3e}  '
            f'pde {L_comp["pde"].item():.3e}  '
            f'bc {L_comp["bc"].item():.3e}  '
            f'best {best:.3e}'
        )


toc_adam = time.time()

print(
    f'Adam phase finished, elapsed {toc_adam - tic:.1f} s, '
    f'best {best:.3e}'
)


# ======================================================================
# load best model
# ======================================================================
net.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
net.eval()


# ======================================================================
# prediction / metrics
# ======================================================================
def predict_np(xnp, ynp, batch_size=65536):
    xnp = np.asarray(xnp).reshape(-1)
    ynp = np.asarray(ynp).reshape(-1)

    pred = []

    with torch.no_grad():
        for i in range(0, len(xnp), batch_size):
            xb = t(xnp[i:i + batch_size])
            yb = t(ynp[i:i + batch_size])

            ub = net(xb, yb)

            pred.append(ub.cpu().numpy().ravel())

    return np.concatenate(pred)


def rel_l2(ue, up):
    return np.linalg.norm(ue - up) / np.linalg.norm(ue)


def linf(ue, up):
    return np.max(np.abs(ue - up))


def region_metrics(name, xg, yg, store):
    XX, YY = np.meshgrid(xg, yg)

    ue = u_exact_np(XX.ravel(), YY.ravel())
    up = predict_np(XX.ravel(), YY.ravel())

    store[name] = (
        rel_l2(ue, up),
        linf(ue, up)
    )

    print(
        f'{name:24s}  '
        f'rel L2 = {store[name][0]:.6e}   '
        f'L inf = {store[name][1]:.6e}'
    )


# ======================================================================
# regional errors
# Keep the same evaluation framework as PVD-Net.
# ======================================================================
strip = 0.01
metrics = {}

region_metrics(
    'x-layer strip',
    np.linspace(0, strip, 201),
    np.linspace(0, 1, 51),
    metrics
)

region_metrics(
    'y-layer strip',
    np.linspace(0, 1, 51),
    np.linspace(0, strip, 201),
    metrics
)

region_metrics(
    'corner region',
    np.linspace(0, strip, 201),
    np.linspace(0, strip, 201),
    metrics
)

region_metrics(
    'global (all grids)',
    np.concatenate([
        np.linspace(0, 1, 101),
        np.linspace(0, strip, 201),
        np.linspace(0, 1, 51)
    ]),
    np.concatenate([
        np.linspace(0, 1, 101),
        np.linspace(0, 1, 51),
        np.linspace(0, strip, 201)
    ]),
    metrics
)


# ======================================================================
# save text results
# ======================================================================
with open(
    os.path.join(out_dir, 'test error.txt'),
    'a',
    encoding='utf-8'
) as f:

    f.write(f'adam epochs: {epochs_adam}\n')
    f.write(f'network parameters: {n_params}\n')
    f.write(f'PDE collocation points: {X_PDE.shape[0]}\n')
    f.write(f'training time (s): {toc_adam - tic:.1f}\n')
    f.write(f'best loss: {best:.3e}\n\n')

    for k, v in metrics.items():
        f.write(
            f'{k}: '
            f'rel L2 = {v[0]:.6e}, '
            f'L inf = {v[1]:.6e}\n'
        )


# ======================================================================
# plots
# ======================================================================
plot_dir = os.path.join(out_dir, 'plots')


# ----------------------------------------------------------------------
# training loss
# ----------------------------------------------------------------------
plt.figure(figsize=(7.2, 5.2))

plt.plot(loss_hist, label='total loss')
plt.plot(pde_hist, label='PDE loss')
plt.plot(bc_hist, label='BC loss')

plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(plot_dir, 'loss.png'),
    dpi=300,
    bbox_inches='tight'
)

plt.close()


# ----------------------------------------------------------------------
# slice plots
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

slices = [
    ('x', 0.5),
    ('x', 0.005),
    ('y', 0.5),
    ('y', 0.005)
]

for ax, (var, val) in zip(axes.ravel(), slices):

    s = np.linspace(0, 1, 801)

    if var == 'x':
        ue = u_exact_np(s, np.full_like(s, val))
        up = predict_np(s, np.full_like(s, val))

        ax.set_xlabel('x')
        ax.set_title(f'slice  y = {val}')

    else:
        ue = u_exact_np(np.full_like(s, val), s)
        up = predict_np(np.full_like(s, val), s)

        ax.set_xlabel('y')
        ax.set_title(f'slice  x = {val}')

    ax.plot(
        s,
        ue,
        'b-',
        label='exact',
        linewidth=2
    )

    ax.plot(
        s,
        up,
        'r--',
        label='PINN',
        linewidth=2
    )

    ax.legend(loc='center right')

    axins = inset_axes(
        ax,
        width='42%',
        height='55%',
        loc='upper right'
    )

    m = s <= strip

    axins.plot(
        s[m],
        ue[m],
        'b-',
        linewidth=2
    )

    axins.plot(
        s[m],
        up[m],
        'r--',
        linewidth=2
    )

    axins.set_xlim(0.0, strip)

    mark_inset(
        ax,
        axins,
        loc1=3,
        loc2=4,
        fc=(0.5, 0.5, 0.5, 0.3),
        ec=(0.5, 0.5, 0.5, 0.3),
        lw=1.0
    )


plt.tight_layout()

plt.savefig(
    os.path.join(plot_dir, 'slices.png'),
    dpi=300,
    bbox_inches='tight'
)

plt.close()


# ======================================================================
# 2D full-field visualization
# exact solution / PINN prediction / pointwise absolute error
# ======================================================================
n_vis = 501

x_vis = np.linspace(0.0, 1.0, n_vis)
y_vis = np.linspace(0.0, 1.0, n_vis)

XX_vis, YY_vis = np.meshgrid(x_vis, y_vis)

U_exact_vis = u_exact_np(
    XX_vis.ravel(),
    YY_vis.ravel()
).reshape(XX_vis.shape)

U_pred_vis = predict_np(
    XX_vis.ravel(),
    YY_vis.ravel()
).reshape(XX_vis.shape)

U_error_vis = np.abs(U_exact_vis - U_pred_vis)

u_vmin = float(
    min(
        U_exact_vis.min(),
        U_pred_vis.min()
    )
)

u_vmax = float(
    max(
        U_exact_vis.max(),
        U_pred_vis.max()
    )
)

extent_vis = [0.0, 1.0, 0.0, 1.0]


# ----------------------------------------------------------------------
# 1. exact solution
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.6, 5.4))

im = ax.imshow(
    U_exact_vis,
    origin='lower',
    extent=extent_vis,
    aspect='equal',
    cmap='jet',
    vmin=u_vmin,
    vmax=u_vmax
)

ax.set_xlabel(r'$x$', fontsize=14)
ax.set_ylabel(r'$y$', fontsize=14)
ax.set_title('Exact solution', fontsize=15)

fig.colorbar(im, ax=ax)

plt.tight_layout()

plt.savefig(
    os.path.join(plot_dir, 'exact_solution_2d.png'),
    dpi=600,
    bbox_inches='tight'
)

plt.close()


# ----------------------------------------------------------------------
# 2. PINN prediction
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.6, 5.4))

im = ax.imshow(
    U_pred_vis,
    origin='lower',
    extent=extent_vis,
    aspect='equal',
    cmap='jet',
    vmin=u_vmin,
    vmax=u_vmax
)

ax.set_xlabel(r'$x$', fontsize=14)
ax.set_ylabel(r'$y$', fontsize=14)
ax.set_title('PINN prediction', fontsize=15)

fig.colorbar(im, ax=ax)

plt.tight_layout()

plt.savefig(
    os.path.join(plot_dir, 'predicted_solution_2d.png'),
    dpi=600,
    bbox_inches='tight'
)

plt.close()


# ----------------------------------------------------------------------
# 3. pointwise absolute error
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.6, 5.4))

im = ax.imshow(
    U_error_vis,
    origin='lower',
    extent=extent_vis,
    aspect='equal',
    cmap='viridis'
)

ax.set_xlabel(r'$x$', fontsize=14)
ax.set_ylabel(r'$y$', fontsize=14)
ax.set_title(
    r'Absolute error $|u_{exact}-u_{pred}|$',
    fontsize=15
)

fig.colorbar(im, ax=ax)

plt.tight_layout()

plt.savefig(
    os.path.join(plot_dir, 'absolute_error_2d.png'),
    dpi=600,
    bbox_inches='tight'
)

plt.close()


# ----------------------------------------------------------------------
# 4. combined figure
# ----------------------------------------------------------------------
fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5.2)
)

im0 = axes[0].imshow(
    U_exact_vis,
    origin='lower',
    extent=extent_vis,
    aspect='equal',
    cmap='jet',
    vmin=u_vmin,
    vmax=u_vmax
)

axes[0].set_title(
    'Exact solution',
    fontsize=14
)

axes[0].set_xlabel(
    r'$x$',
    fontsize=13
)

axes[0].set_ylabel(
    r'$y$',
    fontsize=13
)

fig.colorbar(
    im0,
    ax=axes[0],
    fraction=0.046,
    pad=0.04
)


im1 = axes[1].imshow(
    U_pred_vis,
    origin='lower',
    extent=extent_vis,
    aspect='equal',
    cmap='jet',
    vmin=u_vmin,
    vmax=u_vmax
)

axes[1].set_title(
    'PINN prediction',
    fontsize=14
)

axes[1].set_xlabel(
    r'$x$',
    fontsize=13
)

axes[1].set_ylabel(
    r'$y$',
    fontsize=13
)

fig.colorbar(
    im1,
    ax=axes[1],
    fraction=0.046,
    pad=0.04
)


im2 = axes[2].imshow(
    U_error_vis,
    origin='lower',
    extent=extent_vis,
    aspect='equal',
    cmap='viridis'
)

axes[2].set_title(
    r'Absolute error $|u_{exact}-u_{pred}|$',
    fontsize=14
)

axes[2].set_xlabel(
    r'$x$',
    fontsize=13
)

axes[2].set_ylabel(
    r'$y$',
    fontsize=13
)

fig.colorbar(
    im2,
    ax=axes[2],
    fraction=0.046,
    pad=0.04
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        plot_dir,
        'exact_prediction_error_2d.png'
    ),
    dpi=600,
    bbox_inches='tight'
)

plt.close()


print('2D full-field plots written to', plot_dir)
print('results written to', out_dir)

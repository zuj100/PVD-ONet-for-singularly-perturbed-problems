import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# ----------------------------------------------------------------------
# problem setting
# ----------------------------------------------------------------------
eps = 1.0e-3
xi0 = 20.0
net_width = 100
lr = 1.0e-4

epochs_adam = 100000


kx = np.pi / 2.0


def u_exact_np(x, y):
    ex = np.exp(-2.0 * x / eps)
    ey = np.exp(-3.0 * y / eps)
    return np.cos(kx * x) * (1.0 - ex) * (1.0 - y ** 3) * (1.0 - ey)


def f_exact_np(x, y):
    """Stable evaluation of the manufactured source f(x,y)."""
    ex = np.exp(-2.0 * x / eps)
    ey = np.exp(-3.0 * y / eps)
    p = np.cos(kx * x)
    s = np.sin(kx * x)
    q = 1.0 - y ** 3
    ax = 1.0 - ex
    by = 1.0 - ey
    xx = p * ax
    yy = q * by

    # -eps*X''-(x+2)X', with the O(1/eps) cancellation simplified analytically.
    fx = ((eps * kx ** 2 * p + (x + 2.0) * kx * s) * ax
          + 4.0 * kx * s * ex
          - (2.0 * x / eps) * p * ex)

    # -eps*Y''-(y^3+3)Y', again written after analytical cancellation.
    fy = ((6.0 * eps * y + 3.0 * (y ** 3 + 3.0) * y ** 2) * by
          + 18.0 * y ** 2 * ey
          - (3.0 * y ** 3 / eps) * q * ey)

    return yy * fx + xx * fy + xx * yy


def f0_torch(x, y):
    """O(1) outer source in f = f0 + eps*f1 + exponentially small terms."""
    p = torch.cos(kx * x)
    s = torch.sin(kx * x)
    q = 1.0 - y ** 3
    return ((x + 2.0) * kx * s * q
            + 3.0 * (y ** 3 + 3.0) * y ** 2 * p
            + p * q)


# ----------------------------------------------------------------------
# networks
# ----------------------------------------------------------------------
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


net_o = Net().to(device)
net_x = Net().to(device)
net_y = Net().to(device)
net_c = Net().to(device)
ALL_NETS = [net_o, net_x, net_y, net_c]


def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


for net in ALL_NETS:
    net.apply(init_normal)


# ----------------------------------------------------------------------
# collocation / boundary / matching points
# ----------------------------------------------------------------------
def t(a):
    return torch.tensor(np.asarray(a, dtype=np.float64), dtype=torch.float32,
                        device=device).reshape(-1, 1)


n_o, n_i, n_c = 800, 500, 400
n_bc_o = 200
n_bc_i = 200
n_bc_c = 240

# PDE points
X_O = t(np.random.uniform(0, 1, n_o))
Y_O = t(np.random.uniform(0, 1, n_o))

XI_I = t(np.random.uniform(0, xi0, n_i))
Y_I = t(np.random.uniform(0, 1, n_i))

X_J = t(np.random.uniform(0, 1, n_i))
ETA_J = t(np.random.uniform(0, xi0, n_i))

XI_C = t(np.random.uniform(0, xi0, n_c))
ETA_C = t(np.random.uniform(0, xi0, n_c))

# Outer inflow boundaries x=1 and y=1.
Y_OB = t(np.random.uniform(0, 1, n_bc_o))
X_OB = t(np.random.uniform(0, 1, n_bc_o))

# x-inner: physical boundary xi=0 and matching boundary xi=xi0.
Y_XB = t(np.random.uniform(0, 1, n_bc_i))

# y-inner: physical boundary eta=0 and matching boundary eta=xi0.
X_YB = t(np.random.uniform(0, 1, n_bc_i))

# Corner: two physical corner edges and two far matching edges.
XI_CB = t(np.random.uniform(0, xi0, n_bc_c))
ETA_CB = t(np.random.uniform(0, xi0, n_bc_c))


# ----------------------------------------------------------------------
# losses
# ----------------------------------------------------------------------
def grad(u, v):
    return torch.autograd.grad(u, v, grad_outputs=torch.ones_like(u),
                               create_graph=True, only_inputs=True)[0]


def loss_outer_pde():
    # Reduced O(1) equation:
    #   -(x+2)u_x -(y^3+3)u_y + u = f0(x,y).
    x = X_O.clone().requires_grad_(True)
    y = Y_O.clone().requires_grad_(True)
    u = net_o(x, y)
    r = (-(x + 2.0) * grad(u, x)
         - (y ** 3 + 3.0) * grad(u, y)
         + u - f0_torch(x, y))
    return (r ** 2).mean()


def loss_outer_bc():
    # The reduced transport field is (-(x+2), -(y^3+3)), so its inflow
    # boundary is x=1 union y=1.  The manufactured Dirichlet data are zero.
    one_y = torch.ones_like(Y_OB)
    one_x = torch.ones_like(X_OB)
    l_x1 = (net_o(one_y, Y_OB) ** 2).mean()
    l_y1 = (net_o(X_OB, one_x) ** 2).mean()
    return l_x1 + l_y1


def loss_inner_x_pde():
    # x = eps*xi.  eps*f(eps*xi,y)=O(eps), so the O(1) inner equation is
    # homogeneous.
    xi = XI_I.clone().requires_grad_(True)
    y = Y_I.clone().requires_grad_(True)
    u = net_x(xi, y)
    u_xi = grad(u, xi)
    r = grad(u_xi, xi) + 2.0 * u_xi
    return (r ** 2).mean()


def loss_inner_x_bc():
    # Physical wall at x=0 and Van-Dyke matching at xi=xi0.
    xi_zero = torch.zeros_like(Y_XB)
    xi_far = torch.full_like(Y_XB, xi0)
    wall = (net_x(xi_zero, Y_XB) ** 2).mean()
    match = ((net_x(xi_far, Y_XB)
              - net_o(torch.zeros_like(Y_XB), Y_XB)) ** 2).mean()
    return wall + match


def loss_inner_y_pde():
    # y = eps*eta; eps*f(x,eps*eta)=O(eps).
    x = X_J.clone().requires_grad_(True)
    eta = ETA_J.clone().requires_grad_(True)
    u = net_y(x, eta)
    u_eta = grad(u, eta)
    r = grad(u_eta, eta) + 3.0 * u_eta
    return (r ** 2).mean()


def loss_inner_y_bc():
    # Physical wall at y=0 and Van-Dyke matching at eta=xi0.
    eta_zero = torch.zeros_like(X_YB)
    eta_far = torch.full_like(X_YB, xi0)
    wall = (net_y(X_YB, eta_zero) ** 2).mean()
    match = ((net_y(X_YB, eta_far)
              - net_o(X_YB, torch.zeros_like(X_YB))) ** 2).mean()
    return wall + match


def loss_corner_pde():
    xi = XI_C.clone().requires_grad_(True)
    eta = ETA_C.clone().requires_grad_(True)
    w = net_c(xi, eta)
    w_xi = grad(w, xi)
    w_eta = grad(w, eta)
    r = (grad(w_xi, xi) + 2.0 * w_xi
         + grad(w_eta, eta) + 3.0 * w_eta)
    return (r ** 2).mean()


def loss_corner_bc():
    # Corner boundary data are derived from the composite expansion itself.
    # At x=0:
    #   0 = U_y(0,eta) - u_o(0,0) + U_c(0,eta)
    # hence U_c(0,eta)=u_o(0,0)-U_y(0,eta).
    # At y=0:
    #   0 = U_x(xi,0) - u_o(0,0) + U_c(xi,0)
    # hence U_c(xi,0)=u_o(0,0)-U_x(xi,0).
    zero_eta = torch.zeros_like(ETA_CB)
    zero_xi = torch.zeros_like(XI_CB)
    far_eta = torch.full_like(ETA_CB, xi0)
    far_xi = torch.full_like(XI_CB, xi0)

    uo00_left = net_o(torch.zeros_like(ETA_CB), torch.zeros_like(ETA_CB))
    target_xi0edge = uo00_left - net_y(torch.zeros_like(ETA_CB), ETA_CB)
    l_xi_zero = ((net_c(zero_eta, ETA_CB) - target_xi0edge) ** 2).mean()

    uo00_bottom = net_o(torch.zeros_like(XI_CB), torch.zeros_like(XI_CB))
    target_eta0edge = uo00_bottom - net_x(XI_CB, torch.zeros_like(XI_CB))
    l_eta_zero = ((net_c(XI_CB, zero_xi) - target_eta0edge) ** 2).mean()

    # Matching/decay of the corner correction on the outer edges of the
    # truncated stretched domain.
    l_xi_far = (net_c(far_eta, ETA_CB) ** 2).mean()
    l_eta_far = (net_c(XI_CB, far_xi) ** 2).mean()

    return l_xi_zero + l_eta_zero + l_xi_far + l_eta_far


def composite(x, y):
    # The clamp simply truncates the stretched variables at the matching
    # location.  Because the layer corrections vanish there by construction,
    # the composite remains continuous outside the layer rectangles.
    xi = torch.clamp(x / eps, max=xi0)
    eta = torch.clamp(y / eps, max=xi0)

    uo = net_o(x, y)
    uo_0y = net_o(torch.zeros_like(x), y)
    uo_x0 = net_o(x, torch.zeros_like(y))

    return (uo
            + (net_x(xi, y) - uo_0y)
            + (net_y(x, eta) - uo_x0)
            + net_c(xi, eta))


def loss_components():
    """Return named losses for diagnostics."""
    return {
        'outer_pde': loss_outer_pde(),
        'outer_bc': loss_outer_bc(),
        'x_pde': loss_inner_x_pde(),
        'x_bc': loss_inner_x_bc(),
        'y_pde': loss_inner_y_pde(),
        'y_bc': loss_inner_y_bc(),
        'corner_pde': loss_corner_pde(),
        'corner_bc': loss_corner_bc(),
    }


def total_loss():
    L = loss_components()
    # Preserve the previous v1 convention: layer/corner PDE residuals carry
    # weight 10; all boundary and matching conditions carry unit weight.
    return (L['outer_pde'] + L['outer_bc']
            + 10* (L['x_pde'] + L['y_pde'] + L['corner_pde'])
            + L['x_bc'] + L['y_bc'] + L['corner_bc'])


# ----------------------------------------------------------------------
# training
# ----------------------------------------------------------------------
params = []
for net in ALL_NETS:
    params += list(net.parameters())

script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, 'leading-order')
os.makedirs(os.path.join(out_dir, 'plots'), exist_ok=True)
os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)
NET_NAMES = ['o', 'x', 'y', 'c']
PATHS = [os.path.join(out_dir, 'model', f'best_{n}.pth') for n in NET_NAMES]


def save_best():
    for net, p in zip(ALL_NETS, PATHS):
        torch.save(net.state_dict(), p)


print(f'Adam epochs = {epochs_adam}, output = {out_dir}')

# tic = time.time()
# best = float('inf')
# loss_hist = []
#
# optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.99), eps=1e-15)
# for epoch in range(epochs_adam):
#     optimizer.zero_grad()
#     L = total_loss()
#     L.backward()
#     optimizer.step()
#     v = L.item()
#     loss_hist.append(v)
#     if v < best:
#         best = v
#         save_best()
#     if epoch % 10000 == 0:
#         print(f'[Adam] Epoch {epoch:6d}  total {v:.3e}  best {best:.3e}')
#
# toc_adam = time.time()
# print(f'Adam phase finished, elapsed {toc_adam - tic:.1f} s, best {best:.3e}')



for net, p in zip(ALL_NETS, PATHS):
    net.load_state_dict(torch.load(p, weights_only=True))


# ----------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------
def predict_np(xnp, ynp):
    with torch.no_grad():
        out = composite(t(xnp), t(ynp))
    return out.cpu().numpy().ravel()


def rel_l2(ue, up):
    return np.linalg.norm(ue - up) / np.linalg.norm(ue)


def linf(ue, up):
    return np.max(np.abs(ue - up))


def region_metrics(name, xg, yg, store):
    XX, YY = np.meshgrid(xg, yg)
    ue = u_exact_np(XX.ravel(), YY.ravel())
    up = predict_np(XX.ravel(), YY.ravel())
    store[name] = (rel_l2(ue, up), linf(ue, up))
    print(f'{name:24s}  rel L2 = {store[name][0]:.6e}   L inf = {store[name][1]:.6e}')


strip = 0.01
metrics = {}
region_metrics('x-layer strip', np.linspace(0, strip, 201), np.linspace(0, 1, 51), metrics)
region_metrics('y-layer strip', np.linspace(0, 1, 51), np.linspace(0, strip, 201), metrics)
region_metrics('corner region', np.linspace(0, strip, 201), np.linspace(0, strip, 201), metrics)
region_metrics('global (all grids)',
               np.concatenate([np.linspace(0, 1, 101),
                               np.linspace(0, strip, 201),
                               np.linspace(0, 1, 51)]),
               np.concatenate([np.linspace(0, 1, 101),
                               np.linspace(0, 1, 51),
                               np.linspace(0, strip, 201)]), metrics)

with open(os.path.join(out_dir, 'test error.txt'), 'a', encoding='utf-8') as f:
    f.write(f'adam epochs: {epochs_adam}\n')
    # f.write(f'training time (s): {toc_adam - tic:.1f}\nbest loss: {best:.3e}\n\n')
    for k, v in metrics.items():
        f.write(f'{k}: rel L2 = {v[0]:.6e}, L inf = {v[1]:.6e}\n')
#
# plot_dir = os.path.join(out_dir, 'plots')
#
# plt.figure()
# plt.plot(loss_hist, 'r', label='total loss (Adam phase)')
# plt.yscale('log')
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(plot_dir, 'loss.png'), dpi=300)
# plt.close()

# Slice plots
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
slices = [('x', 0.5), ('x', 0.005), ('y', 0.5), ('y', 0.005)]
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
    ax.plot(s, ue, 'b-', label='exact', linewidth=2)
    ax.plot(s, up, 'r--', label='leading-order PVD-Net', linewidth=2)
    ax.legend(loc='center right')

    axins = inset_axes(ax, width='42%', height='55%', loc='upper right')
    m = s <= strip
    axins.plot(s[m], ue[m], 'b-', linewidth=2)
    axins.plot(s[m], up[m], 'r--', linewidth=2)
    axins.set_xlim(0.0, strip)
    mark_inset(ax, axins, loc1=3, loc2=4,
               fc=(0.5, 0.5, 0.5, 0.3),
               ec=(0.5, 0.5, 0.5, 0.3), lw=1.0)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'slices.png'), dpi=300)
plt.close()



# ----------------------------------------------------------------------
# 2D full-field visualization: exact solution / prediction / abs. error
# ----------------------------------------------------------------------
n_vis = 501
x_vis = np.linspace(0.0, 1.0, n_vis)
y_vis = np.linspace(0.0, 1.0, n_vis)
XX_vis, YY_vis = np.meshgrid(x_vis, y_vis)

U_exact_vis = u_exact_np(XX_vis.ravel(), YY_vis.ravel()).reshape(XX_vis.shape)
U_pred_vis = predict_np(XX_vis.ravel(), YY_vis.ravel()).reshape(XX_vis.shape)
U_error_vis = np.abs(U_exact_vis - U_pred_vis)

u_vmin = float(min(U_exact_vis.min(), U_pred_vis.min()))
u_vmax = float(max(U_exact_vis.max(), U_pred_vis.max()))
extent_vis = [0.0, 1.0, 0.0, 1.0]

# 1) exact solution
fig, ax = plt.subplots(figsize=(6.6, 5.4))
im = ax.imshow(U_exact_vis, origin='lower', extent=extent_vis, aspect='equal',
               cmap='jet', vmin=u_vmin, vmax=u_vmax)
ax.set_xlabel(r'$x$', fontsize=14)
ax.set_ylabel(r'$y$', fontsize=14)
ax.set_title('Exact solution', fontsize=15)
fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'exact_solution_2d.png'), dpi=600, bbox_inches='tight')
plt.close()

# 2) predicted solution
fig, ax = plt.subplots(figsize=(6.6, 5.4))
im = ax.imshow(U_pred_vis, origin='lower', extent=extent_vis, aspect='equal',
               cmap='jet', vmin=u_vmin, vmax=u_vmax)
ax.set_xlabel(r'$x$', fontsize=14)
ax.set_ylabel(r'$y$', fontsize=14)
ax.set_title('Leading-order 2D PVD-Net prediction', fontsize=15)
fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'predicted_solution_2d.png'), dpi=600, bbox_inches='tight')
plt.close()

# 3) pointwise absolute error
fig, ax = plt.subplots(figsize=(6.6, 5.4))
im = ax.imshow(U_error_vis, origin='lower', extent=extent_vis,
               aspect='equal', cmap='viridis')
ax.set_xlabel(r'$x$', fontsize=14)
ax.set_ylabel(r'$y$', fontsize=14)
ax.set_title(r'Absolute error $|u_{exact}-u_{pred}|$', fontsize=15)
fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'absolute_error_2d.png'), dpi=600, bbox_inches='tight')
plt.close()

# 4) combined figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
im0 = axes[0].imshow(U_exact_vis, origin='lower', extent=extent_vis, aspect='equal',
                     cmap='jet', vmin=u_vmin, vmax=u_vmax)
axes[0].set_title('Exact solution', fontsize=14)
axes[0].set_xlabel(r'$x$', fontsize=13)
axes[0].set_ylabel(r'$y$', fontsize=13)
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(U_pred_vis, origin='lower', extent=extent_vis, aspect='equal',
                     cmap='jet', vmin=u_vmin, vmax=u_vmax)
axes[1].set_title('Leading-order 2D PVD-Net prediction', fontsize=14)
axes[1].set_xlabel(r'$x$', fontsize=13)
axes[1].set_ylabel(r'$y$', fontsize=13)
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

im2 = axes[2].imshow(U_error_vis, origin='lower', extent=extent_vis,
                     aspect='equal', cmap='viridis')
axes[2].set_title(r'Absolute error $|u_{exact}-u_{pred}|$', fontsize=14)
axes[2].set_xlabel(r'$x$', fontsize=13)
axes[2].set_ylabel(r'$y$', fontsize=13)
fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'exact_prediction_error_2d.png'), dpi=600, bbox_inches='tight')
plt.close()

print('2D full-field plots written to', plot_dir)
print('results written to', out_dir)






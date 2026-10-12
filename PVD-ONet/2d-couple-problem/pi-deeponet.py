import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RegularGridInterpolator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------
# problem / operator setting
# ----------------------------------------------------------------------
eps = 1.0e-3
kx = np.pi / 2.0
num = 1100
ntrain = 1000
ntest = 100
batchsize = 50
net_width = 200
learning_rate = 1.0e-3
epochs = 10000

script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, 'baseline-pi-deeponet-2d')
plot_dir = os.path.join(out_dir, 'plots')
model_dir = os.path.join(out_dir, 'model')
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

path_best = os.path.join(model_dir, 'best_model.pth')

# ----------------------------------------------------------------------
# standard PI-DeepONet
# ----------------------------------------------------------------------
class deeponet(nn.Module):
    def __init__(self, branch_size, trunk_size):
        super().__init__()
        self.branch_size = branch_size
        self.trunk_size = trunk_size
        self.branch_net_ls = self.branch_net()
        self.trunk_net_ls = self.trunk_net()

    def branch_net(self):
        layers = []
        prev_size = self.branch_size[0]
        for hidden_size in self.branch_size[1:-1]:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.SiLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, self.branch_size[-1]))
        return nn.Sequential(*layers)

    def trunk_net(self):
        layers = []
        prev_size = self.trunk_size[0]
        for hidden_size in self.trunk_size[1:-1]:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.SiLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, self.trunk_size[-1]))
        return nn.Sequential(*layers)

    def forward(self, x_func, x_loc):
        y_func = self.branch_net_ls(x_func)
        y_loc = self.trunk_net_ls(x_loc)
        return torch.einsum('bi,bi->b', y_func, y_loc)


branch_size = [1, net_width, net_width, net_width, net_width, net_width, net_width]
trunk_size = [2, net_width, net_width, net_width, net_width, net_width, net_width]

net = deeponet(branch_size, trunk_size).to(device)

def init_normal(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

net.apply(init_normal)

total_params = sum(p.numel() for p in net.parameters())
print(f'Total parameters: {total_params:,}')

# ----------------------------------------------------------------------
# original PDE and reference solution
# ----------------------------------------------------------------------
def u0_exact_np(x, y):
    return np.cos(kx*x)*(1.0-np.exp(-2.0*x/eps))*(1.0-y**3)*(1.0-np.exp(-3.0*y/eps))

def f0_torch(x, y):
    p = torch.cos(kx*x)
    s = torch.sin(kx*x)
    q = 1.0-y**3
    return (x+2.0)*kx*s*q + 3.0*(y**3+3.0)*y**2*p + p*q

def grad(u, v):
    return torch.autograd.grad(
        u, v,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        only_inputs=True
    )[0]

def tensor(x):
    return torch.tensor(x, dtype=torch.float32)

def split_tensor(a):
    a = tensor(a)
    return a[:ntrain], a[-ntest:]

# ----------------------------------------------------------------------
# training and testing samples
# ----------------------------------------------------------------------
c_all = np.random.uniform(0, 1, (num, 1)).astype(np.float32)
c_train_np = c_all[:ntrain].copy()
c_test_np = c_all[-ntest:].copy()
c_train, c_test = split_tensor(c_all)

XY_train, XY_test = split_tensor(np.random.uniform(0, 1, (num, 2)))

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(c_train, XY_train),
    batch_size=batchsize,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(c_test, XY_test),
    batch_size=batchsize,
    shuffle=False
)

# ----------------------------------------------------------------------
# PDE and boundary losses
# ----------------------------------------------------------------------
def loss_pde(c, xy):
    c = c.to(device).view(-1, 1)
    xy = xy.to(device).detach().requires_grad_(True)

    x = xy[:, 0:1]
    y = xy[:, 1:2]

    u = net(c, xy).reshape(-1, 1)

    du = grad(u, xy)
    du_dx = du[:, 0:1]
    du_dy = du[:, 1:2]

    d2u_dx2 = grad(du_dx, xy)[:, 0:1]
    d2u_dy2 = grad(du_dy, xy)[:, 1:2]

    residual = (
        -eps*(d2u_dx2+d2u_dy2)
        -(x+2.0)*du_dx
        -(y**3+3.0)*du_dy
        +u
        -f0_torch(x, y)
    )

    return torch.mean(residual**2)

def loss_bc(c, xy):
    c = c.to(device).view(-1, 1)
    xy = xy.to(device)

    x = xy[:, 0:1]
    y = xy[:, 1:2]

    bc_xy = torch.cat([
        torch.cat([torch.zeros_like(y), y], dim=1),
        torch.cat([torch.ones_like(y), y], dim=1),
        torch.cat([x, torch.zeros_like(x)], dim=1),
        torch.cat([x, torch.ones_like(x)], dim=1)
    ], dim=0)

    c_rep = c.repeat(4, 1)
    u_bc = net(c_rep, bc_xy).reshape(-1, 1)

    return torch.mean((u_bc-c_rep)**2)

def loss_components(batch):
    c, xy = batch
    return {
        'pde': loss_pde(c, xy),
        'bc': loss_bc(c, xy)
    }

def total_loss_from_components(L):
    return L['pde'] + L['bc']

# ----------------------------------------------------------------------
# training
# ----------------------------------------------------------------------
optimizer = optim.Adam(
    net.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.99),
    eps=1e-15
)

train_loss_ls = np.zeros(epochs)
test_loss_ls = np.zeros(epochs)

best_loss = float('inf')
best_epoch = 0
tic = time.time()

for ep in range(epochs):
    net.train()
    train_sum = {'total': 0.0, 'pde': 0.0, 'bc': 0.0}

    for batch in train_loader:
        optimizer.zero_grad()

        L = loss_components(batch)
        loss = total_loss_from_components(L)

        loss.backward()
        optimizer.step()

        train_sum['total'] += loss.item()
        for k in L:
            train_sum[k] += L[k].item()

    for k in train_sum:
        train_sum[k] /= len(train_loader)

    train_loss_ls[ep] = train_sum['total']

    net.eval()
    test_sum = {'total': 0.0, 'pde': 0.0, 'bc': 0.0}

    for batch in test_loader:
        L = loss_components(batch)
        loss = total_loss_from_components(L)

        test_sum['total'] += loss.item()
        for k in L:
            test_sum[k] += L[k].item()

    for k in test_sum:
        test_sum[k] /= len(test_loader)

    test_loss_ls[ep] = test_sum['total']

    if test_sum['total'] < best_loss:
        best_loss = test_sum['total']
        best_epoch = ep + 1
        torch.save(net.state_dict(), path_best)

        with open(os.path.join(model_dir, 'best epoch.txt'), 'a', encoding='utf-8') as f:
            f.write(f'best epoch: {best_epoch}\nbest test loss: {best_loss:.12e}\n')

    if ep % 1000 == 0:
        print(
            f"Train Epoch: {ep}  Loss: {train_sum['total']:.5e} "
            f"test Loss: {test_sum['total']:.5e}  best: {best_loss:.5e}"
        )
        print('-'*100)

toc = time.time()

net.load_state_dict(
    torch.load(path_best, map_location=device, weights_only=True)
)
net.eval()

plt.figure(figsize=(7, 5))
plt.plot(train_loss_ls, 'r', label='train loss')
plt.plot(test_loss_ls, 'g', label='test loss')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'loss.png'), dpi=600)
plt.close()

# ----------------------------------------------------------------------
# PI-DeepONet prediction
# ----------------------------------------------------------------------
def predict_np(c_value, xnp, ynp, chunk=20000):
    xnp = np.asarray(xnp).ravel()
    ynp = np.asarray(ynp).ravel()
    out = []

    with torch.no_grad():
        for s in range(0, len(xnp), chunk):
            e = min(s+chunk, len(xnp))

            x = torch.tensor(
                xnp[s:e, None], dtype=torch.float32, device=device
            )
            y = torch.tensor(
                ynp[s:e, None], dtype=torch.float32, device=device
            )
            c = torch.full(
                (e-s, 1), float(c_value),
                dtype=torch.float32, device=device
            )

            out.append(
                net(c, torch.cat([x, y], dim=1)).cpu().numpy().ravel()
            )

    return np.concatenate(out)

# ----------------------------------------------------------------------
# numerical reference: finite difference method
# ----------------------------------------------------------------------
def shishkin_grid(n, eps, beta, tau=2.0):
    if n % 2 == 0:
        raise ValueError('reference grid size must be odd')

    m = (n-1)//2
    sigma = min(0.5, tau*eps*np.log(n)/beta)

    return np.concatenate([
        np.linspace(0.0, sigma, m+1),
        np.linspace(sigma, 1.0, m+1)[1:]
    ])

def build_reference_phi(n=401):
    xr = shishkin_grid(n, eps, 2.0)
    yr = shishkin_grid(n, eps, 3.0)

    nx = len(xr)
    ny = len(yr)
    n_unknown = (nx-2)*(ny-2)

    def ind(i, j):
        return (j-1)*(nx-2)+(i-1)

    rows, cols, vals = [], [], []
    rhs = np.zeros(n_unknown)

    for j in range(1, ny-1):
        yj = yr[j]
        hym = yr[j]-yr[j-1]
        hyp = yr[j+1]-yr[j]

        cym = -2.0*eps/(hym*(hym+hyp))
        cyp = -2.0*eps/(hyp*(hym+hyp))
        cy0 = 2.0*eps/(hym*hyp)

        by = -(yj**3+3.0)
        cyp += by/hyp
        cy0 -= by/hyp

        for i in range(1, nx-1):
            xx = xr[i]
            hxm = xr[i]-xr[i-1]
            hxp = xr[i+1]-xr[i]

            cxm = -2.0*eps/(hxm*(hxm+hxp))
            cxp = -2.0*eps/(hxp*(hxm+hxp))
            cx0 = 2.0*eps/(hxm*hxp)

            bx = -(xx+2.0)
            cxp += bx/hxp
            cx0 -= bx/hxp

            k = ind(i, j)

            rows.append(k)
            cols.append(k)
            vals.append(cx0+cy0+1.0)

            if i == 1:
                rhs[k] -= cxm
            else:
                rows.append(k)
                cols.append(ind(i-1, j))
                vals.append(cxm)

            if i == nx-2:
                rhs[k] -= cxp
            else:
                rows.append(k)
                cols.append(ind(i+1, j))
                vals.append(cxp)

            if j == 1:
                rhs[k] -= cym
            else:
                rows.append(k)
                cols.append(ind(i, j-1))
                vals.append(cym)

            if j == ny-2:
                rhs[k] -= cyp
            else:
                rows.append(k)
                cols.append(ind(i, j+1))
                vals.append(cyp)

    phi_inner = spsolve(
        csr_matrix(
            (vals, (rows, cols)),
            shape=(n_unknown, n_unknown)
        ),
        rhs
    )

    phi = np.ones((ny, nx))
    phi[1:-1, 1:-1] = phi_inner.reshape(ny-2, nx-2)

    return xr, yr, phi

print('Building numerical reference solution...')

x_ref, y_ref, phi_ref = build_reference_phi(401)

phi_interp = RegularGridInterpolator(
    (y_ref, x_ref),
    phi_ref,
    method='linear',
    bounds_error=False,
    fill_value=None
)

def reference_np(c_value, xnp, ynp):
    xnp = np.asarray(xnp)
    ynp = np.asarray(ynp)

    pts = np.stack(
        [ynp.ravel(), xnp.ravel()],
        axis=-1
    )

    phi = phi_interp(pts).reshape(xnp.shape)

    return u0_exact_np(xnp, ynp) + float(c_value)*phi

def rel_l2(ue, up):
    return np.linalg.norm(ue-up)/np.linalg.norm(ue)

def linf(ue, up):
    return np.max(np.abs(ue-up))

def region_error_for_c(c_value, xg, yg):
    XX, YY = np.meshgrid(xg, yg)

    ue = reference_np(c_value, XX, YY).ravel()
    up = predict_np(c_value, XX.ravel(), YY.ravel())

    return rel_l2(ue, up), linf(ue, up)

# ----------------------------------------------------------------------
# error calculation: four regions
# ----------------------------------------------------------------------
strip = 0.01

regions = {
    'x-layer strip': (
        np.linspace(0, strip, 201),
        np.linspace(0, 1, 51)
    ),
    'y-layer strip': (
        np.linspace(0, 1, 51),
        np.linspace(0, strip, 201)
    ),
    'corner region': (
        np.linspace(0, strip, 201),
        np.linspace(0, strip, 201)
    ),
    'global (all grids)': (
        np.concatenate([
            np.linspace(0, 1, 101),
            np.linspace(0, strip, 201),
            np.linspace(0, 1, 51)
        ]),
        np.concatenate([
            np.linspace(0, 1, 101),
            np.linspace(0, 1, 51),
            np.linspace(0, strip, 201)
        ])
    )
}

all_errors = {name: [] for name in regions}

for i, cv in enumerate(c_test_np[:, 0]):
    for name, (xg, yg) in regions.items():
        all_errors[name].append(
            region_error_for_c(cv, xg, yg)
        )

    if (i+1) % 10 == 0:
        print(f'evaluated {i+1}/{ntest} test boundary values')

mean_errors = {}

for name in regions:
    arr = np.asarray(all_errors[name])

    mean_errors[name] = (
        arr[:, 0].mean(),
        arr[:, 1].mean()
    )

    print(
        f'{name:24s}  '
        f'mean rel L2 = {mean_errors[name][0]:.6e}   '
        f'mean L inf = {mean_errors[name][1]:.6e}'
    )

with open(
    os.path.join(out_dir, 'test error.txt'),
    'a', encoding='utf-8'
) as f:

    f.write(
        f'epochs: {epochs}\n'
        f'parameters: {total_params}\n'
        f'training time (s): {toc-tic:.1f}\n'
        f'best epoch: {best_epoch}\n'
        f'best test physics loss: {best_loss:.6e}\n\n'
    )

    for name, v in mean_errors.items():
        f.write(
            f'{name}: mean rel L2 = {v[0]:.6e}, '
            f'mean L inf = {v[1]:.6e}\n'
        )

np.savez(
    os.path.join(out_dir, 'test_errors.npz'),
    c_test=c_test_np[:, 0],
    **{
        name.replace(' ', '_').replace('(', '').replace(')', ''):
        np.asarray(v)
        for name, v in all_errors.items()
    }
)

# ----------------------------------------------------------------------
# 2D field visualization
# ----------------------------------------------------------------------
sel = [10, 11, 12, 13, 14, 15]
n_vis = 301

x_vis = np.linspace(0, 1, n_vis)
y_vis = np.linspace(0, 1, n_vis)
XXv, YYv = np.meshgrid(x_vis, y_vis)

for ind in sel:
    cv = float(c_test_np[ind, 0])

    Ue = reference_np(cv, XXv, YYv)
    Up = predict_np(
        cv, XXv.ravel(), YYv.ravel()
    ).reshape(XXv.shape)

    Er = np.abs(Ue-Up)

    vmin = min(Ue.min(), Up.min())
    vmax = max(Ue.max(), Up.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    im0 = axes[0].imshow(
        Ue, origin='lower',
        extent=[0, 1, 0, 1],
        aspect='equal',
        cmap='jet', vmin=vmin, vmax=vmax
    )
    axes[0].set_title(rf'Numerical reference, $c={cv:.3f}$')
    axes[0].set_xlabel(r'$x$')
    axes[0].set_ylabel(r'$y$')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(
        Up, origin='lower',
        extent=[0, 1, 0, 1],
        aspect='equal',
        cmap='jet', vmin=vmin, vmax=vmax
    )
    axes[1].set_title('Baseline PI-DeepONet')
    axes[1].set_xlabel(r'$x$')
    axes[1].set_ylabel(r'$y$')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(
        Er, origin='lower',
        extent=[0, 1, 0, 1],
        aspect='equal',
        cmap='viridis'
    )
    axes[2].set_title('Absolute error')
    axes[2].set_xlabel(r'$x$')
    axes[2].set_ylabel(r'$y$')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()

    plt.savefig(
        os.path.join(plot_dir, f'test_{ind:03d}_field.png'),
        dpi=600, bbox_inches='tight'
    )
    plt.close()

# ----------------------------------------------------------------------
# 1D slice visualization
# ----------------------------------------------------------------------
cv = float(c_test_np[0, 0])

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
        ue = reference_np(cv, s, np.full_like(s, val))
        up = predict_np(cv, s, np.full_like(s, val))
        ax.set_xlabel(r'$x$')
        ax.set_title(rf'Slice $y={val}$')
    else:
        ue = reference_np(cv, np.full_like(s, val), s)
        up = predict_np(cv, np.full_like(s, val), s)
        ax.set_xlabel(r'$y$')
        ax.set_title(rf'Slice $x={val}$')

    ax.plot(
        s, ue, 'b-',
        label='Numerical reference',
        linewidth=2
    )
    ax.plot(
        s, up, 'r--',
        label='Baseline PI-DeepONet',
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

    axins.plot(s[m], ue[m], 'b-', linewidth=2)
    axins.plot(s[m], up[m], 'r--', linewidth=2)
    axins.set_xlim(0.0, strip)

    mark_inset(
        ax, axins,
        loc1=3, loc2=4,
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

print('results written to', out_dir)

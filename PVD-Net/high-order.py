import os
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
net_width = 50
lr = 1.0e-4

epochs_adam =100000

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

    fx = ((eps * kx ** 2 * p + (x + 2.0) * kx * s) * ax
          + 4.0 * kx * s * ex
          - (2.0 * x / eps) * p * ex)

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


def f1_torch(x, y):
    """O(eps) outer source coefficient."""
    p = torch.cos(kx * x)
    q = 1.0 - y ** 3
    return kx ** 2 * p * q + 6.0 * y * p


def f_exact_torch(x, y):
    """Stable torch evaluation of the complete manufactured source."""
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


# ----------------------------------------------------------------------
# networks: unchanged nine-network framework
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


net_o0 = Net().to(device)
net_o1 = Net().to(device)
net_x0, net_xc, net_x1 = (Net().to(device) for _ in range(3))
net_y0, net_yc, net_y1 = (Net().to(device) for _ in range(3))
net_c = Net().to(device)

ALL_NETS = [net_o0, net_o1,
            net_x0, net_xc, net_x1,
            net_y0, net_yc, net_y1,
            net_c]
def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


print('\n================ Parameter Count ================')

for name, net in zip(
        ['u_o0', 'u_o1',
         'U_x0', 'U_xc', 'U_x1',
         'U_y0', 'U_yc', 'U_y1',
         'U_cor'],
        ALL_NETS):

    n_param = count_parameters(net)
    print(f'{name:8s}: {n_param:,} parameters')

total_params = sum(
    count_parameters(net)
    for net in ALL_NETS
)

print('-------------------------------------------------')
print(f'Total trainable parameters: {total_params:,}')
print('=================================================')

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


for net in ALL_NETS:
    net.apply(init_normal)


# ----------------------------------------------------------------------
# collocation / boundary / matching points
# Same PDE counts as the leading-order independent-BC script.
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

# Outer reduced inflow boundaries x=1 and y=1.
Y_OB = t(np.random.uniform(0, 1, n_bc_o))
X_OB = t(np.random.uniform(0, 1, n_bc_o))

# x-inner: physical wall xi=0 and matching line xi=xi0.
Y_XB = t(np.random.uniform(0, 1, n_bc_i))

# y-inner: physical wall eta=0 and matching line eta=xi0.
X_YB = t(np.random.uniform(0, 1, n_bc_i))

# Corner: physical corner edges xi=0, eta=0 and far edges xi=xi0, eta=eta0.
XI_CB = t(np.random.uniform(0, xi0, n_bc_c))
ETA_CB = t(np.random.uniform(0, xi0, n_bc_c))


# ----------------------------------------------------------------------
# differential helpers and high-order building blocks
# ----------------------------------------------------------------------
def grad(u, v):
    return torch.autograd.grad(u, v, grad_outputs=torch.ones_like(u),
                               create_graph=True, only_inputs=True)[0]


def outer_total(x, y):
    return net_o0(x, y) + eps * net_o1(x, y)


def inner_x_total(xi, y):
    """Ux = U_x0 + eps*(xi*U_xc + U_x1), unchanged Van Dyke ansatz."""
    return net_x0(xi, y) + eps * (xi * net_xc(xi, y) + net_x1(xi, y))


def inner_y_total(x, eta):
    """Uy = U_y0 + eps*(eta*U_yc + U_y1), unchanged Van Dyke ansatz."""
    return net_y0(x, eta) + eps * (eta * net_yc(x, eta) + net_y1(x, eta))


def x_matching_extension(x, y):
    """High-order matching-value extension Mx(x,y) from xi=xi0."""
    xi0_t = torch.full_like(y, xi0)
    return (net_x0(xi0_t, y)
            + x * net_xc(xi0_t, y)
            + eps * net_x1(xi0_t, y))


def y_matching_extension(x, y):
    """High-order matching-value extension My(x,y) from eta=eta0."""
    eta0_t = torch.full_like(x, xi0)
    return (net_y0(x, eta0_t)
            + y * net_yc(x, eta0_t)
            + eps * net_y1(x, eta0_t))


# ----------------------------------------------------------------------
# independent asymptotic-subproblem losses
# ----------------------------------------------------------------------
def loss_outer_pde():
    """O(1) and O(eps) outer equations."""
    x = X_O.clone().requires_grad_(True)
    y = Y_O.clone().requires_grad_(True)
    u0 = net_o0(x, y)
    u1 = net_o1(x, y)

    u0_x = grad(u0, x)
    u0_y = grad(u0, y)
    u0_xx = grad(u0_x, x)
    u0_yy = grad(u0_y, y)

    r0 = (-(x + 2.0) * u0_x
          - (y ** 3 + 3.0) * u0_y
          + u0 - f0_torch(x, y))

    u1_x = grad(u1, x)
    u1_y = grad(u1, y)
    r1 = (-(x + 2.0) * u1_x
          - (y ** 3 + 3.0) * u1_y
          + u1 - u0_xx - u0_yy - f1_torch(x, y))

    return (r0 ** 2).mean() + (r1 ** 2).mean()


def loss_outer_bc():
    """Order-by-order homogeneous BC on the reduced inflow x=1 union y=1."""
    one_y = torch.ones_like(Y_OB)
    one_x = torch.ones_like(X_OB)

    l0 = ((net_o0(one_y, Y_OB) ** 2).mean()
          + (net_o0(X_OB, one_x) ** 2).mean())
    l1 = ((net_o1(one_y, Y_OB) ** 2).mean()
          + (net_o1(X_OB, one_x) ** 2).mean())
    return l0 + l1


def loss_inner_x_pde():
    """Exact transformed nonhomogeneous x-inner equation for the total Ux."""
    xi = XI_I.clone().requires_grad_(True)
    y = Y_I.clone().requires_grad_(True)
    U = inner_x_total(xi, y)

    U_xi = grad(U, xi)
    U_y = grad(U, y)
    x_phys = eps * xi

    r = (grad(U_xi, xi)
         + (2.0 + eps * xi) * U_xi
         + eps * (y ** 3 + 3.0) * U_y
         - eps * U
         + (eps ** 2) * grad(U_y, y)
         + eps * f_exact_torch(x_phys, y))
    return (r ** 2).mean()


def loss_inner_x_bc():
    """
    Independent x-inner closure.

    Physical wall, order by order:
        U_x0(0,y)=0, U_x1(0,y)=0.
    No wall datum is imposed on U_xc: xi*U_xc vanishes identically at xi=0.

    Van Dyke matching at xi=xi0:
        U_x0 = u_o0(0,y),
        U_xc = d_x u_o0(0,y),
        U_x1 = u_o1(0,y).
    """
    y = Y_XB
    xi_zero = torch.zeros_like(y)
    xi_far = torch.full_like(y, xi0)

    wall0 = (net_x0(xi_zero, y) ** 2).mean()
    wall1 = (net_x1(xi_zero, y) ** 2).mean()

    xz = torch.zeros_like(y).requires_grad_(True)
    u0_at_wall = net_o0(xz, y)
    du0dx = grad(u0_at_wall, xz)

    match0 = ((net_x0(xi_far, y) - u0_at_wall) ** 2).mean()
    matchc = ((net_xc(xi_far, y) - du0dx) ** 2).mean()
    match1 = ((net_x1(xi_far, y) - net_o1(torch.zeros_like(y), y)) ** 2).mean()

    return wall0 + wall1 + match0 + matchc + match1


def loss_inner_y_pde():
    """Exact transformed nonhomogeneous y-inner equation for the total Uy."""
    x = X_J.clone().requires_grad_(True)
    eta = ETA_J.clone().requires_grad_(True)
    V = inner_y_total(x, eta)

    V_eta = grad(V, eta)
    V_x = grad(V, x)
    y_phys = eps * eta

    r = (grad(V_eta, eta)
         + (3.0 + (eps ** 3) * eta ** 3) * V_eta
         + eps * (x + 2.0) * V_x
         - eps * V
         + (eps ** 2) * grad(V_x, x)
         + eps * f_exact_torch(x, y_phys))
    return (r ** 2).mean()


def loss_inner_y_bc():
    """
    Independent y-inner closure.

    Physical wall, order by order:
        U_y0(x,0)=0, U_y1(x,0)=0.
    No wall datum is imposed on U_yc because eta*U_yc vanishes at eta=0.

    Van Dyke matching at eta=eta0:
        U_y0 = u_o0(x,0),
        U_yc = d_y u_o0(x,0),
        U_y1 = u_o1(x,0).
    """
    x = X_YB
    eta_zero = torch.zeros_like(x)
    eta_far = torch.full_like(x, xi0)

    wall0 = (net_y0(x, eta_zero) ** 2).mean()
    wall1 = (net_y1(x, eta_zero) ** 2).mean()

    yz = torch.zeros_like(x).requires_grad_(True)
    u0_at_wall = net_o0(x, yz)
    du0dy = grad(u0_at_wall, yz)

    match0 = ((net_y0(x, eta_far) - u0_at_wall) ** 2).mean()
    matchc = ((net_yc(x, eta_far) - du0dy) ** 2).mean()
    match1 = ((net_y1(x, eta_far) - net_o1(x, torch.zeros_like(x))) ** 2).mean()

    return wall0 + wall1 + match0 + matchc + match1


def corner_base(xi, eta):
    """
    Outer + x-inner correction + y-inner correction in the corner rectangle,
    excluding the corner network itself.
    """
    x = eps * xi
    y = eps * eta
    Ux = inner_x_total(xi, y)
    Uy = inner_y_total(x, eta)
    Mx = x_matching_extension(x, y)
    My = y_matching_extension(x, y)
    return outer_total(x, y) + (Ux - Mx) + (Uy - My)


def loss_corner_pde():
    """
    Independent corner-correction equation.

    Let B be the outer+x-inner+y-inner base approximation.  In stretched
    corner variables the exact operator is

        Lc Z = Z_xixi + Z_etaeta
               + (2+eps*xi) Z_xi
               + (3+eps^3*eta^3) Z_eta - eps Z.

    Since the full approximation is B + Wc, Wc satisfies

        Lc Wc = -[Lc B + eps*f(eps*xi,eps*eta)].

    This is algebraically equivalent to imposing the transformed PDE on
    B+Wc, but it identifies an explicit inhomogeneous PDE for the corner
    correction itself instead of treating the whole composite as the unknown.
    """
    xi = XI_C.clone().requires_grad_(True)
    eta = ETA_C.clone().requires_grad_(True)
    x = eps * xi
    y = eps * eta

    B = corner_base(xi, eta)
    B_xi = grad(B, xi)
    B_eta = grad(B, eta)
    L_B = (grad(B_xi, xi)
           + grad(B_eta, eta)
           + (2.0 + eps * xi) * B_xi
           + (3.0 + (eps ** 3) * eta ** 3) * B_eta
           - eps * B)

    W = net_c(xi, eta)
    W_xi = grad(W, xi)
    W_eta = grad(W, eta)
    L_W = (grad(W_xi, xi)
           + grad(W_eta, eta)
           + (2.0 + eps * xi) * W_xi
           + (3.0 + (eps ** 3) * eta ** 3) * W_eta
           - eps * W)

    rhs = -(L_B + eps * f_exact_torch(x, y))
    r = L_W - rhs
    return (r ** 2).mean()


def loss_corner_bc():
    """
    Dirichlet closure of the corner correction on all four sides.

    On xi=0, the x-inner profile already enforces the x=0 wall.  The remaining
    y-inner correction must therefore be cancelled by the corner correction:

        Wc(0,eta) = My(0,eps*eta) - Uy(0,eta).

    On eta=0 analogously:

        Wc(xi,0) = Mx(eps*xi,0) - Ux(xi,0).

    On the two far sides of the truncated stretched corner domain:

        Wc(xi0,eta)=0, Wc(xi,eta0)=0.
    """
    # xi = 0 edge
    eta = ETA_CB
    xi_zero = torch.zeros_like(eta)
    x_zero = torch.zeros_like(eta)
    y_phys = eps * eta
    target_xi_zero = (y_matching_extension(x_zero, y_phys)
                      - inner_y_total(x_zero, eta))
    l_xi_zero = ((net_c(xi_zero, eta) - target_xi_zero) ** 2).mean()

    # eta = 0 edge
    xi = XI_CB
    eta_zero = torch.zeros_like(xi)
    x_phys = eps * xi
    y_zero = torch.zeros_like(xi)
    target_eta_zero = (x_matching_extension(x_phys, y_zero)
                       - inner_x_total(xi, y_zero))
    l_eta_zero = ((net_c(xi, eta_zero) - target_eta_zero) ** 2).mean()

    # far matching/decay edges
    xi_far = torch.full_like(eta, xi0)
    eta_far = torch.full_like(xi, xi0)
    l_xi_far = (net_c(xi_far, eta) ** 2).mean()
    l_eta_far = (net_c(xi, eta_far) ** 2).mean()

    return l_xi_zero + l_eta_zero + l_xi_far + l_eta_far


# ----------------------------------------------------------------------
# uniformly valid high-order composite: unchanged v1 construction
# ----------------------------------------------------------------------
def composite(x, y):
    xi = torch.clamp(x / eps, max=xi0)
    eta = torch.clamp(y / eps, max=xi0)
    Mx = x_matching_extension(eps * xi, y)
    My = y_matching_extension(x, eps * eta)
    Ux = inner_x_total(xi, y)
    Uy = inner_y_total(x, eta)
    Wc = net_c(xi, eta)
    return outer_total(x, y) + (Ux - Mx) + (Uy - My) + Wc


def loss_components():
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
    # Same convention as the independent-BC leading-order code:
    # inner/corner PDE residuals carry weight 10; all BC/matching losses weight 1.
    return (L['outer_pde'] + L['outer_bc']
            + 10*(L['x_pde'] + L['y_pde'] + L['corner_pde'])
            + L['x_bc'] + L['y_bc'] + L['corner_bc'])


# ----------------------------------------------------------------------
# training
# ----------------------------------------------------------------------
params = []
for net in ALL_NETS:
    params += list(net.parameters())

script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, 'high-order')
os.makedirs(os.path.join(out_dir, 'plots'), exist_ok=True)
os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)
NET_NAMES = ['o0', 'o1', 'x0', 'xc', 'x1', 'y0', 'yc', 'y1', 'c']
PATHS = [os.path.join(out_dir, 'model', f'best_{n}.pth') for n in NET_NAMES]


def save_best():
    for net, p in zip(ALL_NETS, PATHS):
        torch.save(net.state_dict(), p)


print(f'Adam epochs = {epochs_adam}, output = {out_dir}')

tic = time.time()
best = float('inf')
loss_hist = []

optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.99), eps=1e-15)
for epoch in range(epochs_adam):
    optimizer.zero_grad()
    L = total_loss()
    L.backward()
    optimizer.step()
    v = L.item()
    loss_hist.append(v)
    if v < best:
        best = v
        save_best()
    if epoch % 10000 == 0:
        print(f'[Adam] Epoch {epoch:6d}  total {v:.3e}  best {best:.3e}')

toc_adam = time.time()
print(f'Adam phase finished, elapsed {toc_adam - tic:.1f} s, best {best:.3e}')




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
    f.write(f'training time (s): {toc_adam - tic:.1f}\nbest loss: {best:.3e}\n\n')
    for k, v in metrics.items():
        f.write(f'{k}: rel L2 = {v[0]:.6e}, L inf = {v[1]:.6e}\n')

#
plot_dir = os.path.join(out_dir, 'plots')

plt.figure()
plt.plot(loss_hist, 'r', label='total loss (Adam phase)')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'loss.png'), dpi=300)
plt.close()

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
    ax.plot(s, up, 'r--', label='high-order PVD-Net', linewidth=2)
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

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
panels = [('full domain', np.linspace(0, 1, 301), np.linspace(0, 1, 301)),
          ('x-layer strip', np.linspace(0, strip, 301), np.linspace(0, 1, 101)),
          ('corner region', np.linspace(0, strip, 301), np.linspace(0, strip, 301))]
for ax, (name, xg, yg) in zip(axes, panels):
    XX, YY = np.meshgrid(xg, yg)
    ue = u_exact_np(XX.ravel(), YY.ravel())
    up = predict_np(XX.ravel(), YY.ravel())
    err = np.abs(ue - up).reshape(XX.shape)
    im = ax.imshow(np.log10(err + 1e-16), origin='lower',
                   extent=[xg[0], xg[-1], yg[0], yg[-1]],
                   aspect='auto', cmap='jet')
    ax.set_title(f'log10 |error|  ({name})')
    fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'error_heatmap.png'), dpi=300)
plt.close()
# ----------------------------------------------------------------------
# Local zoom plots:
#   Row 1: x-boundary layer region
#   Row 2: y-boundary layer region
#   Row 3: corner-layer region
#
#   Col 1: Exact
#   Col 2: Prediction
#   Col 3: Absolute error
# ----------------------------------------------------------------------

strip = 0.01

local_regions = [
    (
        r'$x$-boundary layer',
        np.linspace(0.0, strip, 201),
        np.linspace(0.0, 1.0, 51)
    ),
    (
        r'$y$-boundary layer',
        np.linspace(0.0, 1.0, 301),
        np.linspace(0.0, strip, 301)
    ),
    (
        'Corner layer',
        np.linspace(0.0, strip, 301),
        np.linspace(0.0, strip, 301)
    ),
]

fig, axes = plt.subplots(
    3, 3,
    figsize=(14, 12),
    constrained_layout=True
)

for row, (region_name, xg, yg) in enumerate(local_regions):

    XX, YY = np.meshgrid(xg, yg)

    # Exact solution
    U_exact = u_exact_np(XX, YY)

    # PVD-Net prediction
    U_pred = predict_np(
        XX.ravel(),
        YY.ravel()
    ).reshape(XX.shape)

    # Absolute error
    U_err = np.abs(U_exact - U_pred)

    # --------------------------------------------------------------
    # Use the same color range for Exact and Prediction
    # --------------------------------------------------------------
    vmin = min(U_exact.min(), U_pred.min())
    vmax = max(U_exact.max(), U_pred.max())

    # Exact
    im0 = axes[row, 0].imshow(
        U_exact,
        origin='lower',
        extent=[xg[0], xg[-1], yg[0], yg[-1]],
        aspect='auto',
        cmap='jet',
        vmin=vmin,
        vmax=vmax
    )

    axes[row, 0].set_title(f'{region_name}: Exact')
    axes[row, 0].set_xlabel(r'$x$')
    axes[row, 0].set_ylabel(r'$y$')

    fig.colorbar(
        im0,
        ax=axes[row, 0],
        fraction=0.046,
        pad=0.04
    )

    # Prediction
    im1 = axes[row, 1].imshow(
        U_pred,
        origin='lower',
        extent=[xg[0], xg[-1], yg[0], yg[-1]],
        aspect='auto',
        cmap='jet',
        vmin=vmin,
        vmax=vmax
    )

    axes[row, 1].set_title(f'{region_name}: Prediction')
    axes[row, 1].set_xlabel(r'$x$')
    axes[row, 1].set_ylabel(r'$y$')

    fig.colorbar(
        im1,
        ax=axes[row, 1],
        fraction=0.046,
        pad=0.04
    )

    # Absolute error
    im2 = axes[row, 2].imshow(
        U_err,
        origin='lower',
        extent=[xg[0], xg[-1], yg[0], yg[-1]],
        aspect='auto',
        cmap='jet'
    )

    axes[row, 2].set_title(f'{region_name}: Absolute error')
    axes[row, 2].set_xlabel(r'$x$')
    axes[row, 2].set_ylabel(r'$y$')

    fig.colorbar(
        im2,
        ax=axes[row, 2],
        fraction=0.046,
        pad=0.04
    )


plt.savefig(
    os.path.join(plot_dir, 'local_zoom_three_regions.png'),
    dpi=300,
    bbox_inches='tight'
)

plt.close()

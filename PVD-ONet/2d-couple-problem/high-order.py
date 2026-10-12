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

if torch.cuda.is_available(): print(f"CUDA is available. Using device {torch.cuda.current_device()}: {torch.cuda.get_device_name()}")
else: print("CUDA is not available. Using CPU.")

torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------
# problem / operator setting
# ----------------------------------------------------------------------
eps = 1.0e-3
xi0 = 20.0
kx = np.pi/2.0
num = 1100
ntrain = 1000
ntest = 100
batchsize = 50
width =90
learning_rate = 1.0e-3
epochs =10000

script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, 'high-order-pi-deeponet-weight10')
plot_dir = os.path.join(out_dir, 'plots')
model_dir = os.path.join(out_dir, 'model')
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
NET_NAMES = ['o0','o1','x0','xc','x1','y0','yc','y1','c']
PATHS = [os.path.join(model_dir, f'best_{name}.pth') for name in NET_NAMES]

# ----------------------------------------------------------------------
# DeepONet: same style as the supplied high-order 1D operator code
# ----------------------------------------------------------------------
class deeponet(nn.Module):
    def __init__(self, branch_size, trunk_size):
        super(deeponet, self).__init__()
        self.branch_size = branch_size
        self.trunk_size = trunk_size
        self.branch_net_ls = self.branch_net()
        self.trunk_net_ls = self.trunk_net()
    def branch_net(self):
        layers = []
        prev_size = self.branch_size[0]
        for hidden_size in self.branch_size[1:-1]:
            layers.append(nn.Linear(prev_size, hidden_size)); layers.append(nn.SiLU()); prev_size = hidden_size
        layers.append(nn.Linear(prev_size, self.branch_size[-1]))
        return nn.Sequential(*layers)
    def trunk_net(self):
        layers = []
        prev_size = self.trunk_size[0]
        for hidden_size in self.trunk_size[1:-1]:
            layers.append(nn.Linear(prev_size, hidden_size)); layers.append(nn.SiLU()); prev_size = hidden_size
        layers.append(nn.Linear(prev_size, self.trunk_size[-1]))
        return nn.Sequential(*layers)
    def forward(self, x_func, x_loc):
        y_func = self.branch_net_ls(x_func)
        y_loc = self.trunk_net_ls(x_loc)
        return torch.einsum('bi,bi->b', y_func, y_loc)

branch_size = [1,width,width,width,width,width,width]
trunk_size = [2,width,width,width,width,width,width]
net_o0 = deeponet(branch_size, trunk_size).to(device)
net_o1 = deeponet(branch_size, trunk_size).to(device)
net_x0 = deeponet(branch_size, trunk_size).to(device)
net_xc = deeponet(branch_size, trunk_size).to(device)
net_x1 = deeponet(branch_size, trunk_size).to(device)
net_y0 = deeponet(branch_size, trunk_size).to(device)
net_yc = deeponet(branch_size, trunk_size).to(device)
net_y1 = deeponet(branch_size, trunk_size).to(device)
net_c = deeponet(branch_size, trunk_size).to(device)
ALL_NETS = [net_o0,net_o1,net_x0,net_xc,net_x1,net_y0,net_yc,net_y1,net_c]

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)
for net in ALL_NETS: net.apply(init_normal)

def count_parameters(net): return sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f'Total trainable parameters: {sum(count_parameters(net) for net in ALL_NETS):,}')
print('=================================================')

# ----------------------------------------------------------------------
# source and c=0 exact solution
# ----------------------------------------------------------------------
def u0_exact_np(x, y):
    return np.cos(kx*x)*(1.0-np.exp(-2.0*x/eps))*(1.0-y**3)*(1.0-np.exp(-3.0*y/eps))

def f0_torch(x, y):
    p = torch.cos(kx*x); s = torch.sin(kx*x); q = 1.0-y**3
    return (x+2.0)*kx*s*q + 3.0*(y**3+3.0)*y**2*p + p*q

def f1_torch(x, y):
    p = torch.cos(kx*x); q = 1.0-y**3
    return kx**2*p*q + 6.0*y*p

def f_exact_torch(x, y):
    ex = torch.exp(-2.0*x/eps); ey = torch.exp(-3.0*y/eps); p = torch.cos(kx*x); s = torch.sin(kx*x); q = 1.0-y**3; ax = 1.0-ex; by = 1.0-ey; xx = p*ax; yy = q*by
    fx = (eps*kx**2*p + (x+2.0)*kx*s)*ax + 4.0*kx*s*ex - (2.0*x/eps)*p*ex
    fy = (6.0*eps*y + 3.0*(y**3+3.0)*y**2)*by + 18.0*y**2*ey - (3.0*y**3/eps)*q*ey
    return yy*fx + xx*fy + xx*yy

def grad(u, v): return torch.autograd.grad(u, v, grad_outputs=torch.ones_like(u), create_graph=True, only_inputs=True)[0]
def tensor(x): return torch.tensor(x, dtype=torch.float32)
def split_tensor(a):
    a = tensor(a)
    return a[:ntrain], a[-ntest:]
def xy(x, y): return torch.cat([x,y], dim=1)

# ----------------------------------------------------------------------
# 1100 operator samples: c ~ U(-3,3), 1000 train + 100 test
# One paired point for each subproblem/BC per operator sample, following the supplied operator code.
# ----------------------------------------------------------------------
c_all = np.random.uniform(0,1,(num,1)).astype(np.float32)
c_train_np = c_all[:ntrain].copy(); c_test_np = c_all[-ntest:].copy()
c_train, c_test = split_tensor(c_all)
X_O_train,X_O_test = split_tensor(np.random.uniform(0,1,(num,1)))
Y_O_train,Y_O_test = split_tensor(np.random.uniform(0,1,(num,1)))
XI_I_train,XI_I_test = split_tensor(np.random.uniform(0,xi0,(num,1)))
Y_I_train,Y_I_test = split_tensor(np.random.uniform(0,1,(num,1)))
X_J_train,X_J_test = split_tensor(np.random.uniform(0,1,(num,1)))
ETA_J_train,ETA_J_test = split_tensor(np.random.uniform(0,xi0,(num,1)))
XI_C_train,XI_C_test = split_tensor(np.random.uniform(0,xi0,(num,1)))
ETA_C_train,ETA_C_test = split_tensor(np.random.uniform(0,xi0,(num,1)))
Y_OB_train,Y_OB_test = split_tensor(np.random.uniform(0,1,(num,1)))
X_OB_train,X_OB_test = split_tensor(np.random.uniform(0,1,(num,1)))
Y_XB_train,Y_XB_test = split_tensor(np.random.uniform(0,1,(num,1)))
X_YB_train,X_YB_test = split_tensor(np.random.uniform(0,1,(num,1)))
XI_CB_train,XI_CB_test = split_tensor(np.random.uniform(0,xi0,(num,1)))
ETA_CB_train,ETA_CB_test = split_tensor(np.random.uniform(0,xi0,(num,1)))

train_dataset = torch.utils.data.TensorDataset(c_train,X_O_train,Y_O_train,XI_I_train,Y_I_train,X_J_train,ETA_J_train,XI_C_train,ETA_C_train,Y_OB_train,X_OB_train,Y_XB_train,X_YB_train,XI_CB_train,ETA_CB_train)
test_dataset = torch.utils.data.TensorDataset(c_test,X_O_test,Y_O_test,XI_I_test,Y_I_test,X_J_test,ETA_J_test,XI_C_test,ETA_C_test,Y_OB_test,X_OB_test,Y_XB_test,X_YB_test,XI_CB_test,ETA_CB_test)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batchsize,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batchsize,shuffle=False)

# ----------------------------------------------------------------------
# DeepONet building blocks: same high-order PVD ansatz
# ----------------------------------------------------------------------
def prep(c, *coords):
    c = c.to(device).view(-1,1)
    return (c,) + tuple(v.to(device) for v in coords)

def outer_total(c, x, y): return net_o0(c,xy(x,y)).reshape(-1,1) + eps*net_o1(c,xy(x,y)).reshape(-1,1)
def inner_x_total(c, xi, y): return net_x0(c,xy(xi,y)).reshape(-1,1) + eps*(xi*net_xc(c,xy(xi,y)).reshape(-1,1) + net_x1(c,xy(xi,y)).reshape(-1,1))
def inner_y_total(c, x, eta): return net_y0(c,xy(x,eta)).reshape(-1,1) + eps*(eta*net_yc(c,xy(x,eta)).reshape(-1,1) + net_y1(c,xy(x,eta)).reshape(-1,1))
def x_matching_extension(c, x, y):
    xi_far = torch.full_like(y,xi0); loc = xy(xi_far,y)
    return net_x0(c,loc).reshape(-1,1) + x*net_xc(c,loc).reshape(-1,1) + eps*net_x1(c,loc).reshape(-1,1)
def y_matching_extension(c, x, y):
    eta_far = torch.full_like(x,xi0); loc = xy(x,eta_far)
    return net_y0(c,loc).reshape(-1,1) + y*net_yc(c,loc).reshape(-1,1) + eps*net_y1(c,loc).reshape(-1,1)

# ----------------------------------------------------------------------
# losses: same high-order 2D PDE/BC/Van-Dyke/corner definitions
# ----------------------------------------------------------------------
def loss_outer_pde(c, x, y):
    c,x,y = prep(c,x,y); x = x.clone().requires_grad_(True); y = y.clone().requires_grad_(True)
    u0 = net_o0(c,xy(x,y)).reshape(-1,1); u1 = net_o1(c,xy(x,y)).reshape(-1,1)
    u0_x = grad(u0,x); u0_y = grad(u0,y); u0_xx = grad(u0_x,x); u0_yy = grad(u0_y,y)
    r0 = -(x+2.0)*u0_x - (y**3+3.0)*u0_y + u0 - f0_torch(x,y)
    u1_x = grad(u1,x); u1_y = grad(u1,y)
    r1 = -(x+2.0)*u1_x - (y**3+3.0)*u1_y + u1 - u0_xx - u0_yy - f1_torch(x,y)
    return torch.mean(r0**2) + torch.mean(r1**2)

def loss_outer_bc(c, yb, xb):
    c,yb,xb = prep(c,yb,xb); one_y = torch.ones_like(yb); one_x = torch.ones_like(xb)
    o0_x1 = net_o0(c,xy(one_y,yb)).reshape(-1,1); o0_y1 = net_o0(c,xy(xb,one_x)).reshape(-1,1)
    o1_x1 = net_o1(c,xy(one_y,yb)).reshape(-1,1); o1_y1 = net_o1(c,xy(xb,one_x)).reshape(-1,1)
    return torch.mean((o0_x1-c)**2) + torch.mean((o0_y1-c)**2) + torch.mean(o1_x1**2) + torch.mean(o1_y1**2)

def loss_inner_x_pde(c, xi, y):
    c,xi,y = prep(c,xi,y); xi = xi.clone().requires_grad_(True); y = y.clone().requires_grad_(True)
    U = inner_x_total(c,xi,y); U_xi = grad(U,xi); U_y = grad(U,y); x_phys = eps*xi
    r = grad(U_xi,xi) + (2.0+eps*xi)*U_xi + eps*(y**3+3.0)*U_y - eps*U + eps**2*grad(U_y,y) + eps*f_exact_torch(x_phys,y)
    return torch.mean(r**2)

def loss_inner_x_bc(c, y):
    c,y = prep(c,y); xi_zero = torch.zeros_like(y); xi_far = torch.full_like(y,xi0)
    wall0 = net_x0(c,xy(xi_zero,y)).reshape(-1,1); wall1 = net_x1(c,xy(xi_zero,y)).reshape(-1,1)
    xz = torch.zeros_like(y).requires_grad_(True); u0_at_wall = net_o0(c,xy(xz,y)).reshape(-1,1); du0dx = grad(u0_at_wall,xz)
    match0 = net_x0(c,xy(xi_far,y)).reshape(-1,1); matchc = net_xc(c,xy(xi_far,y)).reshape(-1,1); match1 = net_x1(c,xy(xi_far,y)).reshape(-1,1)
    u1_at_wall = net_o1(c,xy(torch.zeros_like(y),y)).reshape(-1,1)
    return torch.mean((wall0-c)**2) + torch.mean(wall1**2) + torch.mean((match0-u0_at_wall)**2) + torch.mean((matchc-du0dx)**2) + torch.mean((match1-u1_at_wall)**2)

def loss_inner_y_pde(c, x, eta):
    c,x,eta = prep(c,x,eta); x = x.clone().requires_grad_(True); eta = eta.clone().requires_grad_(True)
    V = inner_y_total(c,x,eta); V_eta = grad(V,eta); V_x = grad(V,x); y_phys = eps*eta
    r = grad(V_eta,eta) + (3.0+eps**3*eta**3)*V_eta + eps*(x+2.0)*V_x - eps*V + eps**2*grad(V_x,x) + eps*f_exact_torch(x,y_phys)
    return torch.mean(r**2)

def loss_inner_y_bc(c, x):
    c,x = prep(c,x); eta_zero = torch.zeros_like(x); eta_far = torch.full_like(x,xi0)
    wall0 = net_y0(c,xy(x,eta_zero)).reshape(-1,1); wall1 = net_y1(c,xy(x,eta_zero)).reshape(-1,1)
    yz = torch.zeros_like(x).requires_grad_(True); u0_at_wall = net_o0(c,xy(x,yz)).reshape(-1,1); du0dy = grad(u0_at_wall,yz)
    match0 = net_y0(c,xy(x,eta_far)).reshape(-1,1); matchc = net_yc(c,xy(x,eta_far)).reshape(-1,1); match1 = net_y1(c,xy(x,eta_far)).reshape(-1,1)
    u1_at_wall = net_o1(c,xy(x,torch.zeros_like(x))).reshape(-1,1)
    return torch.mean((wall0-c)**2) + torch.mean(wall1**2) + torch.mean((match0-u0_at_wall)**2) + torch.mean((matchc-du0dy)**2) + torch.mean((match1-u1_at_wall)**2)

def corner_base(c, xi, eta):
    x = eps*xi; y = eps*eta
    return outer_total(c,x,y) + (inner_x_total(c,xi,y)-x_matching_extension(c,x,y)) + (inner_y_total(c,x,eta)-y_matching_extension(c,x,y))

def loss_corner_pde(c, xi, eta):
    c,xi,eta = prep(c,xi,eta); xi = xi.clone().requires_grad_(True); eta = eta.clone().requires_grad_(True); x = eps*xi; y = eps*eta
    B = corner_base(c,xi,eta); B_xi = grad(B,xi); B_eta = grad(B,eta)
    L_B = grad(B_xi,xi) + grad(B_eta,eta) + (2.0+eps*xi)*B_xi + (3.0+eps**3*eta**3)*B_eta - eps*B
    W = net_c(c,xy(xi,eta)).reshape(-1,1); W_xi = grad(W,xi); W_eta = grad(W,eta)
    L_W = grad(W_xi,xi) + grad(W_eta,eta) + (2.0+eps*xi)*W_xi + (3.0+eps**3*eta**3)*W_eta - eps*W
    rhs = -(L_B + eps*f_exact_torch(x,y))
    return torch.mean((L_W-rhs)**2)

def loss_corner_bc(c, xi, eta):
    c,xi,eta = prep(c,xi,eta)
    xi_zero = torch.zeros_like(eta); x_zero = torch.zeros_like(eta); y_phys = eps*eta
    target_xi_zero = y_matching_extension(c,x_zero,y_phys) - inner_y_total(c,x_zero,eta)
    l_xi_zero = torch.mean((net_c(c,xy(xi_zero,eta)).reshape(-1,1)-target_xi_zero)**2)
    eta_zero = torch.zeros_like(xi); x_phys = eps*xi; y_zero = torch.zeros_like(xi)
    target_eta_zero = x_matching_extension(c,x_phys,y_zero) - inner_x_total(c,xi,y_zero)
    l_eta_zero = torch.mean((net_c(c,xy(xi,eta_zero)).reshape(-1,1)-target_eta_zero)**2)
    xi_far = torch.full_like(eta,xi0); eta_far = torch.full_like(xi,xi0)
    l_xi_far = torch.mean(net_c(c,xy(xi_far,eta)).reshape(-1,1)**2); l_eta_far = torch.mean(net_c(c,xy(xi,eta_far)).reshape(-1,1)**2)
    return l_xi_zero + l_eta_zero + l_xi_far + l_eta_far

def loss_components(batch):
    c,XO,YO,XI,YI,XJ,ETAJ,XIC,ETAC,YOB,XOB,YXB,XYB,XICB,ETACB = batch
    return {'outer_pde':loss_outer_pde(c,XO,YO),'outer_bc':loss_outer_bc(c,YOB,XOB),'x_pde':loss_inner_x_pde(c,XI,YI),'x_bc':loss_inner_x_bc(c,YXB),'y_pde':loss_inner_y_pde(c,XJ,ETAJ),'y_bc':loss_inner_y_bc(c,XYB),'corner_pde':loss_corner_pde(c,XIC,ETAC),'corner_bc':loss_corner_bc(c,XICB,ETACB)}

def total_loss_from_components(L):
    # return (L['outer_pde'] + L['outer_bc'] + (L['x_pde'] + L['y_pde'] + L['corner_pde'])
    #         + L['x_bc'] + L['y_bc'] + L['corner_bc'])
    return (L['outer_pde'] + L['outer_bc'] + 10.0*(L['x_pde']+L['y_pde']+L['corner_pde'])
            + L['x_bc'] + L['y_bc'] + L['corner_bc'])
# ----------------------------------------------------------------------
# training: same operator-learning organization as supplied 1D PI-DeepONet
# ----------------------------------------------------------------------
optimizer = optim.Adam([p for net in ALL_NETS for p in net.parameters()],lr=learning_rate,betas=(0.9,0.99),eps=1e-15)
train_loss_ls = np.zeros(epochs); test_loss_ls = np.zeros(epochs)
best_loss = float('inf'); best_epoch = 0
tic = time.time()
keys = ['total','outer_pde','outer_bc','x_pde','x_bc','y_pde','y_bc','corner_pde','corner_bc']

# for ep in range(epochs):
#     for net in ALL_NETS: net.train()
#     train_sum = {k:0.0 for k in keys}
#     for batch in train_loader:
#         optimizer.zero_grad(); L = loss_components(batch); loss = total_loss_from_components(L); loss.backward(); optimizer.step()
#         train_sum['total'] += loss.item()
#         for k in L: train_sum[k] += L[k].item()
#     for k in train_sum: train_sum[k] /= len(train_loader)
#     train_loss_ls[ep] = train_sum['total']
#
#     for net in ALL_NETS: net.eval()
#     test_sum = {k:0.0 for k in keys}
#     for batch in test_loader:
#         L = loss_components(batch); loss = total_loss_from_components(L); test_sum['total'] += loss.item()
#         for k in L: test_sum[k] += L[k].item()
#     for k in test_sum: test_sum[k] /= len(test_loader)
#     test_loss_ls[ep] = test_sum['total']
#
#     if test_sum['total'] < best_loss:
#         best_loss = test_sum['total']; best_epoch = ep+1
#         for net,p in zip(ALL_NETS,PATHS): torch.save(net.state_dict(),p)
#         with open(os.path.join(model_dir,'best epoch.txt'),'w',encoding='utf-8') as f: f.write(f'best epoch: {best_epoch}\nbest test loss: {best_loss:.12e}\n')
#
#     if ep % 1000 == 0:
#         print(f"Train Epoch: {ep}  Loss: {train_sum['total']:.5e} test Loss: {test_sum['total']:.5e}  best: {best_loss:.5e}")
#         print('-'*100)
#
# toc = time.time()
for net,p in zip(ALL_NETS,PATHS): net.load_state_dict(torch.load(p,map_location=device,weights_only=True))
# plt.figure(figsize=(7,5)); plt.plot(train_loss_ls,'r',label='train loss'); plt.plot(test_loss_ls,'g',label='test loss'); plt.yscale('log'); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(plot_dir,'loss.png'),dpi=600); plt.close()

# ----------------------------------------------------------------------
# uniformly valid high-order composite: same 2D PVD construction
# ----------------------------------------------------------------------
def composite(c, x, y):
    xi = torch.clamp(x/eps, max=xi0); eta = torch.clamp(y/eps, max=xi0)
    Mx = x_matching_extension(c,eps*xi,y); My = y_matching_extension(c,x,eps*eta)
    Ux = inner_x_total(c,xi,y); Uy = inner_y_total(c,x,eta)
    Wc = net_c(c,xy(xi,eta)).reshape(-1,1)
    return outer_total(c,x,y) + (Ux-Mx) + (Uy-My) + Wc

def predict_np(c_value, xnp, ynp, chunk=20000):
    xnp = np.asarray(xnp).ravel(); ynp = np.asarray(ynp).ravel(); out = []
    with torch.no_grad():
        for s in range(0,len(xnp),chunk):
            e = min(s+chunk,len(xnp)); x = torch.tensor(xnp[s:e,None],dtype=torch.float32,device=device); y = torch.tensor(ynp[s:e,None],dtype=torch.float32,device=device); c = torch.full((e-s,1),float(c_value),dtype=torch.float32,device=device)
            out.append(composite(c,x,y).cpu().numpy().ravel())
    return np.concatenate(out)

# ----------------------------------------------------------------------
# numerical reference: solve phi once, u_c = u_0 + c*phi
# ----------------------------------------------------------------------
def shishkin_grid(n, eps, beta, tau=2.0):
    if n % 2 == 0: raise ValueError('reference grid size must be odd')
    m = (n-1)//2; sigma = min(0.5,tau*eps*np.log(n)/beta)
    return np.concatenate([np.linspace(0.0,sigma,m+1),np.linspace(sigma,1.0,m+1)[1:]])

def build_reference_phi(n=401):
    xr = shishkin_grid(n,eps,2.0); yr = shishkin_grid(n,eps,3.0); nx = len(xr); ny = len(yr); n_unknown = (nx-2)*(ny-2)
    def ind(i,j): return (j-1)*(nx-2)+(i-1)
    rows,cols,vals = [],[],[]; rhs = np.zeros(n_unknown)
    for j in range(1,ny-1):
        yj = yr[j]; hym = yr[j]-yr[j-1]; hyp = yr[j+1]-yr[j]
        cym = -2.0*eps/(hym*(hym+hyp)); cyp = -2.0*eps/(hyp*(hym+hyp)); cy0 = 2.0*eps/(hym*hyp); by = -(yj**3+3.0); cyp += by/hyp; cy0 -= by/hyp
        for i in range(1,nx-1):
            xi = xr[i]; hxm = xr[i]-xr[i-1]; hxp = xr[i+1]-xr[i]
            cxm = -2.0*eps/(hxm*(hxm+hxp)); cxp = -2.0*eps/(hxp*(hxm+hxp)); cx0 = 2.0*eps/(hxm*hxp); bx = -(xi+2.0); cxp += bx/hxp; cx0 -= bx/hxp; k = ind(i,j)
            rows.append(k); cols.append(k); vals.append(cx0+cy0+1.0)
            if i == 1: rhs[k] -= cxm
            else: rows.append(k); cols.append(ind(i-1,j)); vals.append(cxm)
            if i == nx-2: rhs[k] -= cxp
            else: rows.append(k); cols.append(ind(i+1,j)); vals.append(cxp)
            if j == 1: rhs[k] -= cym
            else: rows.append(k); cols.append(ind(i,j-1)); vals.append(cym)
            if j == ny-2: rhs[k] -= cyp
            else: rows.append(k); cols.append(ind(i,j+1)); vals.append(cyp)
    phi_inner = spsolve(csr_matrix((vals,(rows,cols)),shape=(n_unknown,n_unknown)),rhs)
    phi = np.ones((ny,nx)); phi[1:-1,1:-1] = phi_inner.reshape(ny-2,nx-2)
    return xr,yr,phi

print('Building numerical reference solution...')
x_ref,y_ref,phi_ref = build_reference_phi(401)
phi_interp = RegularGridInterpolator((y_ref,x_ref),phi_ref,method='linear',bounds_error=False,fill_value=None)
def reference_np(c_value, xnp, ynp):
    xnp = np.asarray(xnp); ynp = np.asarray(ynp); pts = np.stack([ynp.ravel(),xnp.ravel()],axis=-1); phi = phi_interp(pts).reshape(xnp.shape)
    return u0_exact_np(xnp,ynp) + float(c_value)*phi

def rel_l2(ue,up): return np.linalg.norm(ue-up)/np.linalg.norm(ue)
def linf(ue,up): return np.max(np.abs(ue-up))
def region_error_for_c(c_value, xg, yg):
    XX,YY = np.meshgrid(xg,yg); ue = reference_np(c_value,XX,YY).ravel(); up = predict_np(c_value,XX.ravel(),YY.ravel())
    return rel_l2(ue,up),linf(ue,up)

# ----------------------------------------------------------------------
# error calculation: same four regions as the supplied high-order 2D PVD-Net
# ----------------------------------------------------------------------
strip = 0.01
regions = {
    'x-layer strip':(np.linspace(0,strip,201),np.linspace(0,1,51)),
    'y-layer strip':(np.linspace(0,1,51),np.linspace(0,strip,201)),
    'corner region':(np.linspace(0,strip,201),np.linspace(0,strip,201)),
    'global (all grids)':(np.concatenate([np.linspace(0,1,101),np.linspace(0,strip,201),np.linspace(0,1,51)]),np.concatenate([np.linspace(0,1,101),np.linspace(0,1,51),np.linspace(0,strip,201)]))
}
all_errors = {name:[] for name in regions}
for i,cv in enumerate(c_test_np[:,0]):
    for name,(xg,yg) in regions.items(): all_errors[name].append(region_error_for_c(cv,xg,yg))
    if (i+1) % 10 == 0: print(f'evaluated {i+1}/{ntest} test boundary values')

mean_errors = {}
for name in regions:
    arr = np.asarray(all_errors[name]); mean_errors[name] = (arr[:,0].mean(),arr[:,1].mean())
    print(f'{name:24s}  mean rel L2 = {mean_errors[name][0]:.6e}   mean L inf = {mean_errors[name][1]:.6e}')

with open(os.path.join(out_dir,'test error.txt'),'w',encoding='utf-8') as f:
    f.write(f'train functions: {ntrain}\ntest functions: {ntest}\nboundary distribution: U(-3,3)\nbatch size: {batchsize}\nepochs: {epochs}\ntraining time (s): {toc-tic:.1f}\nbest epoch: {best_epoch}\nbest test physics loss: {best_loss:.6e}\n\n')
    for name,v in mean_errors.items(): f.write(f'{name}: mean rel L2 = {v[0]:.6e}, mean L inf = {v[1]:.6e}\n')
np.savez(os.path.join(out_dir,'test_errors.npz'),c_test=c_test_np[:,0],**{name.replace(' ','_').replace('(','').replace(')',''):np.asarray(v) for name,v in all_errors.items()})

# ----------------------------------------------------------------------
# representative test plots: same slice/local-region style as the 2D code
# ----------------------------------------------------------------------
cv = float(c_test_np[0,0])
fig,axes = plt.subplots(2,2,figsize=(13,9))
slices = [('x',0.5),('x',0.005),('y',0.5),('y',0.005)]
for ax,(var,val) in zip(axes.ravel(),slices):
    s = np.linspace(0,1,801)
    if var == 'x': ue = reference_np(cv,s,np.full_like(s,val)); up = predict_np(cv,s,np.full_like(s,val)); ax.set_xlabel('x'); ax.set_title(f'slice  y = {val}')
    else: ue = reference_np(cv,np.full_like(s,val),s); up = predict_np(cv,np.full_like(s,val),s); ax.set_xlabel('y'); ax.set_title(f'slice  x = {val}')
    ax.plot(s,ue,'b-',label='numerical reference',linewidth=2); ax.plot(s,up,'r--',label='high-order PI-DeepONet',linewidth=2); ax.legend(loc='center right')
    axins = inset_axes(ax,width='42%',height='55%',loc='upper right'); m = s <= strip; axins.plot(s[m],ue[m],'b-',linewidth=2); axins.plot(s[m],up[m],'r--',linewidth=2); axins.set_xlim(0.0,strip); mark_inset(ax,axins,loc1=3,loc2=4,fc=(0.5,0.5,0.5,0.3),ec=(0.5,0.5,0.5,0.3),lw=1.0)
plt.tight_layout(); plt.savefig(os.path.join(plot_dir,'slices.png'),dpi=300); plt.close()

local_regions = [(r'$x$-boundary layer',np.linspace(0.0,strip,201),np.linspace(0.0,1.0,51)),(r'$y$-boundary layer',np.linspace(0.0,1.0,301),np.linspace(0.0,strip,301)),('Corner layer',np.linspace(0.0,strip,301),np.linspace(0.0,strip,301))]
fig,axes = plt.subplots(3,3,figsize=(14,12),constrained_layout=True)
for row,(region_name,xg,yg) in enumerate(local_regions):
    XX,YY = np.meshgrid(xg,yg); U_exact = reference_np(cv,XX,YY); U_pred = predict_np(cv,XX.ravel(),YY.ravel()).reshape(XX.shape); U_err = np.abs(U_exact-U_pred); vmin = min(U_exact.min(),U_pred.min()); vmax = max(U_exact.max(),U_pred.max())
    im0 = axes[row,0].imshow(U_exact,origin='lower',extent=[xg[0],xg[-1],yg[0],yg[-1]],aspect='auto',cmap='jet',vmin=vmin,vmax=vmax); axes[row,0].set_title(f'{region_name}: Reference'); axes[row,0].set_xlabel(r'$x$'); axes[row,0].set_ylabel(r'$y$'); fig.colorbar(im0,ax=axes[row,0],fraction=0.046,pad=0.04)
    im1 = axes[row,1].imshow(U_pred,origin='lower',extent=[xg[0],xg[-1],yg[0],yg[-1]],aspect='auto',cmap='jet',vmin=vmin,vmax=vmax); axes[row,1].set_title(f'{region_name}: Prediction'); axes[row,1].set_xlabel(r'$x$'); axes[row,1].set_ylabel(r'$y$'); fig.colorbar(im1,ax=axes[row,1],fraction=0.046,pad=0.04)
    im2 = axes[row,2].imshow(U_err,origin='lower',extent=[xg[0],xg[-1],yg[0],yg[-1]],aspect='auto',cmap='jet'); axes[row,2].set_title(f'{region_name}: Absolute error'); axes[row,2].set_xlabel(r'$x$'); axes[row,2].set_ylabel(r'$y$'); fig.colorbar(im2,ax=axes[row,2],fraction=0.046,pad=0.04)
plt.savefig(os.path.join(plot_dir,'local_zoom_three_regions.png'),dpi=300,bbox_inches='tight'); plt.close()

# Four unseen test functions: full-field reference / prediction / absolute error
sel = [0,25,50,75]
x_vis = np.linspace(0,1,301); y_vis = np.linspace(0,1,301); XXv,YYv = np.meshgrid(x_vis,y_vis)
for ind in sel:
    cv = float(c_test_np[ind,0]); Ue = reference_np(cv,XXv,YYv); Up = predict_np(cv,XXv.ravel(),YYv.ravel()).reshape(XXv.shape); Er = np.abs(Ue-Up); vmin = min(Ue.min(),Up.min()); vmax = max(Ue.max(),Up.max())
    fig,axes = plt.subplots(1,3,figsize=(18,5.2))
    im0 = axes[0].imshow(Ue,origin='lower',extent=[0,1,0,1],aspect='equal',cmap='jet',vmin=vmin,vmax=vmax); axes[0].set_title(rf'Numerical reference, $c={cv:.3f}$'); axes[0].set_xlabel(r'$x$'); axes[0].set_ylabel(r'$y$'); fig.colorbar(im0,ax=axes[0],fraction=0.046,pad=0.04)
    im1 = axes[1].imshow(Up,origin='lower',extent=[0,1,0,1],aspect='equal',cmap='jet',vmin=vmin,vmax=vmax); axes[1].set_title('High-order PI-DeepONet'); axes[1].set_xlabel(r'$x$'); axes[1].set_ylabel(r'$y$'); fig.colorbar(im1,ax=axes[1],fraction=0.046,pad=0.04)
    im2 = axes[2].imshow(Er,origin='lower',extent=[0,1,0,1],aspect='equal',cmap='viridis'); axes[2].set_title('Absolute error'); axes[2].set_xlabel(r'$x$'); axes[2].set_ylabel(r'$y$'); fig.colorbar(im2,ax=axes[2],fraction=0.046,pad=0.04)
    plt.tight_layout(); plt.savefig(os.path.join(plot_dir,f'test_{ind:03d}_field.png'),dpi=600,bbox_inches='tight'); plt.close()

print('results written to',out_dir)

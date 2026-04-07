import torch
import torch.nn as nn
import numpy as np
import matplotlib
from scipy.optimize import root_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes,mark_inset
import torch.optim as optim

if torch.cuda.is_available():
    print(f"CUDA is available. Using device {torch.cuda.current_device()}: {torch.cuda.get_device_name()}")
else:
    print("CUDA is not available. Using CPU.")

class deeponet(nn.Module):
    def __init__(self,branch_size,trunk_size):
        super(deeponet,self).__init__()
        self.branch_size=branch_size
        self.trunk_size=trunk_size
        self.branch_net_ls=self.branch_net()
        self.trunk_net_ls=self.trunk_net()
    def branch_net(self):
        layers=[]
        prev_size = self.branch_size[0]
        for hidden_size in self.branch_size[1:-1]:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.SiLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, self.branch_size[-1]))
        ls = nn.Sequential(*layers)
        return ls
    def trunk_net(self):
        layers=[]
        prev_size = self.trunk_size[0]
        for hidden_size in self.trunk_size[1:-1]:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.SiLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, self.trunk_size[-1]))
        ls = nn.Sequential(*layers)
        return ls

    def forward(self,x_func,x_loc):
        y_func=x_func
        y_loc=x_loc
        y_func=self.branch_net_ls(y_func)
        y_loc=self.trunk_net_ls(y_loc)
        Y=torch.einsum('bi,bi->b',y_func,y_loc) #unaligned
        #Y=torch.einsum('bi,ni->bn',y_func,y_loc) #aligned
        return Y
    def loss(self,y_pred,y_true): #mse
        train_loss=torch.mean(torch.square(y_true-y_pred))
        return train_loss

def tensor(x):
    return torch.tensor(x,dtype=torch.float)

def exact_sol(eps,ab,xf):
    lambda1 = (-1 + np.sqrt(1 - 4 * eps)) / (2 * eps)
    lambda2 = (-1 - np.sqrt(1 - 4 * eps)) / (2 * eps)
    alpha = ab[0]
    beta = ab[1]
    c1 = (-alpha * np.exp(lambda2) + beta) / (np.exp(lambda1) - np.exp(lambda2))
    c2 = (alpha * np.exp(lambda1) - beta) / (np.exp(lambda1) - np.exp(lambda2))
    sol = c1 * np.exp(lambda1 * xf) + c2 * np.exp(lambda2 * xf)
    return sol

def derivative_exact_sol(eps, ab, xf):
    lambda1 = (-1 + np.sqrt(1 - 4 * eps)) / (2 * eps)
    lambda2 = (-1 - np.sqrt(1 - 4 * eps)) / (2 * eps)
    alpha = ab[0]
    beta = ab[1]
    c1 = (-alpha * np.exp(lambda2) + beta) / (np.exp(lambda1) - np.exp(lambda2))
    c2 = (alpha * np.exp(lambda1) - beta) / (np.exp(lambda1) - np.exp(lambda2))
    dsol = c1 * lambda1 * np.exp(lambda1 * xf) + c2 * lambda2 * np.exp(lambda2 * xf)
    return dsol

def find_extrema(eps, ab):
    # 寻找导数的零点
    def find_zero_point():
        def derivative(xf):
            return derivative_exact_sol(eps, ab, xf)
        # 在区间 [0, 1] 内寻找零点
        zero_points = []
        try:
            sol = root_scalar(derivative, bracket=[0, 1], method='brentq')
            if sol.converged:
                zero_points.append(sol.root)
        except ValueError:
            pass  # 如果没有找到零点
        return zero_points
    # 寻找零点和边界点
    critical_points = [0, 1] + find_zero_point()
    critical_points = list(set(critical_points))  # 去重
    # 计算每个点的函数值
    values = [exact_sol(eps, ab, x) for x in critical_points]
    # 找到最大值和最小值
    max_value = max(values)
    max_point = critical_points[values.index(max_value)]
    return [max_point,max_value]



def mean_rel_l2(y_true,y_pred):
    return np.mean(np.linalg.norm(y_true-y_pred,2,axis=1)/
                      np.linalg.norm(y_true,2,axis=1))

def max_error(y_true,y_pred):
    error=np.max(np.abs(y_true-y_pred),axis=1)
    return np.mean(error)

torch.manual_seed(0)
np.random.seed(0)

if not os.path.exists('./pi-deeponet-0-order-5-100-silu'+ '/plots/'):
    os.makedirs('./pi-deeponet-0-order-5-100-silu'  + '/plots/')
if not os.path.exists('./pi-deeponet-0-order-5-100-silu'+ '/model/'):
    os.makedirs('./pi-deeponet-0-order-5-100-silu'  + '/model/')

path='./pi-deeponet-0-order-5-100-silu' + '/plots/'
path_best_inner_model = './pi-deeponet-0-order-5-100-silu' + '/model/best_inner_model.pth'
path_best_outer_model = './pi-deeponet-0-order-5-100-silu' + '/model/best_outer_model.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num=1100
ntrain=1000
ntest=100
net_width = 100
## Parameters###
eps= 0.001
nPt = 1100
batchsize = 50  # 50
learning_rate = 1e-4
epochs=100000
mm=2

net_inner = deeponet([mm,net_width,net_width,net_width,net_width,net_width],[1,net_width,net_width,net_width,net_width,net_width]).to(device)
net_outer = deeponet([mm,net_width,net_width,net_width,net_width,net_width],[1,net_width,net_width,net_width,net_width,net_width]).to(device)
total_params = (
    sum(p.numel() for p in net_inner.parameters())
    + sum(p.numel() for p in net_outer.parameters())
)

print(f"Total parameters: {total_params:,}")

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)

# use the modules apply function to recursively apply the initialization
net_inner.apply(init_normal)
net_outer.apply(init_normal)

############################################################
optimizer = optim.Adam(list(net_inner.parameters())+ list(net_outer.parameters()),
                       lr=learning_rate, betas=(0.9, 0.99), eps=10 ** -15)

a=np.random.uniform(0.4,1.4,(num,1))
b=np.random.uniform(1.5,2.5,(num,1))
ab=np.hstack((a,b))
ab_train=ab[:ntrain,:]
ab_test=ab[-ntest:,:]


ab_train=tensor(ab_train)
ab_test=tensor(ab_test)


x_inner=np.random.uniform(0, 0.02, nPt)
xi_inner=x_inner/eps #(0,20,1100)
xi_inner_reshape=xi_inner.reshape(-1,1)
xi_inner_reshape_train=tensor(xi_inner_reshape[:ntrain,:])
xi_inner_reshape_test=tensor(xi_inner_reshape[-ntest:,:])

x_outer=np.random.uniform(0,1,nPt)
x_outer_reshape=x_outer.reshape(-1,1)
x_outer_reshape_train=tensor(x_outer_reshape[:ntrain,:])
x_outer_reshape_test=tensor(x_outer_reshape[-ntest:,:])

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset
                                               (ab_train,xi_inner_reshape_train,x_outer_reshape_train),
                                               batch_size=batchsize,
                                               shuffle=True)

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset
                                              (ab_test,xi_inner_reshape_test,x_outer_reshape_test),
                                              batch_size=batchsize,
                                              shuffle=False)

xb_0 = np.array([0.], dtype=np.float32)
xb_0 = xb_0.reshape(-1, 1)
xb_1 = np.array([1.], dtype=np.float32)
xb_1 = xb_1.reshape(-1, 1)
xb_20= np.array([20.], dtype=np.float32)
xb_20 = xb_20.reshape(-1, 1)


def loss_outer(ab,x):
    ab = ab.cuda()
    ab = ab.view(-1, 2)
    x = x.to(device)
    x.requires_grad = True
    net_in = x
    y = net_outer(ab,net_in)
    y = y.reshape(-1,1)
    y_x = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    loss_1 = y_x + y
    loss_f = nn.MSELoss()
    loss = loss_f(loss_1, torch.zeros_like(loss_1))
    return loss

def loss_inner(ab,x):
    ab = ab.cuda()
    ab = ab.view(-1, 2)
    x = x.to(device)
    x.requires_grad = True
    net_in = x
    y = net_inner(ab,net_in)
    y = y.reshape(-1,1)
    y_x = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    y_xx = torch.autograd.grad(y_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    loss_1 = y_xx + y_x
    loss_f = nn.MSELoss()
    loss = loss_f(loss_1, torch.zeros_like(loss_1))
    return loss

def loss_bc(ab,x_0,x_1):
    ab = ab.cuda()
    ab = ab.view(-1, 2)
    x_0=np.tile(x_0,(ab.shape[0],1))
    x_0 = torch.FloatTensor(x_0).to(device)
    x_1 = np.tile(x_1, (ab.shape[0], 1))
    x_1 = torch.FloatTensor(x_1).to(device)
    y_out = net_outer(ab,x_1)
    y_out_1 = y_out.reshape(-1,1)
    y_in=net_inner(ab,x_0)
    y_in_0 = y_in.reshape(-1,1)
    loss_f = nn.MSELoss()
    loss = loss_f(y_out_1,ab[:,1].reshape(-1,1))+loss_f(y_in_0,ab[:,0].reshape(-1,1))
    return loss

def loss_BC_match(ab,x, xi):
    ab = ab.cuda()
    ab = ab.view(-1, 2)
    x = np.tile(x, (ab.shape[0], 1))
    x = torch.FloatTensor(x).to(device)
    xi = np.tile(xi, (ab.shape[0], 1))
    xi = torch.FloatTensor(xi).to(device)
    x.requires_grad = True
    y_out_0 = net_outer(ab,x)
    y_in_20 = net_inner(ab,xi)
    y_out_0 = y_out_0.reshape(-1,1)
    y_in_20 = y_in_20.reshape(-1,1)
    loss_f = nn.MSELoss()
    loss = loss_f(y_out_0,y_in_20)
    return loss

train_loss_ls=np.zeros(epochs)
test_loss_ls=np.zeros(epochs)
# best_loss = float('inf')
# for ep in range(epochs):
#     net_inner.train()
#     net_outer.train()
#     train_mse = 0
#     train_loss_outer = 0
#     train_loss_inner = 0
#     train_loss_bc = 0
#     train_loss_match = 0
#     for ab, xi_train, x_out_train in train_loader:
#         net_inner.zero_grad()
#         net_outer.zero_grad()
#
#         Loss_outer = loss_outer(ab,x_out_train)
#         Loss_inner = loss_inner(ab,xi_train)
#         Loss_bc = loss_bc(ab,xb_0,xb_1)
#         Loss_match = loss_BC_match(ab,xb_0,xb_20)
#         loss = Loss_outer+Loss_inner+Loss_bc+Loss_match
#         loss.backward()
#         optimizer.step()
#
#         train_mse += loss.item()
#         train_loss_outer += Loss_outer.item()
#         train_loss_inner += Loss_inner.item()
#         train_loss_bc += Loss_bc.item()
#         train_loss_match += Loss_match.item()
#
#     net_inner.eval()
#     net_outer.eval()
#
#     test_mse = 0
#     test_loss_outer = 0
#     test_loss_inner = 0
#     test_loss_bc = 0
#     test_loss_match = 0
#
#     for ab, xi_test, x_out_test in test_loader:
#         test_outer = loss_outer(ab, x_out_test)
#         test_inner = loss_inner(ab, xi_test)
#         test_bc = loss_bc(ab, xb_0, xb_1)
#         test_match = loss_BC_match(ab, xb_0, xb_20)
#         test_Loss = test_outer + test_inner + test_bc + test_match
#
#         test_mse += test_Loss.item()
#         test_loss_outer += test_outer.item()
#         test_loss_inner += test_inner.item()
#         test_loss_bc += test_bc.item()
#         test_loss_match += test_match.item()
#
#     train_mse /= len(train_loader)
#     train_loss_outer /= len(train_loader)
#     train_loss_inner/= len(train_loader)
#     train_loss_bc /= len(train_loader)
#     train_loss_match /= len(train_loader)
#
#     train_loss_ls[ep] = train_mse
#
#     test_mse /= len(test_loader)
#     test_loss_outer /= len(test_loader)
#     test_loss_inner /= len(test_loader)
#     test_loss_bc /= len(test_loader)
#     test_loss_match /= len(test_loader)
#
#     test_loss_ls[ep] = test_mse
#     if test_mse < best_loss:
#         best_loss = test_mse
#         torch.save(net_outer.state_dict(), path_best_outer_model)
#         torch.save(net_inner.state_dict(), path_best_inner_model)
#         with open('./pi-deeponet-0-order-5-100-silu' + '/model/best epoch', 'w') as f:
#             f.write('best epoch: ')
#             f.write(str(ep + 1))
#     if ep % 10000 == 0:
#         print('Train Epoch: {} \tLoss: {:.10f} train_outer: {:.10f} '
#               .format(ep, train_mse, train_loss_outer))
#         print('train_inner: {:.8f} train_bc: {:.8f} train_match: {:.8f} '
#               .format(train_loss_inner, train_loss_bc, train_loss_match))
#
#         print('                                                                                         ')
#         print('test Loss: {:.10f} test_outer: {:.10f} '
#               .format(test_mse, test_loss_outer))
#         print('test_inner: {:.8f} test_bc: {:.8f} test_match: {:.8f} '
#               .format(test_loss_inner, test_loss_bc, test_loss_match))
#         print('-------------------------------------------------------------------------------------')

net_outer.load_state_dict(torch.load(path_best_outer_model,weights_only=True))
net_inner.load_state_dict(torch.load(path_best_inner_model,weights_only=True))

# plt.figure()
# plt.plot(train_loss_ls, 'r', label='train loss')
# plt.plot(test_loss_ls, 'g', label='test loss')
# plt.yscale('log')
# plt.legend()
# plt.savefig(path + 'loss.png', dpi=600)

in_pt=10000
out_pt=100

x_inner=np.linspace(0,0.02,in_pt)
xi_inner=x_inner/eps
xi_inner_reshape1=xi_inner.reshape(-1,1)
xi_inner_reshape=tensor(xi_inner_reshape1).cuda()
x_outer=np.linspace(0.03,1,out_pt)
x_all=np.hstack((x_inner,x_outer))

xi_in = np.tile(xi_inner.reshape(1, -1), (ntest, 1))
xi_in=tensor(xi_in).cuda()

C_analytical = np.array(list(map(lambda ab: exact_sol(eps, ab, x_all), ab_test)))
#print(C_analytical.shape) [100,10100]

#计算极值点处的误差
extrema_pt_value=np.array(list(map(lambda ab:find_extrema(eps,ab),ab_test)))
#print(extrema_pt_value.shape)#[100,2]
extrema_pt=tensor(extrema_pt_value[:,0].reshape(-1,1)).cuda()
ab_test = ab_test.cuda()
y_outer_extrema_pt=net_outer(ab_test,extrema_pt)
y_inner_extrema_pt=net_inner(ab_test,extrema_pt/eps)

betae=torch.vstack(list(map(lambda ab:net_outer(ab.view(-1,2),tensor(xb_0).to(device)),ab_test)))


y_extrema_val=y_outer_extrema_pt+y_inner_extrema_pt-betae.flatten()
test_extrema_error=np.mean(np.abs(extrema_pt_value[:,1]-y_extrema_val.cpu().detach().numpy()))
print(test_extrema_error)

x_all_reshape = tensor(x_all.reshape(-1, 1)).cuda()

y_outer = torch.vstack(list(map(lambda ab:net_outer(torch.tile(ab.view(-1,2),(10100,1)),x_all_reshape),
                                ab_test)))

y_inner1 = torch.vstack(list(map(lambda ab:net_inner(torch.tile(ab.view(-1,2),(in_pt,1)),xi_inner_reshape),
                                 ab_test)))

y_inner2 = torch.tile(betae,(1,out_pt))
y_inner = torch.hstack((y_inner1, y_inner2))
y=y_inner+y_outer-betae
test_rel_l2=mean_rel_l2(C_analytical,y.cpu().detach().numpy())
test_max=max_error(C_analytical,y.cpu().detach().numpy())

test_rel_l2_inner = mean_rel_l2(C_analytical[:,:10000], y[:,:10000].cpu().detach().numpy())
test_max_inner = max_error(C_analytical[:,:10000], y[:,:10000].cpu().detach().numpy())

with open('./pi-deeponet-0-order-5-100-silu/test error', 'w') as f:
    f.write('test_rel_l2: ')
    f.write(str(test_rel_l2))
    f.write('\ntest max error:')
    f.write(str(test_max))
    f.write('\ntest_rel_l2_inner: ')
    f.write(str(test_rel_l2_inner))
    f.write('\ntest max error_inner:')
    f.write(str(test_max_inner))
    f.write('\ntest extrema point:')
    f.write(str(test_extrema_error))

fig, ax = plt.subplots(1, 1)

for i in range(ntest):
    if i % 25 == 0:
        ax.plot(x_all, C_analytical[i, :], 'b-',label='Analytical solution',linewidth=2, alpha=0.8, zorder=0)  # analytical
        ax.plot(x_all,y[i,:].cpu().detach().numpy(),'r--',label='Leading-order PVD-ONet',linewidth=2, alpha=1.0, zorder=1)
ax.legend(['Analytical solution', 'Leading-order PVD-ONet'], loc='best')

ax.set_xlim(0, 1)
ax.set_ylim(0, 7)
ax.margins(0)
bbox = (0.005, 0.1, 0.6, 1)
axins = inset_axes(ax, width='40%', height='50%', loc="lower center", bbox_to_anchor=bbox,
                           bbox_transform=ax.transAxes)

for i in range(ntest):
    if i % 25 == 0:
        axins.plot(x_all, C_analytical[i,:], 'b-', label='Analytical solution',linewidth=2, alpha=0.8,zorder=0)  # analytical
        axins.plot(x_all, y[i,:].cpu().detach().numpy(), 'r--', label='Leading-order PVD-ONet',
                   linewidth=2, alpha=1.0, zorder=1)
axins.set_xlim(0, 0.015)
axins.set_ylim(4.4, 6)

mark_inset(ax, axins, loc1=3, loc2=1, fc=(0.5, 0.5, 0.5, 0.3), ec=(0.5, 0.5, 0.5, 0.3), lw=1.0)
plt.savefig(path + 'pred.jpg', dpi=600)

#plt.show()




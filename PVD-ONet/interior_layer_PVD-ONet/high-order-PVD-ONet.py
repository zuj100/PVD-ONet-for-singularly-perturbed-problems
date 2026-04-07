import torch
import torch.nn as nn
import numpy as np
import matplotlib
from scipy.integrate import solve_bvp

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes,mark_inset
import torch.optim as optim
from scipy.optimize import root_scalar
import time

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



def mean_rel_l2(y_true,y_pred):
    return np.mean(np.linalg.norm(y_true-y_pred,2,axis=1)/
                      np.linalg.norm(y_true,2,axis=1))

def max_error(y_true,y_pred):
    error=np.max(np.abs(y_true-y_pred),axis=1)
    return np.mean(error)

torch.manual_seed(0)
np.random.seed(0)

if not os.path.exists('./high-order-PVD-ONet'+ '/plots/'):
    os.makedirs('./high-order-PVD-ONet'  + '/plots/')
if not os.path.exists('./high-order-PVD-ONet'+ '/model/'):
    os.makedirs('./high-order-PVD-ONet'  + '/model/')

path='./high-order-PVD-ONet' + '/plots/'
path_best_outer_left0_model = './high-order-PVD-ONet' + '/model/best_outer_left0_model.pth'
path_best_outer_left1_model = './high-order-PVD-ONet' + '/model/best_outer_left1_model.pth'
path_best_outer_right0_model = './high-order-PVD-ONet' + '/model/best_outer_right0_model.pth'
path_best_outer_right1_model = './high-order-PVD-ONet' + '/model/best_outer_right1_model.pth'
path_best_inner0_model = './high-order-PVD-ONet' + '/model/best_inner0_model.pth'
path_best_inner1_model = './high-order-PVD-ONet' + '/model/best_inner1_model.pth'
path_best_inner_c_model = './high-order-PVD-ONet' + '/model/best_inner_c_model.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num=600
ntrain=500
ntest=100

## Parameters###
eps= 0.001
nPt = 1100
batchsize = 50  # 50
learning_rate = 1e-4
epochs=100000
mm=2
width=60

net_outer_left0=deeponet([mm,width,width,width,width,width],[1,width,width,width,width,width]).to(device)
net_outer_left1=deeponet([mm,width,width,width,width,width],[1,width,width,width,width,width]).to(device)
net_outer_right0=deeponet([mm,width,width,width,width,width],[1,width,width,width,width,width]).to(device)
net_outer_right1=deeponet([mm,width,width,width,width,width],[1,width,width,width,width,width]).to(device)

net_inner0=deeponet([mm,width,width,width,width,width],[1,width,width,width,width,width]).to(device)
net_inner1=deeponet([mm,width,width,width,width,width],[1,width,width,width,width,width]).to(device)
net_inner_c=deeponet([mm,width,width,width,width,width],[1,width,width,width,width,width]).to(device)

total_params = (
    sum(p.numel() for p in net_inner0.parameters())
    + sum(p.numel() for p in net_outer_left0.parameters())
    + sum(p.numel() for p in net_outer_right0.parameters())
    + sum(p.numel() for p in net_outer_left1.parameters())
    + sum(p.numel() for p in net_outer_right1.parameters())+
    sum(p.numel() for p in net_inner1.parameters())
    +sum(p.numel() for p in net_inner_c.parameters()))
print("Total parameters:", total_params)

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)

# use the modules apply function to recursively apply the initialization
net_outer_left0.apply(init_normal)
net_outer_left1.apply(init_normal)
net_outer_right0.apply(init_normal)
net_outer_right1.apply(init_normal)

net_inner0.apply(init_normal)
net_inner1.apply(init_normal)
net_inner_c.apply(init_normal)

############################################################
optimizer = optim.Adam(list(net_outer_left0.parameters()) + list(net_outer_left1.parameters())
                           + list(net_outer_right0.parameters()) + list(net_outer_right1.parameters())
                           + list(net_inner0.parameters()) + list(net_inner1.parameters()) + list(net_inner_c.parameters()),
                           lr=learning_rate, betas=(0.9, 0.99), eps=10 ** -15)

a=np.random.uniform(0.6,1.0,(num,1))
b=-a
ab=np.hstack((a,b))
ab_train=ab[:ntrain,:]
ab_test=ab[-ntest:,:]


ab_train=tensor(ab_train)
ab_test=tensor(ab_test)


xi_inner=np.random.uniform(-20, 20, nPt)
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
x_bd= np.array([1/2.], dtype=np.float32)
x_bd = x_bd.reshape(-1, 1)
x0=1/2

def loss_outer_left0(ab,x):
    ab = ab.cuda()
    ab = ab.view(-1, 2)
    x = x.to(device)
    x.requires_grad = True
    net_in = x
    y = net_outer_left0(ab,net_in)
    #print(y.shape)#[50]
    y = y.reshape(-1,1)
    y_x = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    loss_1 = y-y_x* y
    loss_f = nn.MSELoss()
    loss = loss_f(loss_1, torch.zeros_like(loss_1))
    return loss

def loss_outer_left1(ab,x):
    ab = ab.cuda()
    ab = ab.view(-1, 2)
    x = x.to(device)
    x.requires_grad = True
    net_in = x
    y0 = net_outer_left0(ab,net_in).reshape(-1,1)
    y1 = net_outer_left1(ab,net_in).reshape(-1,1)
    y0_x = torch.autograd.grad(y0, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    y0_xx = torch.autograd.grad(y0_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    y1_x = torch.autograd.grad(y1, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    loss_1 = y0_xx - y0 * y1_x - y0_x * y1 + y1
    loss_f = nn.MSELoss()
    loss = loss_f(loss_1, torch.zeros_like(loss_1))
    return loss

def loss_outer_right0(ab,x):
    ab = ab.cuda()
    ab = ab.view(-1, 2)
    x = x.to(device)
    x.requires_grad = True
    net_in = x
    y = net_outer_right0(ab,net_in)
    #print(y.shape)#[50]
    y = y.reshape(-1,1)
    y_x = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    loss_1 = y-y_x* y
    loss_f = nn.MSELoss()
    loss = loss_f(loss_1, torch.zeros_like(loss_1))
    return loss

def loss_outer_right1(ab,x):
    ab = ab.cuda()
    ab = ab.view(-1, 2)
    x = x.to(device)
    x.requires_grad = True
    net_in = x
    y0 = net_outer_right0(ab,net_in).reshape(-1,1)
    y1 = net_outer_right1(ab,net_in).reshape(-1,1)
    y0_x = torch.autograd.grad(y0, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    y0_xx = torch.autograd.grad(y0_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    y1_x = torch.autograd.grad(y1, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    loss_1 = y0_xx - y0 * y1_x - y0_x * y1 + y1
    loss_f = nn.MSELoss()
    loss = loss_f(loss_1, torch.zeros_like(loss_1))
    return loss

def loss_inner(ab,x):
    ab = ab.cuda()
    ab = ab.view(-1, 2)
    x = x.to(device)
    x.requires_grad = True
    net_in = x
    Y0 = net_inner0(ab,net_in).reshape(-1,1)
    Y1 = net_inner_c(ab,net_in).reshape(-1,1)
    Y2 = net_inner1(ab,net_in).reshape(-1,1)

    Y = Y0 + eps * (Y2 + Y1 * x)
    Y_x = torch.autograd.grad(Y, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    Y_xx = torch.autograd.grad(Y_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    
    loss_1 = Y_xx - Y * Y_x + Y * eps
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
    y_left0 = net_outer_left0(ab,x_0).reshape(-1,1)
    y_left1 = net_outer_left1(ab,x_0).reshape(-1,1)
    y_right0 = net_outer_right0(ab,x_1).reshape(-1,1)
    y_right1 = net_outer_right1(ab,x_1).reshape(-1,1)
    loss_f = nn.MSELoss()
    loss_bc = (loss_f(y_left0, ab[:, 0].reshape(-1, 1)) + loss_f(y_left1, 0 * torch.ones_like(y_left1))
               + loss_f(y_right0, ab[:, 1].reshape(-1, 1)) + loss_f(y_right1, 0 * torch.ones_like(y_right1)))
    return loss_bc

def loss_Van_Dyke_match(ab, xb, yb_plus, yb_minus):
    ab = ab.cuda()
    ab = ab.view(-1, 2)
    x = np.tile(xb, (ab.shape[0], 1))
    x = torch.FloatTensor(x).to(device)
    yb_plus = np.tile(yb_plus, (ab.shape[0], 1))
    yb_plus = torch.FloatTensor(yb_plus).to(device)
    yb_minus = np.tile(yb_minus, (ab.shape[0], 1))
    yb_minus = torch.FloatTensor(yb_minus).to(device)
    x.requires_grad = True

    y_outer_left0 = net_outer_left0(ab,x).reshape(-1,1)
    y_outer_left1 = net_outer_left1(ab,x).reshape(-1,1)
    y_outer_right0 = net_outer_right0(ab,x).reshape(-1,1)
    y_outer_right1 = net_outer_right1(ab,x).reshape(-1,1)

    y_inner0_plus = net_inner0(ab,yb_plus).reshape(-1,1)
    y_inner1_plus = net_inner1(ab,yb_plus).reshape(-1,1)
    y_inner_c_plus = net_inner_c(ab,yb_plus).reshape(-1,1)

    y_inner0_minus = net_inner0(ab,yb_minus).reshape(-1,1)
    y_inner1_minus = net_inner1(ab,yb_minus).reshape(-1,1)
    y_inner_c_minus = net_inner_c(ab,yb_minus).reshape(-1,1)

    y_outer_left0_x = torch.autograd.grad(y_outer_left0, x, grad_outputs=torch.ones_like(x),
                                          create_graph=True, only_inputs=True)[0]
    y_outer_right0_x = torch.autograd.grad(y_outer_right0, x, grad_outputs=torch.ones_like(x),
                                           create_graph=True, only_inputs=True)[0]

    loss_f = nn.MSELoss()
    loss = (loss_f(y_outer_left0, y_inner0_minus) + loss_f(y_outer_left1, y_inner1_minus)
            + loss_f(y_outer_left0_x, y_inner_c_minus) + loss_f(y_outer_right0, y_inner0_plus)
            + loss_f(y_outer_right1, y_inner1_plus) + loss_f(y_outer_right0_x, y_inner_c_plus))
    return loss

def minus_term_left(ab,xb0, x):
    ab = ab.cuda()
    ab = ab.view(-1, 2)
    xb0 = torch.FloatTensor(xb0).to(device)
    x = torch.FloatTensor(x).to(device)
    xb0.requires_grad = True
    y_left0 = net_outer_left0(ab,xb0).reshape(-1,1)
    y_left1 = net_outer_left1(ab,xb0).reshape(-1,1)

    y_left0_x = \
    torch.autograd.grad(y_left0, xb0, grad_outputs=torch.ones_like(xb0), create_graph=True, only_inputs=True)[0]

    y = y_left0[0] + eps * y_left1[0] + y_left0_x[0] * (x - x0)
    return y

def minus_term_right(ab,xb0, x):
    ab = ab.cuda()
    ab = ab.view(-1, 2)
    xb0 = torch.FloatTensor(xb0).to(device)
    x = torch.FloatTensor(x).to(device)
    xb0.requires_grad = True
    y_right0 = net_outer_right0(ab,xb0).reshape(-1,1)
    y_right1 = net_outer_right1(ab,xb0).reshape(-1,1)

    y_right0_x = \
    torch.autograd.grad(y_right0, xb0, grad_outputs=torch.ones_like(xb0), create_graph=True, only_inputs=True)[0]

    y = y_right0[0] + eps * y_right1[0] + y_right0_x[0] * (x - x0)
    return y


# train_loss_ls=np.zeros(epochs)
# test_loss_ls=np.zeros(epochs)
# tic = time.time()
# best_loss = float('inf')
# for ep in range(epochs):
#     net_inner0.train()
#     net_inner1.train()
#     net_inner_c.train()
#     net_outer_left0.train()
#     net_outer_left1.train()
#     net_outer_right0.train()
#     net_outer_right1.train()
#
#     train_mse = 0
#     train_loss_outer = 0
#     train_loss_inner = 0
#     train_loss_bc = 0
#     train_loss_match = 0
#
#     for ab, xi_train, x_out_train in train_loader:
#         optimizer.zero_grad()
#
#         Loss_outer = (loss_outer_left0(ab,x_out_train) + loss_outer_left1(ab,x_out_train)
#                     + loss_outer_right0(ab,x_out_train) + loss_outer_right1(ab,x_out_train))
#
#         Loss_inner = loss_inner(ab,xi_train)
#         Loss_bc=loss_bc(ab,xb_0,xb_1)
#         Loss_match=loss_Van_Dyke_match(ab,x_bd,xb_20,-xb_20)
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
#     net_inner0.eval()
#     net_inner1.eval()
#     net_inner_c.eval()
#     net_outer_left0.eval()
#     net_outer_left1.eval()
#     net_outer_right0.eval()
#     net_outer_right1.eval()
#
#     test_mse = 0
#     test_loss_outer = 0
#     test_loss_inner = 0
#     test_loss_bc = 0
#     test_loss_match = 0
#
#
#     for ab, xi_test, x_out_test in test_loader:
#         test_outer = (loss_outer_left0(ab,x_out_test) + loss_outer_left1(ab,x_out_test)
#                     + loss_outer_right0(ab,x_out_test) + loss_outer_right1(ab,x_out_test))
#
#         test_inner = loss_inner(ab, xi_test)
#         test_bc = loss_bc(ab, xb_0, xb_1)
#         test_match = loss_Van_Dyke_match(ab, x_bd,xb_20,-xb_20)
#
#         test_Loss = test_outer+ test_inner + test_bc + test_match
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
#         torch.save(net_outer_left0.state_dict(), path_best_outer_left0_model)
#         torch.save(net_outer_left1.state_dict(), path_best_outer_left1_model)
#         torch.save(net_outer_right0.state_dict(), path_best_outer_right0_model)
#         torch.save(net_outer_right1.state_dict(), path_best_outer_right1_model)
#
#         torch.save(net_inner0.state_dict(), path_best_inner0_model)
#         torch.save(net_inner1.state_dict(), path_best_inner1_model)
#         torch.save(net_inner_c.state_dict(), path_best_inner_c_model)
#
#     if ep % 1000 == 0:
#         print('Train Epoch: {} \tLoss: {:.10f} train_outer: {:.10f} '
#               .format(ep, train_mse, train_loss_outer))
#         print('train_inner: {:.8f} train_bc: {:.8f} train_match: {:.8f} '
#               .format(train_loss_inner, train_loss_bc, train_loss_match))
#
#         print('                                                                                         ')
#         print('test Loss: {:.10f} test_outer: {:.10f}'
#               .format(test_mse, test_loss_outer))
#         print('test_inner: {:.8f} test_bc: {:.8f} test_match: {:.8f} '
#               .format(test_loss_inner, test_loss_bc, test_loss_match))
#         print('-------------------------------------------------------------------------------------')
# toc = time.time()
# elapseTime = toc - tic
#
# plt.figure()
# plt.plot(train_loss_ls, 'r', label='train loss')
# plt.plot(test_loss_ls, 'g', label='test loss')
# plt.yscale('log')
# plt.legend()
# plt.savefig(path + 'loss.png', dpi=600)

net_outer_left0.load_state_dict(torch.load(path_best_outer_left0_model, weights_only=True))
net_outer_left1.load_state_dict(torch.load(path_best_outer_left1_model, weights_only=True))
net_outer_right0.load_state_dict(torch.load(path_best_outer_right0_model, weights_only=True))
net_outer_right1.load_state_dict(torch.load(path_best_outer_right1_model, weights_only=True))
net_inner0.load_state_dict(torch.load(path_best_inner0_model, weights_only=True))
net_inner1.load_state_dict(torch.load(path_best_inner1_model, weights_only=True))
net_inner_c.load_state_dict(torch.load(path_best_inner_c_model, weights_only=True))

in_pt=10000
z_inner = np.linspace(-20, 20, 10000)
z_inner_reshape = z_inner.reshape(-1, 1)
x_inner = z_inner * eps + x0  # (0.48.0.52)
npt = 100
x_outer_left = np.linspace(0, 0.45, npt)
x_outer_right = np.linspace(0.55, 1, npt)
x_all = np.hstack((x_outer_left, x_inner, x_outer_right))

x_left_inner = np.hstack((x_outer_left, x_inner[:5000]))  # 5100
x_right_inner = np.hstack((x_inner[-5000:], x_outer_right))  # 5100

x_left_inner_gpu = torch.tensor(x_left_inner.reshape(-1, 1), dtype=torch.float32).to(device)
x_right_inner_gpu = torch.tensor(x_right_inner.reshape(-1, 1), dtype=torch.float32).to(device)

xi = torch.tensor(z_inner_reshape, dtype=torch.float32).to(device) #(10000,1)

xi_in = np.tile(z_inner.reshape(1, -1), (ntest, 1)) #(100,10000)
xi_in=tensor(xi_in).cuda()



def solve_singular_bvp(t_eval,ab, eps=0.001):
    ya = ab[0]
    yb = ab[1]
    def ode(t, Y):
        y = Y[0]
        yp = Y[1]
        ypp = (y * yp - y) / eps
        return np.vstack((yp, ypp))

    def bc(Y0, Y1):
        return np.array([Y0[0] - ya, Y1[0] - yb])


    # 初始网格
    t = np.linspace(0.0, 1.0, 800)

    Y_guess = np.zeros((2, t.size))
    Y_guess[0] = np.linspace(ya, yb, t.size)
    Y_guess[1] = (yb - ya) * np.ones_like(t)

    # 求解
    sol = solve_bvp(ode, bc, t, Y_guess, tol=1e-6, max_nodes=20000)

    # 在给定点计算解
    y_eval = sol.sol(t_eval)[0]

    return y_eval


y_numerical=np.array(list(map(lambda ab: solve_singular_bvp(x_all,ab,eps=0.001), ab_test.numpy()))) #(100,10200)


ab_test = ab_test.cuda()
y_inner0=torch.vstack(list(map(lambda ab: net_inner0(torch.tile(ab.view(-1,2),(in_pt,1)), xi),ab_test)))  #(100,10000)
y_inner_c=torch.vstack(list(map(lambda ab: net_inner_c(torch.tile(ab.view(-1,2),(in_pt,1)), xi),ab_test))) #(100,10000)
y_inner1=torch.vstack(list(map(lambda ab: net_inner1(torch.tile(ab.view(-1, 2), (in_pt, 1)), xi), ab_test))) #(100,10000)

y_inner = y_inner0 + eps * (xi_in * y_inner_c + y_inner1) #(100,10000)

y_outer_left0=torch.vstack(list(map(lambda ab: net_outer_left0(torch.tile(ab.view(-1,2),(5100,1)), x_left_inner_gpu)
                                    ,ab_test))) #(100,5100)
y_outer_left1=torch.vstack(list(map(lambda ab: net_outer_left1(torch.tile(ab.view(-1,2),(5100,1)), x_left_inner_gpu)
                                    ,ab_test))) #(100,5100)

y_outer_right0=torch.vstack(list(map(lambda ab: net_outer_right0(torch.tile(ab.view(-1,2),(5100,1)), x_right_inner_gpu)
                                    ,ab_test))) #(100,5100)

y_outer_right1=torch.vstack(list(map(lambda ab: net_outer_right1(torch.tile(ab.view(-1,2),(5100,1)), x_right_inner_gpu)
                                    ,ab_test))) #(100,5100)

y_outer_left = y_outer_left0 + eps * y_outer_left1 #(100,5100)
y_outer_right = y_outer_right0+ eps * y_outer_right1 #(100,5100)



match_left = torch.vstack(list(map(lambda ab:minus_term_left(ab,x_bd,x_left_inner), ab_test))) #(100,5100)
match_right = torch.vstack(list(map(lambda ab:minus_term_right(ab,x_bd,x_right_inner), ab_test))) #(100,5100)

match_left_100 = torch.vstack(list(map(lambda ab:minus_term_left(ab,x_bd,x_outer_left), ab_test))) #(100,100)
match_right_100 = torch.vstack(list(map(lambda ab:minus_term_right(ab,x_bd,x_outer_right), ab_test))) #(100,100)

y_inner_left = np.hstack((match_left_100.cpu().detach().numpy(), y_inner[:,:5000].cpu().detach().numpy())) #(100,5100)
y_inner_right = np.hstack((y_inner[:,-5000:].cpu().detach().numpy(), match_right_100.cpu().detach().numpy()))#(100,5100)

y_left = y_inner_left + y_outer_left.cpu().detach().numpy() - match_left.cpu().detach().numpy()
y_right = y_inner_right + y_outer_right.cpu().detach().numpy() - match_right.cpu().detach().numpy()

y = np.hstack((y_left, y_right))
print(y.shape)




test_rel_l2=mean_rel_l2(y_numerical,y)
test_max=max_error(y_numerical,y)

test_rel_l2_inner = mean_rel_l2(y_numerical[:,100:10100],y[:,100:10100])
test_max_inner = max_error(y_numerical[:,100:10100],y[:,100:10100])

with open('high-order-PVD-ONet/test error', 'a') as f:
    f.write('test_rel_l2: ')
    f.write(str(test_rel_l2))
    f.write('\ntest max error:')
    f.write(str(test_max))
    f.write('\ntest_rel_l2_inner: ')
    f.write(str(test_rel_l2_inner))
    f.write('\ntest max error_inner:')
    f.write(str(test_max_inner))
    # f.write('\ntotal time:')
    # f.write(str(elapseTime))

fig, ax = plt.subplots(1, 1)

for i in range(ntest):
    if i % 31 == 0:
        ax.plot(x_all, y_numerical[i, :], 'b-',label='Numerical solution',linewidth=3, alpha=0.8, zorder=0)  # analytical
        ax.plot(x_all,y[i,:],'r--',label='High-order PVD-ONet',linewidth=3, alpha=1.0, zorder=1)
ax.legend(['Numerical solution', 'High-order PVD-ONet'], loc='best')
ax.set_xlim(0, 1)
ax.set_ylim(-2, 2)
ax.margins(0)
bbox = (0.005, 0.1, 0.6, 1)
axins = inset_axes(ax, width='40%', height='50%', loc="lower center", bbox_to_anchor=bbox,
                           bbox_transform=ax.transAxes)

for i in range(ntest):
    if i % 31== 0:
        axins.plot(x_all, y_numerical[i,:], 'b-', label='Numerical solution',linewidth=3, alpha=0.8,zorder=0)  # analytical
        axins.plot(x_all, y[i,:], 'r--', label='high-order PVD-ONet',
                   linewidth=3, alpha=1.0, zorder=1)
axins.set_xlim(0.48, 0.52)
axins.set_ylim(-1.8, 1.8)
mark_inset(ax, axins, loc1=3, loc2=1, fc=(0.5, 0.5, 0.5, 0.3), ec=(0.5, 0.5, 0.5, 0.3), lw=1.0)
plt.savefig(path + 'pred.png', dpi=600)





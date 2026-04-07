import torch
import torch.nn as nn
import numpy as np
import matplotlib
import time
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import torch.optim as optim
from scipy.optimize import root_scalar

if torch.cuda.is_available():
    print(f"CUDA is available. Using device {torch.cuda.current_device()}: {torch.cuda.get_device_name()}")
else:
    print("CUDA is not available. Using CPU.")


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
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.SiLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, self.branch_size[-1]))
        ls = nn.Sequential(*layers)
        return ls

    def trunk_net(self):
        layers = []
        prev_size = self.trunk_size[0]
        for hidden_size in self.trunk_size[1:-1]:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.SiLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, self.trunk_size[-1]))
        ls = nn.Sequential(*layers)
        return ls

    def forward(self, x_func, x_loc):
        y_func = x_func
        y_loc = x_loc
        y_func = self.branch_net_ls(y_func)
        y_loc = self.trunk_net_ls(y_loc)
        Y = torch.einsum('bi,bi->b', y_func, y_loc)  # unaligned
        # Y=torch.einsum('bi,ni->bn',y_func,y_loc) #aligned
        return Y

    def loss(self, y_pred, y_true):  # mse
        train_loss = torch.mean(torch.square(y_true - y_pred))
        return train_loss


def tensor(x):
    return torch.tensor(x, dtype=torch.float)


def mean_rel_l2(y_true, y_pred):
    return np.mean(np.linalg.norm(y_true - y_pred, 2, axis=1) /
                   np.linalg.norm(y_true, 2, axis=1))


def max_error(y_true, y_pred):
    error = np.max(np.abs(y_true - y_pred), axis=1)
    return np.mean(error)


def a_coeff(x):
    return x+1


def b_coeff(x):
    return 5*np.cos(5*x)

def numerical_sol(eps, ab, a_func, b_func, N1=10000, N2=100):
    alpha = ab[0]
    beta = ab[1]
    x1 = np.linspace(0, eps * 20, N1)
    x2 = np.linspace(eps * 30, 1, N2)
    x = np.concatenate((x1, x2))
    N_total = len(x)
    # 初始化系数矩阵 A 和右侧向量 f
    A = np.zeros((N_total, N_total))
    f = np.zeros(N_total)
    # 边界条件：在 x=0 和 x=1 处
    A[0, 0] = 1.0
    f[0] = alpha
    A[-1, -1] = 1.0
    f[-1] = beta

    # 对内部节点 i=1,...,N_total-2 使用非均匀有限差分公式
    for i in range(1, N_total - 1):
        # 两侧间距
        h1 = x[i] - x[i - 1]
        h2 = x[i + 1] - x[i]

        # 非均匀格式逼近二阶导数 y''
        y2_coeff_im1 = 2.0 / (h1 * (h1 + h2))
        y2_coeff_i = -2.0 / (h1 * h2)
        y2_coeff_ip1 = 2.0 / (h2 * (h1 + h2))

        # 非均匀格式逼近一阶导数 y'
        y1_coeff_im1 = -h2 / (h1 * (h1 + h2))
        y1_coeff_i = (h2 - h1) / (h1 * h2)
        y1_coeff_ip1 = h1 / (h2 * (h1 + h2))

        A[i, i - 1] = eps * y2_coeff_im1 + a_func(x[i]) * y1_coeff_im1
        A[i, i] = eps * y2_coeff_i + a_func(x[i]) * y1_coeff_i + b_func(x[i])
        A[i, i + 1] = eps * y2_coeff_ip1 + a_func(x[i]) * y1_coeff_ip1

    # 求解线性系统 A*y = f
    y = np.linalg.solve(A, f)
    return y

torch.manual_seed(0)
np.random.seed(0)

if not os.path.exists('./record-time-high' + '/plots/'):
    os.makedirs('./record-time-high' + '/plots/')
if not os.path.exists('./record-time-high' + '/model/'):
    os.makedirs('./record-time-high' + '/model/')

path = './record-time-high' + '/plots/'
path_best_inner_a_model = './record-time-high' + '/model/best_inner_a_model.pth'
path_best_inner_b_model = './record-time-high' + '/model//best_inner_b_model.pth'
path_best_inner_c_model = './record-time-high' + '/model/best_inner_c_model.pth'
path_best_outer_a_model = './record-time-high' + '/model/best_outer_a_model.pth'
path_best_outer_c_model = './record-time-high' + '/model/best_outer_c_model.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num = 1100
ntrain = 1000
ntest = 100

## Parameters###
eps = 0.001
nPt = 1100
batchsize = 50  # 50
learning_rate = 1e-4
epochs = 100000
mm = 2
width = 40

net_outer_a = deeponet([mm, width, width, width, width, width], [1, width, width, width, width, width]).to(device)
net_outer_c = deeponet([mm, width, width, width, width, width], [1, width, width, width, width, width]).to(device)
net_inner_a = deeponet([mm, width, width, width, width, width], [1, width, width, width, width, width]).to(device)
net_inner_b = deeponet([mm, width, width, width, width, width], [1, width, width, width, width, width]).to(device)
net_inner_c = deeponet([mm, width, width, width, width, width], [1, width, width, width, width, width]).to(device)


def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)


# use the modules apply function to recursively apply the initialization
net_inner_a.apply(init_normal)
net_inner_b.apply(init_normal)
net_inner_c.apply(init_normal)

net_outer_a.apply(init_normal)
net_outer_c.apply(init_normal)

############################################################
optimizer = optim.Adam(list(net_inner_a.parameters()) + list(net_inner_b.parameters())
                       + list(net_inner_c.parameters()) + list(net_outer_a.parameters())
                       + list(net_outer_c.parameters()), lr=learning_rate, betas=(0.9, 0.99), eps=10 ** -15)

# a=np.random.uniform(0.5,3.5,(num,1))
# b=np.random.uniform(1.5,4.5,(num,1))
a = np.random.uniform(0.4, 1.4, (num, 1))
b = np.random.uniform(1.5, 2.5, (num, 1))
ab = np.hstack((a, b))
ab_train = ab[:ntrain, :]
ab_test = ab[-ntest:, :]

ab_train = tensor(ab_train)
ab_test = tensor(ab_test)

x_inner = np.random.uniform(0, 0.02, nPt)
xi_inner = x_inner / eps  # (0,20,1100)
xi_inner_reshape = xi_inner.reshape(-1, 1)
xi_inner_reshape_train = tensor(xi_inner_reshape[:ntrain, :])
xi_inner_reshape_test = tensor(xi_inner_reshape[-ntest:, :])

x_outer = np.random.uniform(0, 1, nPt)
x_outer_reshape = x_outer.reshape(-1, 1)
x_outer_reshape_train = tensor(x_outer_reshape[:ntrain, :])
x_outer_reshape_test = tensor(x_outer_reshape[-ntest:, :])

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset
                                           (ab_train, xi_inner_reshape_train, x_outer_reshape_train),
                                           batch_size=batchsize,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset
                                          (ab_test, xi_inner_reshape_test, x_outer_reshape_test),
                                          batch_size=batchsize,
                                          shuffle=False)

xb_0 = np.array([0.], dtype=np.float32)
xb_0 = xb_0.reshape(-1, 1)
xb_1 = np.array([1.], dtype=np.float32)
xb_1 = xb_1.reshape(-1, 1)
xb_20 = np.array([20.], dtype=np.float32)
xb_20 = xb_20.reshape(-1, 1)


def loss_outer_a(ab, x):
    ab = ab.to(device)
    ab = ab.view(-1, 2)
    x = x.to(device)
    a_x = a_coeff(x)
    b_x = b_coeff(x.cpu().numpy())
    b_x = tensor(b_x).to(device)
    x.requires_grad = True
    net_in = x
    y = net_outer_a(ab, net_in)
    # print(y.shape)#[50]
    y = y.reshape(-1, 1)
    y_x = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    loss_1 = a_x * y_x + b_x * y
    loss_f = nn.MSELoss()
    loss = loss_f(loss_1, torch.zeros_like(loss_1))
    return loss


def loss_outer_c(ab, x):
    ab = ab.to(device)
    ab = ab.view(-1, 2)
    x = x.to(device)
    a_x = a_coeff(x)
    b_x = b_coeff(x.cpu().numpy())
    b_x = tensor(b_x).to(device)
    x.requires_grad = True
    net_in = x
    y1 = net_outer_c(ab, net_in)
    y0 = net_outer_a(ab, net_in)
    y1 = y1.reshape(-1, 1)
    y0 = y0.reshape(-1, 1)
    y1_x = torch.autograd.grad(y1, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    y0_x = torch.autograd.grad(y0, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    y0_xx = torch.autograd.grad(y0_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    loss_1 = a_x * y1_x + b_x * y1 + y0_xx
    loss_f = nn.MSELoss()
    loss = loss_f(loss_1, torch.zeros_like(loss_1))
    return loss


def loss_inner(ab, x):
    ab = ab.to(device)
    ab = ab.view(-1, 2)
    x = x.to(device)
    a_x = a_coeff(x * eps)
    b_x = b_coeff((x * eps).cpu().numpy())
    b_x = tensor(b_x).to(device)
    x.requires_grad = True
    net_in = x
    Y0 = net_inner_a(ab, net_in)
    Y1 = net_inner_b(ab, net_in)
    Y2 = net_inner_c(ab, net_in)

    Y0 = Y0.reshape(-1, 1)
    Y1 = Y1.reshape(-1, 1)
    Y2 = Y2.reshape(-1, 1)
    Y = Y0 + eps * (Y2 + Y1 * x)
    Y_x = torch.autograd.grad(Y, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    Y_xx = torch.autograd.grad(Y_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    loss_1 = Y_xx + a_x * Y_x + b_x * Y * eps
    loss_f = nn.MSELoss()
    loss = loss_f(loss_1, torch.zeros_like(loss_1))
    return loss


def loss_bc(ab, x_0, x_1):
    ab = ab.to(device)
    ab = ab.view(-1, 2)
    x_0 = np.tile(x_0, (ab.shape[0], 1))
    x_0 = torch.FloatTensor(x_0).to(device)
    x_1 = np.tile(x_1, (ab.shape[0], 1))
    x_1 = torch.FloatTensor(x_1).to(device)

    out_a = net_outer_a(ab, x_1)
    out_c = net_outer_c(ab, x_1)
    a_1 = out_a.reshape(-1, 1)
    c_1 = out_c.reshape(-1, 1)
    in_a = net_inner_a(ab, x_0)
    in_c = net_inner_c(ab, x_0)
    a_hat_0 = in_a.reshape(-1, 1)
    c_hat_0 = in_c.reshape(-1, 1)
    loss_f = nn.MSELoss()
    loss_bc = loss_f(a_1, ab[:, 1].reshape(-1, 1)) + loss_f(c_1, 0 * torch.ones_like(c_1)) \
              + loss_f(a_hat_0, ab[:, 0].reshape(-1, 1)) + loss_f(c_hat_0, 0 * torch.ones_like(c_hat_0))
    return loss_bc


def loss_Van_Dyke_match(ab, x, xi):
    ab = ab.to(device)
    ab = ab.view(-1, 2)
    x = np.tile(x, (ab.shape[0], 1))
    x = torch.FloatTensor(x).to(device)
    xi = np.tile(xi, (ab.shape[0], 1))
    xi = torch.FloatTensor(xi).to(device)
    x.requires_grad = True
    a_0 = net_outer_a(ab, x)
    c_0 = net_outer_c(ab, x)
    a_hat_xi = net_inner_a(ab, xi)
    c_hat_xi = net_inner_c(ab, xi)
    b_hat_xi = net_inner_b(ab, xi)

    a_0 = a_0.reshape(-1, 1)
    c_0 = c_0.reshape(-1, 1)
    a_hat_xi = a_hat_xi.reshape(-1, 1)
    b_hat_xi = b_hat_xi.reshape(-1, 1)
    c_hat_xi = c_hat_xi.reshape(-1, 1)

    a0_x = torch.autograd.grad(a_0, x, grad_outputs=torch.ones_like(x),
                               create_graph=True, only_inputs=True)[0]
    loss_f = nn.MSELoss()
    loss = loss_f(a_0, a_hat_xi) + loss_f(c_0, c_hat_xi) + loss_f(a0_x, b_hat_xi)
    return loss


def minus_term(ab, xb0, x):
    ab = ab.to(device)
    ab = ab.view(-1, 2)
    xb0 = torch.Tensor(xb0).to(device)
    x = torch.Tensor(x).to(device)
    xb0.requires_grad = True
    a_0 = net_outer_a(ab, xb0)
    c_0 = net_outer_c(ab, xb0)
    a_0 = a_0.reshape(-1, 1)
    c_0 = c_0.reshape(-1, 1)
    a0_x = torch.autograd.grad(a_0, xb0, grad_outputs=torch.ones_like(xb0),
                               create_graph=True, only_inputs=True)[0]
    y = a_0[0] + eps * c_0[0] + a0_x[0] * x
    return y


train_loss_ls = np.zeros(epochs)
test_loss_ls = np.zeros(epochs)
best_loss = float('inf')
tic=time.time()
for ep in range(epochs):
    net_inner_a.train()
    net_inner_b.train()
    net_inner_c.train()
    net_outer_a.train()
    net_outer_c.train()
    train_mse = 0
    train_loss_outer_a = 0
    train_loss_outer_c = 0
    train_loss_inner = 0
    train_loss_bc = 0
    train_loss_match = 0
    train_loss_data_inner = 0
    train_loss_data_outer = 0
    for ab, xi_train, x_out_train in train_loader:
        net_inner_a.zero_grad()
        net_inner_b.zero_grad()
        net_inner_c.zero_grad()
        net_outer_a.zero_grad()
        net_outer_c.zero_grad()

        Loss_outer_a = loss_outer_a(ab, x_out_train)
        Loss_outer_c = loss_outer_c(ab, x_out_train)
        Loss_inner = loss_inner(ab, xi_train)
        Loss_bc = loss_bc(ab, xb_0, xb_1)
        Loss_match = loss_Van_Dyke_match(ab, xb_0, xb_20)
        loss = Loss_outer_a + Loss_outer_c + Loss_inner + Loss_bc + Loss_match
        loss.backward()
        optimizer.step()

        train_mse += loss.item()
        train_loss_outer_a += Loss_outer_a.item()
        train_loss_outer_c += Loss_outer_c.item()
        train_loss_inner += Loss_inner.item()
        train_loss_bc += Loss_bc.item()
        train_loss_match += Loss_match.item()

    net_inner_a.eval()
    net_inner_b.eval()
    net_inner_c.eval()
    net_outer_a.eval()
    net_outer_c.eval()

    test_mse = 0
    test_loss_outer_a = 0
    test_loss_outer_c = 0
    test_loss_inner = 0
    test_loss_bc = 0
    test_loss_match = 0
    test_loss_data_inner = 0
    test_loss_data_outer = 0

    for ab, xi_test, x_out_test in test_loader:
        test_outer_a = loss_outer_a(ab, x_out_test)
        test_outer_c = loss_outer_c(ab, x_out_test)
        test_inner = loss_inner(ab, xi_test)
        test_bc = loss_bc(ab, xb_0, xb_1)
        test_match = loss_Van_Dyke_match(ab, xb_0, xb_20)
        test_Loss = test_outer_a + test_outer_c + test_inner + test_bc + test_match

        test_mse += test_Loss.item()
        test_loss_outer_a += test_outer_a.item()
        test_loss_outer_c += test_outer_c.item()
        test_loss_inner += test_inner.item()
        test_loss_bc += test_bc.item()
        test_loss_match += test_match.item()

    train_mse /= len(train_loader)
    train_loss_outer_a /= len(train_loader)
    train_loss_outer_c /= len(train_loader)
    train_loss_inner /= len(train_loader)
    train_loss_bc /= len(train_loader)
    train_loss_match /= len(train_loader)

    train_loss_ls[ep] = train_mse

    test_mse /= len(test_loader)
    test_loss_outer_a /= len(test_loader)
    test_loss_outer_c /= len(test_loader)
    test_loss_inner /= len(test_loader)
    test_loss_bc /= len(test_loader)
    test_loss_match /= len(test_loader)

    test_loss_ls[ep] = test_mse
    if test_mse < best_loss:
        best_loss = test_mse
        torch.save(net_outer_a.state_dict(), path_best_outer_a_model)
        torch.save(net_outer_c.state_dict(), path_best_outer_c_model)

        torch.save(net_inner_a.state_dict(), path_best_inner_a_model)
        torch.save(net_inner_b.state_dict(), path_best_inner_b_model)
        torch.save(net_inner_c.state_dict(), path_best_inner_c_model)
        with open('./record-time-high' + '/model/best epoch', 'w') as f:
            f.write('best epoch: ')
            f.write(str(ep + 1))
    if ep % 10000 == 0:
        print('Train Epoch: {} \tLoss: {:.10f} train_outer_a: {:.10f} train_outer_c: {:.10f}'
              .format(ep, train_mse, train_loss_outer_a, train_loss_outer_c))
        print('train_inner: {:.8f} train_bc: {:.8f} train_match: {:.8f} '
              .format(train_loss_inner, train_loss_bc, train_loss_match))

        print('                                                                                         ')
        print('test Loss: {:.10f} test_outer_a: {:.10f} test_outer_c: {:.10f}'
              .format(test_mse, test_loss_outer_a, test_loss_outer_c))
        print('test_inner: {:.8f} test_bc: {:.8f} test_match: {:.8f} '
              .format(test_loss_inner, test_loss_bc, test_loss_match))
        print('-------------------------------------------------------------------------------------')
toc = time.time()
elapseTime = toc - tic
print("elapse time = ", elapseTime)
net_outer_a.load_state_dict(torch.load(path_best_outer_a_model, weights_only=True))
net_outer_c.load_state_dict(torch.load(path_best_outer_c_model, weights_only=True))
net_inner_a.load_state_dict(torch.load(path_best_inner_a_model, weights_only=True))
net_inner_b.load_state_dict(torch.load(path_best_inner_b_model, weights_only=True))
net_inner_c.load_state_dict(torch.load(path_best_inner_c_model, weights_only=True))

# plt.figure()
# plt.plot(train_loss_ls, 'r', label='train loss')
# plt.plot(test_loss_ls, 'g', label='test loss')
# plt.yscale('log')
# plt.legend()
# plt.savefig(path + 'loss.png', dpi=600)

in_pt = 10000
out_pt = 100
total_pt = in_pt + out_pt
x_inner = np.linspace(0, 0.02, in_pt)
xi_inner = x_inner / eps
xi_inner_reshape1 = xi_inner.reshape(-1, 1)
xi_inner_reshape = tensor(xi_inner_reshape1).to(device)
x_outer = np.linspace(0.03, 1, out_pt)
x_all = np.hstack((x_inner, x_outer))

xi_in = np.tile(xi_inner.reshape(1, -1), (ntest, 1))
xi_in = tensor(xi_in).to(device)
y_numerical_sol = np.array(list(map(lambda ab: numerical_sol(eps, ab.numpy(), a_coeff, b_coeff), ab_test)))
print(y_numerical_sol.shape)
ab_test = ab_test.to(device)
x_all_reshape = tensor(x_all.reshape(-1, 1)).to(device)

y_outer = torch.vstack(list(map(lambda ab: net_outer_a(torch.tile(ab.view(-1, 2), (10100, 1)), x_all_reshape),
                                ab_test))) \
          + eps * \
          torch.vstack(list(map(lambda ab: net_outer_c(torch.tile(ab.view(-1, 2), (10100, 1)), x_all_reshape),
                                ab_test)))
y_inner1 = torch.vstack(list(map(lambda ab: net_inner_a(torch.tile(ab.view(-1, 2), (in_pt, 1)), xi_inner_reshape),
                                 ab_test))) \
           + eps * (torch.vstack(
    list(map(lambda ab: net_inner_c(torch.tile(ab.view(-1, 2), (in_pt, 1)), xi_inner_reshape), ab_test)))
                    + torch.vstack(
            list(map(lambda ab: net_inner_b(torch.tile(ab.view(-1, 2), (in_pt, 1)), xi_inner_reshape), ab_test)))
                    * xi_in)

betae = torch.vstack(list(map(lambda ab: minus_term(ab, xb_0, x_outer), ab_test)))
# print(betae.shape)#[100,100]
y_inner2 = betae
y_inner = torch.hstack((y_inner1, y_inner2))
betae2 = torch.vstack(list(map(lambda ab: minus_term(ab, xb_0, x_all), ab_test)))
y = y_inner + y_outer - betae2
test_rel_l2 = mean_rel_l2(y_numerical_sol, y.cpu().detach().numpy())
test_max = max_error(y_numerical_sol, y.cpu().detach().numpy())

test_rel_l2_inner = mean_rel_l2(y_numerical_sol[:, :10000], y[:, :10000].cpu().detach().numpy())
test_max_inner = max_error(y_numerical_sol[:, :10000], y[:, :10000].cpu().detach().numpy())

with open('record-time-high/test error', 'w') as f:
    f.write('test_rel_l2: ')
    f.write(str(test_rel_l2))
    f.write('\ntest max error:')
    f.write(str(test_max))
    f.write('\ntest_rel_l2_inner: ')
    f.write(str(test_rel_l2_inner))
    f.write('\ntest max error_inner:')
    f.write(str(test_max_inner))
    f.write('\ntotal time:')
    f.write(str(elapseTime))
fig, ax = plt.subplots(1, 1)

for i in range(ntest):
    if i % 25 == 0:
        ax.plot(x_all, y_numerical_sol[i, :], 'b-', label='Numerical solution', alpha=0.8,linewidth=2)
        ax.plot(x_all, y[i, :].cpu().detach().numpy(), 'r--', label='High-order PVD-ONet', alpha=1,linewidth=2)
ax.legend(['Numerical solution', 'High-order PVD-ONet'], loc='best')
ax.set_xlim(0, 1)
ax.set_ylim(0, 3)
ax.margins(0)
axins = inset_axes(ax, width='40%', height='50%', loc="lower center", bbox_to_anchor=(0.005, 0.1, 0.6, 1),
                       bbox_transform=ax.transAxes)
for i in range(ntest):
    if i % 25 == 0:
        axins.plot(x_all, y_numerical_sol[i, :], 'b-', label='Numerical solution',alpha=0.8,linewidth=2)
        axins.plot(x_all, y[i, :].cpu().detach().numpy(), 'r--', label='High-order PVD-ONet',
                   alpha=0.8,linewidth=2)
axins.set_xlim(0, 0.015)
axins.set_ylim(0.9, 1.6)

mark_inset(ax, axins, loc1=3, loc2=1, fc=(0.5, 0.5, 0.5, 0.3), ec=(0.5, 0.5, 0.5, 0.3), lw=1.0)
plt.savefig(path + 'pred.png', dpi=600)





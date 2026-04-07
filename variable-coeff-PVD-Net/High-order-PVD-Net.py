import torch
import numpy as np
from matplotlib import pyplot as plt
import torch.optim as optim
import torch.nn as nn
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import os
import matplotlib
matplotlib.use('Agg')
from scipy.interpolate import interp1d

torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    print('cuda')
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    print('cpu')


# def my function

def mean_rel_l2_0(y_true, y_pred):
    return np.linalg.norm(y_true - y_pred, 2) / np.linalg.norm(y_true, 2)

def max_error_0(y_true, y_pred):
    error = np.abs(y_true - y_pred)
    return np.max(error)


def tensor(x):
    return torch.tensor(x, dtype=torch.float)
def a_coeff(x):
    return x+1

def b_coeff(x):
    return 5*np.cos(5*x)


def numerical_sol(eps, a_func, b_func, alpha, beta, N1=10000, N2=100):
    """
    利用非均匀有限差分法求解边界值问题
         eps*y''(x) + a(x)*y'(x) + b(x)*y(x) = 0,   y(0)=alpha, y(1)=beta,
    其中在区间 [0,0.02] 上使用 N1 个分段（密集网格），在 [0.03,1] 上使用 N2 个分段。

    参数：
      eps    -- 小参数 ε
      a_func -- 系数函数 a(x)，要求 a(x)>0
      b_func -- 系数函数 b(x)
      alpha  -- 左边界条件 y(0)=alpha
      beta   -- 右边界条件 y(1)=beta
      N1     -- [0,0.02] 区间的分段数
      N2     -- [0.03,1] 区间的分段数

    返回：
      x -- 拼接后的非均匀网格节点数组
      y -- 数值解数组
    """
    # 生成两个区间的均匀网格
    x1 = np.linspace(0, 0.02, N1)
    x2 = np.linspace(0.03, 1, N2)
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


## Parameters###
eps = 0.001
net_width = 40
A = 20
Pt_num = 201
batchsize = 50
learning_rate = 1e-4
epochs = 100000
loop = 1


class Net(nn.Module):
    def __init__(self, input_n, net_width):
        super(Net, self).__init__()
        self.input_n = input_n
        self.net_width = net_width
        self.main = nn.Sequential(
            nn.Linear(self.input_n, self.net_width),
            nn.SiLU(),
            nn.Linear(self.net_width, self.net_width),
            nn.SiLU(),
            nn.Linear(self.net_width, self.net_width),
            nn.SiLU(),
            nn.Linear(self.net_width, self.net_width),
            nn.SiLU(),
            nn.Linear(self.net_width, self.net_width),
            nn.SiLU(),
            nn.Linear(self.net_width, 1),
        )

    def forward(self, x):
        output = self.main(x)
        return output


if not os.path.exists('./high-order-PVD-Net-new-sample' + '/plots/'):
    os.makedirs('./high-order-PVD-Net-new-sample' + '/plots/')

path = './high-order-PVD-Net-new-sample' + '/plots/'

test_rel_l2_loop=np.zeros(loop)
test_max_error_loop=np.zeros(loop)
test_rel_l2_inner_loop=np.zeros(loop)
test_max_error_inner_loop=np.zeros(loop)

################################################################
for lp in range(loop):
    if not os.path.exists('./high-order-PVD-Net-new-sample' + '/model/loop_{:.0f}/'.format(lp + 1)):
        os.makedirs('./high-order-PVD-Net-new-sample' + '/model/loop_{:.0f}/'.format(lp + 1))
    path_best_inner_a_model = './high-order-PVD-Net-new-sample' + '/model/loop_{:.0f}/best_inner_a_model.pth'.format(lp + 1)
    path_best_inner_b_model = './high-order-PVD-Net-new-sample' + '/model/loop_{:.0f}/best_inner_b_model.pth'.format(lp + 1)
    path_best_inner_c_model = './high-order-PVD-Net-new-sample' + '/model/loop_{:.0f}/best_inner_c_model.pth'.format(lp + 1)

    path_best_outer_a_model = './high-order-PVD-Net-new-sample' + '/model/loop_{:.0f}/best_outer_a_model.pth'.format(lp + 1)
    path_best_outer_c_model = './high-order-PVD-Net-new-sample' + '/model/loop_{:.0f}/best_outer_c_model.pth'.format(lp + 1)

    net_outer_a = Net(1, net_width).to(device)
    net_outer_c = Net(1, net_width).to(device)
    net_inner_a = Net(1, net_width).to(device)
    net_inner_b = Net(1, net_width).to(device)
    net_inner_c = Net(1, net_width).to(device)

    ###### Initialize the neural network using a standard method ##############
    def init_normal(m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight)


    # use the modules apply function to recursively apply the initialization
    net_inner_a.apply(init_normal)
    net_inner_b.apply(init_normal)
    net_inner_c.apply(init_normal)

    net_outer_a.apply(init_normal)
    net_outer_c.apply(init_normal)

    optimizer = optim.Adam(list(net_inner_a.parameters()) + list(net_inner_b.parameters())
                           + list(net_inner_c.parameters()) + list(net_outer_a.parameters())
                           + list(net_outer_c.parameters()), lr=learning_rate, betas=(0.9, 0.99), eps=10 ** -15)
    x_inner = np.random.uniform(0, 0.02, Pt_num)
    xi_inner = x_inner / eps
    xi_inner_reshape = xi_inner.reshape(-1, 1)

    x_outer = np.random.uniform(0, 1, Pt_num)
    x_outer_reshape = x_outer.reshape(-1, 1)

    xb_0 = np.array([0.], dtype=np.float32)
    xb_0 = xb_0.reshape(-1, 1)
    xb_1 = np.array([1.], dtype=np.float32)
    xb_1 = xb_1.reshape(-1, 1)
    xb_20 = np.array([20.], dtype=np.float32)
    xb_20 = xb_20.reshape(-1, 1)

    a = 1.
    b = 2.


    def loss_outer_a(x):
        a_x = tensor(a_coeff(x)).to(device)
        b_x = tensor(b_coeff(x)).to(device)
        x = tensor(x).to(device)
        x.requires_grad = True
        net_in = x
        C = net_outer_a(net_in)
        C = C.view(len(C), -1)
        c_x = torch.autograd.grad(C, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        loss_1 = a_x * c_x + b_x * C
        loss_f = nn.MSELoss()
        loss = loss_f(loss_1, torch.zeros_like(loss_1))
        return loss


    def loss_outer_c(x):
        a_x = tensor(a_coeff(x)).to(device)
        b_x = tensor(b_coeff(x)).to(device)
        x = tensor(x).to(device)
        x.requires_grad = True
        net_in = x
        y1 = net_outer_c(net_in)
        y0 = net_outer_a(net_in)
        y1 = y1.view(len(y1), -1)
        y0 = y0.view(len(y0), -1)
        y1_x = torch.autograd.grad(y1, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        y0_x = torch.autograd.grad(y0, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        y0_xx = torch.autograd.grad(y0_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        loss_1 = a_x * y1_x + b_x * y1 + y0_xx
        loss_f = nn.MSELoss()
        loss = loss_f(loss_1, torch.zeros_like(loss_1))
        return loss


    def loss_inner(x):  # xi
        a_x = tensor(a_coeff(x* eps)).to(device)
        b_x = tensor(b_coeff(x* eps)).to(device)
        x = tensor(x).to(device)
        x.requires_grad = True
        net_in = x
        Y0 = net_inner_a(net_in)
        Y1 = net_inner_b(net_in)
        Y2 = net_inner_c(net_in)

        Y0 = Y0.view(len(Y0), -1)
        Y1 = Y1.view(len(Y1), -1)
        Y2 = Y2.view(len(Y2), -1)
        Y = Y0 + eps * (Y2 + Y1 * x)
        Y_x = torch.autograd.grad(Y, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        Y_xx = torch.autograd.grad(Y_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        loss_1 = Y_xx + a_x * Y_x + b_x * Y * eps
        loss_f = nn.MSELoss()
        loss = loss_f(loss_1, torch.zeros_like(loss_1))
        return loss


    def loss_bc(x_0, x_1):
        x_0 = tensor(x_0).to(device)
        x_1 = tensor(x_1).to(device)
        out_a = net_outer_a(x_1)
        out_c = net_outer_c(x_1)
        a_1 = out_a.view(len(out_a), -1)
        c_1 = out_c.view(len(out_c), -1)
        in_a = net_inner_a(x_0)
        in_c = net_inner_c(x_0)
        a_hat_0 = in_a.view(len(in_a), -1)
        c_hat_0 = in_c.view(len(in_c), -1)
        loss_f = nn.MSELoss()
        loss_bc = loss_f(a_1, b * torch.ones_like(a_1)) + loss_f(c_1, 0 * torch.ones_like(c_1)) \
                  + loss_f(a_hat_0, a * torch.ones_like(a_hat_0)) + loss_f(c_hat_0, 0 * torch.ones_like(c_hat_0))
        return loss_bc


    def loss_Van_Dyke_match(x, xi):
        x = tensor(x).to(device)
        xi = tensor(xi).to(device)
        x.requires_grad = True
        a_0 = net_outer_a(x)
        c_0 = net_outer_c(x)
        a_hat_xi = net_inner_a(xi)
        c_hat_xi = net_inner_c(xi)
        b_hat_xi = net_inner_b(xi)

        a_0 = a_0.view(len(a_0), -1)
        c_0 = c_0.view(len(c_0), -1)
        a_hat_xi = a_hat_xi.view(len(a_hat_xi), -1)
        b_hat_xi = b_hat_xi.view(len(b_hat_xi), -1)
        c_hat_xi = c_hat_xi.view(len(c_hat_xi), -1)

        a0_x = torch.autograd.grad(a_0, x, grad_outputs=torch.ones_like(x),
                                   create_graph=True, only_inputs=True)[0]
        loss_f = nn.MSELoss()
        loss = loss_f(a_0, a_hat_xi) + loss_f(c_0, c_hat_xi) + loss_f(a0_x, b_hat_xi)
        return loss


    def minus_term(xb0, x):
        xb0 = tensor(xb0).to(device)
        x = tensor(x).to(device)
        xb0.requires_grad = True
        a_0 = net_outer_a(xb0)
        c_0 = net_outer_c(xb0)
        a_0 = a_0.view(len(a_0), -1)
        c_0 = c_0.view(len(c_0), -1)
        a0_x = torch.autograd.grad(a_0, xb0, grad_outputs=torch.ones_like(xb0),
                                   create_graph=True, only_inputs=True)[0]

        y = a_0[0] + eps * c_0[0] + a0_x[0] * x
        return y


    loss_list = np.zeros(epochs)
    best_loss = float('inf')
    for epoch in range(epochs):


        net_inner_a.zero_grad()
        net_inner_b.zero_grad()
        net_inner_c.zero_grad()
        net_outer_a.zero_grad()
        net_outer_c.zero_grad()

        Loss_outer_a = loss_outer_a(x_outer_reshape)
        Loss_outer_c = loss_outer_c(x_outer_reshape)
        Loss_inner = loss_inner(xi_inner_reshape)
        Loss_bc=loss_bc(xb_0,xb_1)
        Loss_match=loss_Van_Dyke_match(xb_0,xb_20)

        loss = Loss_outer_a+Loss_outer_c+Loss_inner+Loss_bc+Loss_match
        loss.backward()

        optimizer.step()


        loss_list[epoch] = loss.item()
        if loss < best_loss:
            best_loss = loss
            torch.save(net_outer_a.state_dict(), path_best_outer_a_model)
            torch.save(net_outer_c.state_dict(), path_best_outer_c_model)

            torch.save(net_inner_a.state_dict(), path_best_inner_a_model)
            torch.save(net_inner_b.state_dict(), path_best_inner_b_model)
            torch.save(net_inner_c.state_dict(), path_best_inner_c_model)


            with open('./high-order-PVD-Net-new-sample' + '/model/loop_{:.0f}/best epoch'.format(lp + 1), 'w') as f:
                f.write('best epoch: ')
                f.write(str(epoch + 1))

        if epoch % 1000 == 0:
            print('Train Epoch: {} \tLoss: {:.10f} Loss_outer_a: {:.10f} Loss_outer_c: {:.10f} '
                  .format(epoch, loss.item(),Loss_outer_a.item(),Loss_outer_c.item()))
            print('Loss_inner: {:.8f}  Loss_match: {:.8f} Loss_bc: {:.8f} '
                  .format(Loss_inner.item(),Loss_match.item(),Loss_bc.item()))
            print('-----------------------------------------------------------------------------------------')
    plt.figure()
    plt.plot(loss_list, 'r', label='total loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(path + 'loss_{:.0f}.png'.format(lp + 1), dpi=600)

    net_outer_a.load_state_dict(torch.load(path_best_outer_a_model, weights_only=True))
    net_outer_c.load_state_dict(torch.load(path_best_outer_c_model, weights_only=True))

    net_inner_a.load_state_dict(torch.load(path_best_inner_a_model, weights_only=True))
    net_inner_b.load_state_dict(torch.load(path_best_inner_b_model, weights_only=True))
    net_inner_c.load_state_dict(torch.load(path_best_inner_c_model, weights_only=True))



    x_inner = np.linspace(0, 0.02, 10000)  # 0.005
    xi_inner = x_inner / eps
    xi_inner_reshape1 = xi_inner.reshape(-1, 1)
    xi_inner_reshape = tensor(xi_inner_reshape1)
    x_outer = np.linspace(0.03, 1, 100)  # 100
    x_all = np.hstack((x_inner, x_outer))  # 200

    x_max = np.array([0.003])
    xi_max = x_max / eps
    xi_max_reshape1 = xi_max.reshape(-1, 1)
    xi_max_reshape = tensor(xi_max_reshape1)

    y_inner_max = net_inner_a(xi_max_reshape.to(device)) + \
                  eps * (net_inner_c(xi_max_reshape.to(device)) +
                         net_inner_b(xi_max_reshape.to(device)) * xi_max_reshape.to(device))
    y_outer_max = net_outer_a(tensor(x_max.reshape(-1, 1)).to(device)) \
                  + eps * net_outer_c(tensor(x_max.reshape(-1, 1)).to(device))
    betae_max = minus_term(xb_0, x_max)
    y_junction_pred = y_inner_max + y_outer_max - betae_max


    x_all_reshape = x_all.reshape(-1, 1)
    x_all_reshape = tensor(x_all_reshape)
    y_outer = net_outer_a(x_all_reshape.to(device)) + eps * net_outer_c(x_all_reshape.to(device))
    y_inner1 = net_inner_a(xi_inner_reshape.to(device)) + \
               eps * (net_inner_c(xi_inner_reshape.to(device))
                      + net_inner_b(xi_inner_reshape.to(device)) * xi_inner_reshape.to(device))
    betae = minus_term(xb_0, x_outer)
    y_inner2 = betae
    y_inner = torch.hstack((y_inner1.reshape(-1), y_inner2))  # 200
    betae2 = minus_term(xb_0, x_all)
    y = y_inner + y_outer.reshape(-1) - betae2
    y_numerical = numerical_sol(eps, a_coeff, b_coeff, a, b)


    y_interp = interp1d(x_all, y_numerical, kind='linear')
    y_junction_numerical = y_interp(x_max)
    junction_error = np.abs(y_junction_pred.cpu().detach().numpy() - y_junction_numerical)

    y_anal_10101 = np.hstack((y_numerical, y_junction_numerical))
    y_pred_10101 = np.hstack((y.cpu().detach().numpy(), y_junction_pred[0].cpu().detach().numpy()))

    test_rel_l2 = mean_rel_l2_0(y_anal_10101, y_pred_10101)
    test_max = max_error_0(y_anal_10101, y_pred_10101)
    test_rel_l2_inner = mean_rel_l2_0(y_numerical[:10000], y[:10000].cpu().detach().numpy())
    test_max_inner = max_error_0(y_numerical[:10000], y[:10000].cpu().detach().numpy())
    
    test_rel_l2_loop[lp] = test_rel_l2
    test_max_error_loop[lp] = test_max
    test_rel_l2_inner_loop[lp] = test_rel_l2_inner
    test_max_error_inner_loop[lp] = test_max_inner
    
    with open('./high-order-PVD-Net-new-sample/test error', 'a') as f:
        f.write('\nloop_{:.0f}\n'.format(lp + 1))
        f.write('test_rel_l2: ')
        f.write(str(test_rel_l2))
        f.write('\ntest max error:')
        f.write(str(test_max))
        f.write('\ntest_rel_l2_inner: ')
        f.write(str(test_rel_l2_inner))
        f.write('\ntest max error_inner:')
        f.write(str(test_max_inner))
        f.write('\njunction point error:')
        f.write(str(junction_error))

    fig, ax = plt.subplots(1, 1)
    ax.plot(x_all, y_numerical[:], 'b-', label='Numerical solution', alpha=0.8,linewidth=2)
    ax.plot(x_all, y.cpu().detach().numpy(), 'r--', label='High-order PVD-Net', alpha=1,linewidth=2)  # PINN
    ax.legend(loc='best')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 3)
    ax.margins(0)
    axins = inset_axes(ax, width='40%', height='50%', loc="lower center", bbox_to_anchor=(0.005, 0.1, 0.6, 1),
                       bbox_transform=ax.transAxes)
    axins.plot(x_all, y_numerical[:], 'b-', label='Numerical solution', alpha=0.8,linewidth=2)
    axins.plot(x_all, y.cpu().detach().numpy(), 'r--', label='High-order PVD-Net', alpha=1,linewidth=2)  # PINN
    axins.set_xlim(0, 0.015)
    axins.set_ylim(1, 1.5)

    mark_inset(ax, axins, loc1=3, loc2=1, fc=(0.5, 0.5, 0.5, 0.3), ec=(0.5, 0.5, 0.5, 0.3), lw=1.0)
    plt.savefig(path + 'pred_{:.0f}.png'.format(lp + 1), dpi=600)








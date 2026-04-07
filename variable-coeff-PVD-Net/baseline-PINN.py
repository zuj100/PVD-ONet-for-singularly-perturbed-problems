import torch
import numpy as np
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch.optim as optim
import time
import math
import torch.nn as nn
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import os

torch.manual_seed(1)
np.random.seed(1)
if not os.path.exists('./baseline-PINN-5-200-silu' + '/plots/'):
    os.makedirs('./baseline-PINN-5-200-silu' + '/plots/')
path = './baseline-PINN-5-200-silu' + '/plots/'
# CUDA support
if torch.cuda.is_available():
    print('cuda')
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    print('cpu')

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
    return  y

eps = 0.001
net_width = 200  # 40
in_Pt_num = 1000
out_Pt_num = 100
batchsize = 50  # 50
learning_rate = 1e-4
epochs = 100000
loop = 1

for lp in range(loop):
    if not os.path.exists('./baseline-PINN-5-200-silu' + '/model/loop_{:.0f}/'.format(lp + 1)):
        os.makedirs('./baseline-PINN-5-200-silu' + '/model/loop_{:.0f}/'.format(lp + 1))
    path_best_model = './baseline-PINN-5-200-silu' + '/model/loop_{:.0f}/best_model.pth'.format(lp + 1)


    class Net(nn.Module):
        # The __init__ function stack the layers of the
        # network Sequentially
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


    baseline_net = Net(1, net_width).to(device)


    def init_normal(m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight)


    # use the modules apply function to recursively apply the initialization
    baseline_net.apply(init_normal)
    ############################################################
    optimizer = optim.Adam(baseline_net.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=10 ** -15)
    x_inner = np.random.uniform(0, 0.008, in_Pt_num)
    x_inner_reshape = x_inner.reshape(-1, 1)
    x_outer = np.random.uniform(0.008, 1, out_Pt_num)
    x_outer_reshape = x_outer.reshape(-1, 1)
    x_all = np.hstack((x_inner, x_outer))
    x_all_reshape = x_all.reshape(-1, 1)
    a = 1.
    b = 2.

    x_bd = np.array([0., 1.])
    y_bd = np.array([a, b])
    x_bd = x_bd.reshape(-1, 1)
    y_bd = y_bd.reshape(-1, 1)


    def Loss_eqn(x):
        a_x = tensor(a_coeff(x)).to(device)
        b_x = tensor(b_coeff(x)).to(device)
        x = tensor(x).to(device)
        x.requires_grad = True
        y = baseline_net(x)
        y = y.view(len(y), -1)
        y_x = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        y_xx = torch.autograd.grad(y_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        tmp = eps * y_xx + a_x * y_x + b_x * y
        criterion = nn.MSELoss()
        return criterion(tmp, torch.zeros_like(tmp))

    def Loss_BC(xb, yb):
        xb = torch.Tensor(xb).to(device)
        yb = torch.Tensor(yb).to(device)
        out = baseline_net(xb)
        y = out.view(len(out), -1)
        loss_f = nn.MSELoss()
        loss_bc = loss_f(y, yb)
        return loss_bc


    loss_ls = np.zeros(epochs)
    loss_eqn_ls = np.zeros(epochs)
    loss_bc_ls = np.zeros(epochs)

    best_loss = float('inf')
    for epoch in range(epochs):
        baseline_net.zero_grad()
        loss_eqn = Loss_eqn(x_all_reshape)
        loss_bc = Loss_BC(x_bd, y_bd)
        loss = loss_eqn + loss_bc
        loss.backward()
        optimizer.step()
        loss_ls[epoch] = loss.item()
        loss_eqn_ls[epoch] = loss_eqn.item()
        loss_bc_ls[epoch] = loss_bc.item()
        if loss < best_loss:
            best_loss = loss
            torch.save(baseline_net.state_dict(), path_best_model)
        if epoch % 1000 == 0:
            print('Train Epoch: {} \tLoss: {:.10f} Loss_eqn: {:.10f} Loss_bc: {:.10f} '
                  .format(epoch, loss.item(), loss_eqn.item(), loss_bc.item()))

    plt.figure()
    plt.plot(loss_ls, 'r', label='total loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(path + 'loss_{:.0f}.png'.format(lp + 1), dpi=600)

    baseline_net.load_state_dict(torch.load(path_best_model, weights_only=True))

    in_npt = 10000
    x_inner = np.linspace(0, 0.02, in_npt)
    out_npt = 100
    x_outer = np.linspace(0.03, 1, out_npt)
    x_all = np.hstack((x_inner, x_outer))
    x_all_reshape = x_all.reshape(-1, 1)
    y_all_true = numerical_sol(eps, a_coeff, b_coeff, a, b)
    y_all_pred = baseline_net(tensor(x_all_reshape).to(device)).reshape(-1)

    test_rel_l2 = mean_rel_l2_0(y_all_true, y_all_pred.cpu().detach().numpy())
    test_max = max_error_0(y_all_true, y_all_pred.cpu().detach().numpy())

    test_rel_l2_inner = mean_rel_l2_0(y_all_true[:10000], y_all_pred[:10000].cpu().detach().numpy())
    test_max_inner = max_error_0(y_all_true[:10000], y_all_pred[:10000].cpu().detach().numpy())

    with open('./baseline-PINN-5-200-silu/test error', 'w') as f:
        f.write('\nloop_{:.0f}\n'.format(lp + 1))
        f.write('test_rel_l2: ')
        f.write(str(test_rel_l2))
        f.write('\ntest max error:')
        f.write(str(test_max))
        f.write('\ntest_rel_l2_inner: ')
        f.write(str(test_rel_l2_inner))
        f.write('\ntest max error_inner:')
        f.write(str(test_max_inner))


        fig, ax = plt.subplots(1, 1)
        ax.plot(x_all, y_all_true[:], 'b--', label='Analytical solution', alpha=1.0)  # analytical
        ax.plot(x_all, y_all_pred[:].cpu().detach().numpy(), 'r-', label='PINN', alpha=1.)  # PINN
        ax.legend(loc='best')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 7)
        ax.margins(0)
        bbox = [0, 0, 1, 1]
        axins = inset_axes(ax, width='20%', height='60%', loc="center", bbox_to_anchor=bbox,
                           bbox_transform=ax.transAxes)
        axins.plot(x_all, y_all_true[:], 'b--', label='Analytical solution', alpha=1.0)  # analytical
        axins.plot(x_all, y_all_pred[:].cpu().detach().numpy(), 'r-', label='PINN', alpha=1.)  # PINN
        # axins.legend(loc='best')
        axins.set_xlim(0, 0.015)
        axins.set_ylim(2.5, 5.5)

        mark_inset(ax, axins, loc1=3, loc2=1, fc=(0.5, 0.5, 0.5, 0.3), ec=(0.5, 0.5, 0.5, 0.3), lw=1.0)
        plt.savefig(path + 'pred_{:.0f}.png'.format(lp + 1), dpi=600)

# plt.show()





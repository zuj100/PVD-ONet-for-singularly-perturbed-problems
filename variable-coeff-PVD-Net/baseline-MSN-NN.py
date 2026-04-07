import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch.optim as optim
import time
import torch.nn as nn
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import os
from scipy.interpolate import interp1d



torch.manual_seed(42)
np.random.seed(42)

if not os.path.exists('./baseline-MSM-NN-new-sample' + '/plots/'):
    os.makedirs('./baseline-MSM-NN-new-sample' + '/plots/')
path = './baseline-MSM-NN-new-sample' + '/plots/'

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
    return y


## Parameters###
eps = 0.001
net_width = 100
Pt_num = 201
batchsize = 50
learning_rate = 1e-4
epochs = 100000
loop = 1

test_rel_l2_loop = np.zeros(loop)
test_max_error_loop = np.zeros(loop)
test_rel_l2_inner_loop = np.zeros(loop)
test_max_error_inner_loop = np.zeros(loop)
outer_max_point_error_loop = np.zeros(loop)

for lp in range(loop):
    if not os.path.exists('./baseline-MSM-NN-new-sample' + '/model/loop_{:.0f}/'.format(lp + 1)):
        os.makedirs('./baseline-MSM-NN-new-sample' + '/model/loop_{:.0f}/'.format(lp + 1))
    path_best_inner_model = './baseline-MSM-NN-new-sample' + '/model/loop_{:.0f}/best_inner_model.pth'.format(
        lp + 1)
    path_best_outer_model = './baseline-MSM-NN-new-sample' + '/model/loop_{:.0f}/best_outer_model.pth'.format(
        lp + 1)


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


    net_inner = Net(1, net_width).to(device)
    net_outer = Net(1, net_width).to(device)


    ###### Initialize the neural network using a standard method ##############
    def init_normal(m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight)


    # use the modules apply function to recursively apply the initialization
    net_inner.apply(init_normal)
    net_outer.apply(init_normal)
    ############################################################
    optimizer = optim.Adam(list(net_inner.parameters()) + list(net_outer.parameters()),
                           lr=learning_rate, betas=(0.9, 0.99), eps=10 ** -15)

    ###### Definte the PDE and physics loss here ##############
    z_inner = 1 - np.random.uniform(0, 1, Pt_num)
    z_inner_reshape = z_inner.reshape(-1, 1)
    x_outer = np.random.uniform(0, 1, Pt_num)
    x_outer_reshape = x_outer.reshape(-1, 1)

    x_inner_bdy = np.array([0.], dtype=np.float32).reshape(-1, 1)
    x_outer_bdy = np.array([1.], dtype=np.float32).reshape(-1, 1)
    x_match = np.exp(-1 / eps, dtype=np.float32).reshape(-1, 1)
    a = 1.
    b = 2.


    def eqn_outer(x):
        a_x = tensor(a_coeff(x)).to(device)
        b_x = tensor(b_coeff(x)).to(device)
        x = tensor(x).to(device)
        x.requires_grad = True
        y_outer = net_outer(x)
        y_outer = y_outer.view(len(y_outer), -1)
        y_outer_x = \
            torch.autograd.grad(y_outer, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        y_outer_xx = \
            torch.autograd.grad(y_outer_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        tmp = eps * y_outer_xx + a_x * y_outer_x + b_x * y_outer
        criterion = nn.MSELoss()
        return criterion(tmp, torch.zeros_like(tmp))


    def eqn_inner(x):
        a_x = tensor(a_coeff(-eps*np.log(x))).to(device)
        b_x = tensor(b_coeff(-eps*np.log(x))).to(device)
        x = tensor(x).to(device)
        x.requires_grad = True
        y_inner = net_inner(x)
        y_inner = y_inner.view(len(y_inner), -1)
        y_inner_x = \
            torch.autograd.grad(y_inner, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        y_inner_xx = \
            torch.autograd.grad(y_inner_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        tmp =x * y_inner_x + x**2 * y_inner_xx - x * a_x * y_inner_x + eps * y_inner * b_x
        criterion = nn.MSELoss()
        return criterion(tmp, torch.zeros_like(tmp))


    ###### Define boundary conditions ##############
    ###################################################################

    def Loss_BC_outer(xb):
        xb = torch.FloatTensor(xb).to(device)
        out = net_outer(xb)
        y = out.view(len(out), -1)
        loss_f = nn.MSELoss()
        loss_bc = loss_f(y, b * torch.ones_like(y))
        return loss_bc


    def Loss_BC_inner(xb):
        xb = torch.FloatTensor(xb).to(device)
        out = net_inner(xb)
        y = out.view(len(out), -1)
        loss_f = nn.MSELoss()
        loss_bc = loss_f(y, a * torch.ones_like(y))
        return loss_bc


    def Loss_BC_match(xb_out, yb_in):
        xb = torch.FloatTensor(xb_out).to(device)
        yb = torch.FloatTensor(yb_in).to(device)
        out = net_inner(yb)
        output_inner = out.view(len(out), -1)
        out = net_outer(xb)
        output_outer = out.view(len(out), -1)
        loss_f = nn.MSELoss()
        loss_bc = loss_f(output_inner, output_outer)
        return loss_bc


    ######## Main loop ###########

    tic = time.time()

    loss_eqn = np.zeros(epochs)
    loss_inner = np.zeros(epochs)
    loss_outer = np.zeros(epochs)
    best_loss = float('inf')
    for epoch in range(epochs):

        net_inner.zero_grad()
        net_outer.zero_grad()
        loss_eqn_outer = eqn_outer(x_outer_reshape)
        loss_eqn_inner = eqn_inner(z_inner_reshape)
        loss_bc_inner = Loss_BC_inner(x_outer_bdy)
        loss_bc_outer = Loss_BC_outer(x_outer_bdy)
        loss_bc_match = Loss_BC_match(x_inner_bdy, x_match)

        loss = loss_eqn_outer + loss_eqn_inner + loss_bc_inner + loss_bc_outer + loss_bc_match
        loss.backward()

        optimizer.step()

        loss_eqn[epoch] = loss.item()
        loss_inner[epoch] = loss_eqn_inner.item()
        loss_outer[epoch] = loss_eqn_outer.item()
        if loss < best_loss:
            best_loss = loss
            torch.save(net_outer.state_dict(), path_best_outer_model)
            torch.save(net_inner.state_dict(), path_best_inner_model)

            with open('./baseline-MSM-NN-new-sample' + '/model/loop_{:.0f}/best epoch'.format(lp + 1),
                      'w') as f:
                f.write('best epoch: ')
                f.write(str(epoch + 1))

        if epoch % 1000 == 0:
            print('Train Epoch: {} \tLoss: {:.10f} Loss_eqn-inner: {:.10f} Loss_eqn-outer: {:.10f} '
                  .format(epoch, loss.item(), loss_eqn_inner.item(), loss_eqn_outer.item()))
            print('Loss_bc_inner: {:.8f} Loss_bc_outer: {:.8f}  Loss_bc_match: {:.8f}  '
                  .format(loss_bc_inner.item(), loss_bc_outer.item(), loss_bc_match.item()))
            print('-----------------------------------------------------------------------------------')
    toc = time.time()
    elapseTime = toc - tic
    print("elapse time = ", elapseTime)

    plt.figure()
    plt.plot(loss_eqn, 'r', label='total loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(path + 'loss_{:.0f}.png'.format(lp + 1), dpi=600)

    net_outer.load_state_dict(torch.load(path_best_outer_model, weights_only=True))
    net_inner.load_state_dict(torch.load(path_best_inner_model, weights_only=True))

    x_inner = np.linspace(0, 0.02, 10000)
    npt = 100
    x_outer = np.linspace(0.03, 1, npt)
    x_all = np.hstack((x_inner, x_outer))
    z_inner = np.exp(-x_all / eps)
    z_inner_reshape = z_inner.reshape(-1, 1)
    
    x_max = np.array([0.003])
    z_max = np.exp(-x_max / eps)
    z_max_reshape = z_max.reshape(-1, 1)
    x_max_reshape = x_max.reshape(-1, 1)
    y_inner_max = net_inner(tensor(z_max_reshape).to(device)).reshape(-1)
    y_outer_max = net_outer(tensor(x_max_reshape).to(device)).reshape(-1)
    
    
    y_inner = net_inner(tensor(z_inner_reshape).to(device)).reshape(-1)
    y_outer = net_outer(tensor(x_all.reshape(-1, 1)).to(device)).reshape(-1)
    x_0 = tensor(x_inner_bdy)
    betae = net_outer(x_0.to(device))[0]
    y = y_inner.cpu().detach().numpy() + y_outer.cpu().detach().numpy() - betae.cpu().detach().numpy()
    y_numerical = numerical_sol(eps, a_coeff, b_coeff, a, b)
    y_junction_pred = y_inner_max.cpu().detach().numpy() + y_outer_max.cpu().detach().numpy() - betae.cpu().detach().numpy()
    
    y_interp = interp1d(x_all, y_numerical, kind='linear')
    y_junction_numerical = y_interp(x_max)
    junction_error = np.abs(y_junction_pred - y_junction_numerical)
    
    y_numerical_10101 = np.hstack((y_numerical, y_junction_numerical))
    y_pred_10101 = np.hstack((y, y_junction_pred))

    test_rel_l2 = mean_rel_l2_0(y_numerical_10101, y_pred_10101)
    test_max = max_error_0(y_numerical_10101, y_pred_10101)
    test_rel_l2_inner = mean_rel_l2_0(y_numerical[:10000], y[:10000])
    test_max_inner = max_error_0(y_numerical[:10000], y[:10000])

    test_rel_l2_loop[lp] = test_rel_l2
    test_max_error_loop[lp] = test_max
    test_rel_l2_inner_loop[lp] = test_rel_l2_inner
    test_max_error_inner_loop[lp] = test_max_inner


    with open('./baseline-MSM-NN-new-sample/test error', 'a') as f:
        f.write('\nloop_{:.0f}\n'.format(lp + 1))
        f.write('test_rel_l2: ')
        f.write(str(test_rel_l2))
        f.write('\ntest max error:')
        f.write(str(test_max))
        f.write('\ntest_rel_l2_inner: ')
        f.write(str(test_rel_l2_inner))
        f.write('\ntest max error_inner:')
        f.write(str(test_max_inner))
        f.write('\n junction point error:')
        f.write(str(junction_error))
       

    fig, ax = plt.subplots(1, 1)
    ax.plot(x_all, y_numerical[:], 'b--', label='Analytical solution', alpha=1.0)  # analytical
    ax.plot(x_all, y, 'r-', label='0-order approximate solution', alpha=1.)  # PINN
    ax.legend(loc='best')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 3)
    ax.margins(0)
    bbox = [0, 0, 1, 1]
    axins = inset_axes(ax, width='20%', height='60%', loc="center", bbox_to_anchor=bbox, bbox_transform=ax.transAxes)
    axins.plot(x_all, y_numerical[:], 'b--', label='Analytical solution', alpha=1.0)  # analytical
    axins.plot(x_all, y, 'r-', label='0-order approximate solution', alpha=1.)  # PINN
    # axins.legend(loc='best')
    axins.set_xlim(0, 0.015)
    axins.set_ylim(0.5, 2)

    mark_inset(ax, axins, loc1=3, loc2=1, fc=(0.5, 0.5, 0.5, 0.3), ec=(0.5, 0.5, 0.5, 0.3), lw=1.0)
    plt.savefig(path + 'pred_{:.0f}.png'.format(lp + 1), dpi=600)
    # plt.show()





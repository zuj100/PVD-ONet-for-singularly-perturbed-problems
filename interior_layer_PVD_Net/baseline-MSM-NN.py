import torch
import numpy as np
import matplotlib
from scipy.integrate import solve_bvp

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch.optim as optim
import time
import torch.nn as nn
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import os

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


## Parameters###
eps = 0.001
net_width = 110
Pt_num = 201
batchsize = 50
learning_rate = 1e-4
epochs = 100000
loop = 1
x0=1/2

for lp in range(loop):
    if not os.path.exists('./baseline-MSM-NN-new-sample' + '/model/loop_{:.0f}/'.format(lp + 1)):
        os.makedirs('./baseline-MSM-NN-new-sample' + '/model/loop_{:.0f}/'.format(lp + 1))
    path_best_inner_left_model = './baseline-MSM-NN-new-sample' + '/model/loop_{:.0f}/best_inner_left_model.pth'.format(
        lp + 1)
    path_best_inner_right_model = './baseline-MSM-NN-new-sample' + '/model/loop_{:.0f}/best_inner_right_model.pth'.format(
        lp + 1)
    path_best_outer_left_model = './baseline-MSM-NN-new-sample' + '/model/loop_{:.0f}/best_model.pth'.format(
        lp + 1)
    path_best_outer_right_model = './baseline-MSM-NN-new-sample' + '/model/loop_{:.0f}/best_outer_right_model.pth'.format(
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


    net_inner_left = Net(1, net_width).to(device)
    net_inner_right = Net(1, net_width).to(device)
    net_outer_left = Net(1, net_width).to(device)
    net_outer_right = Net(1, net_width).to(device)


    ###### Initialize the neural network using a standard method ##############
    def init_normal(m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight)


    # use the modules apply function to recursively apply the initialization
    net_inner_left.apply(init_normal)
    net_inner_right.apply(init_normal)
    net_outer_left.apply(init_normal)
    net_outer_right.apply(init_normal)
    ############################################################
    optimizer = optim.Adam(
        list(net_inner_left.parameters()) + list(net_inner_right.parameters())
        +list(net_outer_left.parameters()) + list(net_outer_right.parameters()),
        lr=learning_rate, betas=(0.9, 0.99), eps=10 ** -15)

    ###### Definte the PDE and physics loss here ##############
    z_inner = 1 - np.random.uniform(0, 1, Pt_num)
    z_inner_reshape = z_inner.reshape(-1, 1)

    x_outer_left = np.random.uniform(0, x0, Pt_num)
    x_outer_left_reshape = x_outer_left.reshape(-1, 1)
    x_outer_right = np.random.uniform(x0, 1, Pt_num)
    x_outer_right_reshape = x_outer_right.reshape(-1, 1)

    x_inner_bdy = np.array([0.], dtype=np.float32).reshape(-1, 1)
    x_outer_bdy = np.array([1.], dtype=np.float32).reshape(-1, 1)
    x_match_left = np.exp(-1 / (2*eps), dtype=np.float32).reshape(-1, 1)
    x_match_right = np.exp(-1 / (2 * eps), dtype=np.float32).reshape(-1, 1)
    x_0 = np.array([1 / 2.], dtype=np.float32).reshape(-1, 1)
    a = 1.
    b = -1.


    def eqn_outer_left(x):
        x = torch.Tensor(x).to(device)
        x.requires_grad = True
        y_outer = net_outer_left(x)
        y_outer = y_outer.view(len(y_outer), -1)
        y_outer_x = \
            torch.autograd.grad(y_outer, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        y_outer_xx = \
            torch.autograd.grad(y_outer_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        tmp = eps * y_outer_xx -y_outer* y_outer_x + y_outer
        criterion = nn.MSELoss()
        return criterion(tmp, torch.zeros_like(tmp))

    def eqn_outer_right(x):
        x = torch.Tensor(x).to(device)
        x.requires_grad = True
        y_outer = net_outer_right(x)
        y_outer = y_outer.view(len(y_outer), -1)
        y_outer_x = \
            torch.autograd.grad(y_outer, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        y_outer_xx = \
            torch.autograd.grad(y_outer_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        tmp = eps * y_outer_xx -y_outer* y_outer_x + y_outer
        criterion = nn.MSELoss()
        return criterion(tmp, torch.zeros_like(tmp))

    def eqn_inner_left(x):
        x = torch.Tensor(x).to(device)
        x.requires_grad = True
        y_inner = net_inner_left(x)
        y_inner = y_inner.view(len(y_inner), -1)
        y_inner_x = \
            torch.autograd.grad(y_inner, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        y_inner_xx = \
            torch.autograd.grad(y_inner_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        tmp = x ** 2 * y_inner_xx +x*y_inner_x+ eps * y_inner-y_inner*x*y_inner_x
        criterion = nn.MSELoss()
        return criterion(tmp, torch.zeros_like(tmp))

    def eqn_inner_right(x):
        x = torch.Tensor(x).to(device)
        x.requires_grad = True
        y_inner = net_inner_right(x)
        y_inner = y_inner.view(len(y_inner), -1)
        y_inner_x = \
            torch.autograd.grad(y_inner, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        y_inner_xx = \
            torch.autograd.grad(y_inner_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        tmp = x ** 2 * y_inner_xx +x*y_inner_x+ eps * y_inner+y_inner*x*y_inner_x
        criterion = nn.MSELoss()
        return criterion(tmp, torch.zeros_like(tmp))

    ###### Define boundary conditions ##############
    ###################################################################

    def Loss_BC_outer_left(xb):
        xb = torch.FloatTensor(xb).to(device)
        out = net_outer_left(xb)
        cNN = out.view(len(out), -1)
        loss_f = nn.MSELoss()
        loss_bc = loss_f(cNN, a * torch.ones_like(cNN))
        return loss_bc

    def Loss_BC_outer_right(xb):
        xb = torch.FloatTensor(xb).to(device)
        out = net_outer_right(xb)
        cNN = out.view(len(out), -1)
        loss_f = nn.MSELoss()
        loss_bc = loss_f(cNN, b * torch.ones_like(cNN))
        return loss_bc




    def Loss_BC_match(xb, yb_plus,yb_minus):
        xb = torch.FloatTensor(xb).to(device)
        yb_plus = torch.FloatTensor(yb_plus).to(device)
        yb_minus = torch.FloatTensor(yb_minus).to(device)

        out = net_inner_left(yb_minus)
        output_inner_left = out.view(len(out), -1)

        out = net_inner_right(yb_plus)
        output_inner_right = out.view(len(out), -1)

        out = net_outer_left(xb)
        output_outer_left = out.view(len(out), -1)

        out = net_outer_right(xb)
        output_outer_right = out.view(len(out), -1)

        loss_f = nn.MSELoss()
        loss_bc = loss_f(output_inner_left, output_outer_left)+loss_f(output_inner_right, output_outer_right)
        return loss_bc


    ####### Main loop ###########

    tic = time.time()

    loss_eqn = np.zeros(epochs)

    best_loss=float('inf')
    for epoch in range(epochs):
        optimizer.zero_grad()


        loss_eqn_outer = eqn_outer_left(x_outer_left_reshape)+eqn_outer_right(x_outer_right_reshape)
        loss_eqn_inner = eqn_inner_left(z_inner_reshape)+eqn_inner_right(z_inner_reshape)
        loss_bc_outer = Loss_BC_outer_left(x_inner_bdy)+Loss_BC_outer_right(x_outer_bdy)
        loss_bc_match = Loss_BC_match(x_0,x_match_right,x_match_left)

        loss = loss_eqn_outer + loss_eqn_inner+ loss_bc_outer + loss_bc_match
        loss.backward()

        optimizer.step()

        loss_eqn[epoch] = loss.item()

        if loss<best_loss:
            best_loss=loss
            torch.save(net_outer_left.state_dict(),path_best_outer_left_model)
            torch.save(net_outer_right.state_dict(), path_best_outer_right_model)
            torch.save(net_inner_left.state_dict(),path_best_inner_left_model)
            torch.save(net_inner_right.state_dict(), path_best_inner_right_model)



        if epoch % 1000 == 0:
            print('Train Epoch: {} \tLoss: {:.10f} Loss_eqn-inner: {:.10f} Loss_eqn-outer: {:.10f} '
                  .format(epoch, loss.item(), loss_eqn_inner.item(), loss_eqn_outer.item()))
            print(' Loss_bc_outer: {:.8f}  Loss_bc_match: {:.8f}  '
                  .format( loss_bc_outer.item(), loss_bc_match.item()))
            print('-----------------------------------------------------------------------------------')
    toc = time.time()
    elapseTime = toc - tic
    print("elapse time = ", elapseTime)

    plt.figure()
    plt.plot(loss_eqn, 'r', label='total loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(path + 'loss_{:.0f}.png'.format(lp + 1), dpi=600)

    net_outer_left.load_state_dict(torch.load(path_best_outer_left_model, weights_only=True))
    net_outer_right.load_state_dict(torch.load(path_best_outer_right_model, weights_only=True))
    net_inner_left.load_state_dict(torch.load(path_best_inner_left_model, weights_only=True))
    net_inner_right.load_state_dict(torch.load(path_best_inner_right_model, weights_only=True))

    x_inner = np.linspace(0.48, 0.52, 10000)
    npt = 100
    x_outer_left = np.linspace(0, 0.45, npt)
    x_outer_right = np.linspace(0.55, 1, npt)
    x_all = np.hstack((x_outer_left, x_inner, x_outer_right))
    x_all_left = np.hstack((x_outer_left, x_inner[:5000]))
    x_all_right = np.hstack((x_inner[-5000:], x_outer_right))

    z_inner_left=np.exp((x_all_left-x0)/eps)
    z_inner_right = np.exp(-(x_all_right- x0) / eps)

    z_inner_left = torch.tensor(z_inner_left.reshape(-1,1), dtype=torch.float32).to(device)
    z_inner_right = torch.tensor(z_inner_right.reshape(-1, 1), dtype=torch.float32).to(device)

    x_all_left_reshape=torch.tensor(x_all_left.reshape(-1,1), dtype=torch.float32).to(device)
    x_all_right_reshape = torch.tensor(x_all_right.reshape(-1, 1), dtype=torch.float32).to(device)

    y_inner_left = net_inner_left(z_inner_left).reshape(-1)
    y_inner_right = net_inner_right(z_inner_right).reshape(-1)

    y_outer_left = net_outer_left(x_all_left_reshape).reshape(-1)
    y_outer_right = net_outer_right(x_all_right_reshape).reshape(-1)

    x_0 = tensor(x_0)
    left = net_outer_left(x_0.to(device))[0]
    right = net_outer_right(x_0.to(device))[0]

    y_left = y_inner_left.cpu().detach().numpy() + y_outer_left.cpu().detach().numpy() - left.cpu().detach().numpy()
    y_right = y_inner_right.cpu().detach().numpy() + y_outer_right.cpu().detach().numpy() - right.cpu().detach().numpy()
    y = np.hstack((y_left, y_right))

    def solve_singular_bvp(t_eval, eps=0.001, ya=1.0, yb=-1.0):
        """
        输入:
            t_eval : array-like，需要计算解的点
            eps    : 小参数
            ya,yb  : 边界条件 y(0), y(1)

        输出:
            y_eval : 对应 t_eval 的数值解
        """

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

    y_numerical=solve_singular_bvp(x_all)

    test_rel_l2 = mean_rel_l2_0(y_numerical, y)
    test_max = max_error_0(y_numerical, y)

    test_rel_l2_inner = mean_rel_l2_0(y_numerical[100:10100], y[100:10100])
    test_max_inner = max_error_0(y_numerical[100:10100], y[100:10100])

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
        f.write('\ntotal time:')
        f.write(str(elapseTime))

    fig, ax = plt.subplots(1, 1)
    ax.plot(x_all,  y_numerical, 'b', label='Numerical solution', alpha=1.0)  # analytical
    ax.plot(x_all, y, 'r--', label='0-order approximate solution', alpha=1.)  # PINN
    ax.legend(loc='best')
    ax.set_xlim(0, 1)
    ax.set_ylim(-2, 2)
    ax.margins(0)
    bbox = [0, 0, 1, 1]
    axins = inset_axes(ax, width='20%', height='60%', loc="lower left", bbox_to_anchor=bbox, bbox_transform=ax.transAxes)
    axins.plot(x_all, y_numerical, 'b', label='Numerical solution', alpha=1.0)  # analytical
    axins.plot(x_all, y, 'r--', label='0-order approximate solution', alpha=1.)  # PINN
    # axins.legend(loc='best')
    axins.set_xlim(0.48, 0.55)
    axins.set_ylim(-1.8, 1.8)

    mark_inset(ax, axins, loc1=3, loc2=1, fc=(0.5, 0.5, 0.5, 0.3), ec=(0.5, 0.5, 0.5, 0.3), lw=1.0)
    plt.savefig(path + 'pred_{:.0f}.png'.format(lp + 1), dpi=600)
    # plt.show()





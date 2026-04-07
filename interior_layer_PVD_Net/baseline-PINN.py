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
from scipy.integrate import solve_bvp

torch.manual_seed(42)
np.random.seed(42)

if not os.path.exists('./baseline-PINN' + '/plots/'):
    os.makedirs('./baseline-PINN' + '/plots/')
path = './baseline-PINN' + '/plots/'
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
net_width = 300
Pt_num = 201
batchsize = 50
learning_rate = 1e-4
epochs = 100000
loop = 1
x0=1/2

for lp in range(loop):
    if not os.path.exists('./baseline-PINN' + '/model/loop_{:.0f}/'.format(lp + 1)):
        os.makedirs('./baseline-PINN' + '/model/loop_{:.0f}/'.format(lp + 1))
    path_best_model='./baseline-PINN'+'/model/loop_{:.0f}/best_model.pth'.format(lp + 1)


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


    net = Net(1, net_width).to(device)


    ###### Initialize the neural network using a standard method ##############
    def init_normal(m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight)


    # use the modules apply function to recursively apply the initialization
    net.apply(init_normal)
    ############################################################
    optimizer=optim.Adam(list(net.parameters()),
                         lr=learning_rate, betas=(0.9, 0.99), eps=10 ** -15)


    ###### Definte the PDE and physics loss here ##############


    x_outer = np.random.uniform(0, 1, Pt_num)
    x_outer_reshape = x_outer.reshape(-1, 1)
    x_outer_left_bdy = np.array([0.], dtype=np.float32).reshape(-1, 1)
    x_outer_right_bdy = np.array([1.], dtype=np.float32).reshape(-1, 1)
    a = 1.
    b = -1.



    def eqn_outer(x):
        x = torch.Tensor(x).to(device)
        x.requires_grad = True
        y_outer = net(x)
        y_outer = y_outer.view(len(y_outer), -1)
        y_outer_x = \
        torch.autograd.grad(y_outer, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        y_outer_xx = \
            torch.autograd.grad(y_outer_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]

        tmp = eps*y_outer_xx-y_outer*y_outer_x+y_outer
        criterion = nn.MSELoss()
        return criterion(tmp, torch.zeros_like(tmp))




    ###### Define boundary conditions ##############

    ###################################################################

    def Loss_BC_left(xb):
        xb = torch.FloatTensor(xb).to(device)
        out = net(xb)
        cNN = out.view(len(out), -1)
        loss_f = nn.MSELoss()
        loss_bc = loss_f(cNN, a* torch.ones_like(cNN))
        return loss_bc

    def Loss_BC_right(xb):
        xb = torch.FloatTensor(xb).to(device)
        out = net(xb)
        cNN = out.view(len(out), -1)
        loss_f = nn.MSELoss()
        loss_bc = loss_f(cNN, b* torch.ones_like(cNN))
        return loss_bc







    ###### Main loop ###########

    tic = time.time()

    loss_eqn_list = np.zeros(epochs)

    best_loss=float('inf')
    for epoch in range(epochs):
        net.zero_grad()

        loss_eqn= eqn_outer(x_outer_reshape)
        loss_bc_left= Loss_BC_left(x_outer_left_bdy)
        loss_bc_right = Loss_BC_right(x_outer_right_bdy)

        loss = (loss_eqn + loss_bc_left + loss_bc_right)
        loss.backward()

        optimizer.step()

        loss_eqn_list[epoch] = loss.item()
        if loss<best_loss:
            best_loss=loss
            torch.save(net.state_dict(),path_best_model)
        if epoch % 1000 == 0:
            print('Train Epoch: {} \tLoss: {:.10f} Loss_eqn: {:.10f} Loss_bc_left: {:.10f}  Loss_bc_right: {:.10f}'
                  .format(epoch, loss.item(), loss_eqn.item(), loss_bc_left.item(), loss_bc_right.item()))
            print('-----------------------------------------------------------------------------------')
    toc = time.time()
    elapseTime = toc - tic
    print("elapse time = ", elapseTime)

    plt.figure()
    plt.plot(loss_eqn_list, 'r', label='total loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(path + 'loss_{:.0f}.png'.format(lp + 1), dpi=600)

    net.load_state_dict(torch.load(path_best_model,weights_only=True))


    z_inner = np.linspace(-20, 20, 10000)
    x_inner = z_inner * eps+x0 # (0.48.0.52)
    npt = 100
    x_outer_left = np.linspace(0, 0.45, npt)
    x_outer_right = np.linspace(0.55, 1, npt)
    x_all = np.hstack((x_outer_left, x_inner, x_outer_right))

    x_all_reshape=tensor(x_all.reshape(-1,1)).to(device)

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
    y=net(x_all_reshape).reshape(-1)
    y=y.cpu().detach().numpy()


    test_rel_l2 = mean_rel_l2_0(y_numerical, y)
    test_max = max_error_0(y_numerical, y)

    test_rel_l2_inner = mean_rel_l2_0(y_numerical[100:10100], y[100:10100])
    test_max_inner = max_error_0(y_numerical[100:10100], y[100:10100])

    with open('baseline-PINN/test error', 'a') as f:
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
    ax.plot(x_all, y_numerical, 'b', label='Numerical solution', alpha=1.0 ,linewidth=3)  # Numerical
    ax.plot(x_all, y ,'r--', label='baseline-PINN approximate solution', alpha=1.,linewidth=3)  # PINN

    ax.legend(loc='best')
    ax.set_xlim(0, 1)
    ax.set_ylim(-2,2)
    ax.margins(0)
    bbox = [0, 0, 1, 1]
    axins = inset_axes(ax, width='40%', height='30%', loc="lower right", bbox_to_anchor=bbox, bbox_transform=ax.transAxes)
    axins.plot(x_all, y_numerical, 'b-', label='Numerical solution', alpha=1.0,linewidth=3)  # Numerical
    axins.plot(x_all, y, 'r--', label='baseline-PINN approximate solution', alpha=1.,linewidth=3)  # PINN
    # axins.legend(loc='best')
    axins.set_xlim(0.48, 0.52)
    axins.set_ylim(-1.8,1.8)

    mark_inset(ax, axins, loc1=3, loc2=1, fc=(0.5, 0.5, 0.5, 0.3), ec=(0.5, 0.5, 0.5, 0.3), lw=1.0)
    plt.savefig(path + 'pred_{:.0f}.png'.format(lp + 1), dpi=600)
    plt.show()







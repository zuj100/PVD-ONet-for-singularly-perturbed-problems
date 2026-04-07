import torch
import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim
import time
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.integrate import solve_bvp

if not os.path.exists('./baseline-BL-PINN-new-sample' + '/plots/'):
    os.makedirs('./baseline-BL-PINN-new-sample' + '/plots/')
path = './baseline-BL-PINN-new-sample' + '/plots/'

torch.manual_seed(42)
np.random.seed(42)

h_n = 150
input_n = 1


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_n, h_n),
            nn.SiLU(),
            nn.Linear(h_n, h_n),
            nn.SiLU(),
            nn.Linear(h_n, h_n),
            nn.SiLU(),
            nn.Linear(h_n, h_n),
            nn.SiLU(),
            nn.Linear(h_n, h_n),
            nn.SiLU(),
            nn.Linear(h_n, 1),
        )

    def forward(self, x):
        output = self.main(x)
        return output


if torch.cuda.is_available():
    print('cuda')
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    print('cpu')

## Parameters###
Diff = 0.001
nPt = 201
x0 = 1 / 2.

x_outer_left = np.random.uniform(0, x0, nPt).astype(np.float32)
x_outer_left_reshape = x_outer_left.reshape(-1, 1)
x_outer_right = np.random.uniform(x0, 1, nPt).astype(np.float32)
x_outer_right_reshape = x_outer_right.reshape(-1, 1)

xi_inner = np.random.uniform(-1, 1, nPt).astype(np.float32)
xi_inner = np.reshape(xi_inner, (nPt, 1))

a = 1.
b = -1.
xb = np.array([0., 1.], dtype=np.float32)
cb = np.array([a, b], dtype=np.float32)
xb = xb.reshape(-1, 1)
cb = cb.reshape(-1, 1)

xb_left = np.array([0.], dtype=np.float32)
xb_left = xb_left.reshape(-1, 1)
xb_right = np.array([1.], dtype=np.float32)
xb_right = xb_right.reshape(-1, 1)
x_20 = np.array([1.], dtype=np.float32).reshape(-1, 1)
x_0 = np.array([1 / 2.], dtype=np.float32).reshape(-1, 1)

learning_rate = 1e-4
inf_scale = 20.

loop = 1
epochs = 100000

for lp in range(loop):
    if not os.path.exists('./baseline-BL-PINN-new-sample' + '/model/loop_{:.0f}/'.format(lp + 1)):
        os.makedirs('./baseline-BL-PINN-new-sample' + '/model/loop_{:.0f}/'.format(lp + 1))
    path_best_inner_model = './baseline-BL-PINN-new-sample' + '/model/loop_{:.0f}/best_inner_model.pth'.format(lp + 1)
    path_best_outer_left_model = './baseline-BL-PINN-new-sample' + '/model/loop_{:.0f}/best_outer_left_model.pth'.format(lp + 1)
    path_best_outer_right_model = './baseline-BL-PINN-new-sample' + '/model/loop_{:.0f}/best_outer_right_model.pth'.format(lp + 1)

    net_inner = Net().to(device)
    net_outer_left = Net().to(device)
    net_outer_right = Net().to(device)


    ###### Initialize the neural network using a standard method ##############
    def init_normal(m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight)


    # use the modules apply function to recursively apply the initialization
    net_inner.apply(init_normal)
    net_outer_left.apply(init_normal)
    net_outer_right.apply(init_normal)

    ############################################################
    optimizer = optim.Adam(
        list(net_inner.parameters()) + list(net_outer_left.parameters()) + list(net_outer_right.parameters()),
        lr=learning_rate, betas=(0.9, 0.99), eps=10 ** -15)


    ###### Definte the PDE and physics loss here ##############
    def criterion_outer_left(x):
        x = torch.Tensor(x).to(device)
        x.requires_grad = True
        net_in = x
        y = net_outer_left(net_in)
        y = y.view(len(y), -1)
        y_x = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        loss_1 = y - y * y_x
        # MSE LOSS
        loss_f = nn.MSELoss()
        # Note our target is zero. It is residual so we use zeros_like
        loss = loss_f(loss_1, torch.zeros_like(loss_1))
        return loss


    def criterion_outer_right(x):
        x = torch.Tensor(x).to(device)
        x.requires_grad = True
        net_in = x
        y = net_outer_right(net_in)
        y = y.view(len(y), -1)
        y_x = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        loss_1 = y - y * y_x
        # MSE LOSS
        loss_f = nn.MSELoss()
        # Note our target is zero. It is residual so we use zeros_like
        loss = loss_f(loss_1, torch.zeros_like(loss_1))
        return loss


    def criterion_inner(x):
        x = torch.Tensor(x).to(device)
        x.requires_grad = True
        net_in = x
        y = net_inner(net_in)
        y = y.view(len(y), -1)
        y_x = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        y_x_outer = torch.autograd.grad(y_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        loss_1 = y_x_outer / inf_scale - y * y_x
        # MSE LOSS
        loss_f = nn.MSELoss()
        # Note our target is zero. It is residual so we use zeros_like
        loss = loss_f(loss_1, torch.zeros_like(loss_1))
        return loss


    ###### Define boundary conditions ##############
    def Loss_BC_outer_left(xb):
        xb = torch.Tensor(xb).to(device)
        out = net_outer_left(xb)
        cNN = out.view(len(out), -1)
        loss_f = nn.MSELoss()
        loss_bc = loss_f(cNN, a * torch.ones_like(cNN))
        return loss_bc


    def Loss_BC_outer_right(xb):
        xb = torch.Tensor(xb).to(device)
        out = net_outer_right(xb)
        cNN = out.view(len(out), -1)
        loss_f = nn.MSELoss()
        loss_bc = loss_f(cNN, b * torch.ones_like(cNN))
        return loss_bc


    def Loss_BC_match(xb, yb_plus, yb_minus):
        xb = torch.Tensor(xb).to(device)
        yb_plus = torch.Tensor(yb_plus).to(device)
        yb_minus = torch.Tensor(yb_minus).to(device)
        out = net_inner(yb_plus)
        output_inner_plus = out.view(len(out), -1)
        out = net_inner(yb_minus)
        output_inner_minus = out.view(len(out), -1)

        out = net_outer_left(xb)
        output_outer_left = out.view(len(out), -1)
        out = net_outer_right(xb)
        output_outer_right = out.view(len(out), -1)

        loss_f = nn.MSELoss()
        loss_bc = loss_f(output_inner_plus, output_outer_right) + loss_f(output_inner_minus, output_outer_left)
        return loss_bc


    tic = time.time()
    loss_eqn = np.zeros(epochs)
    loss_inner = np.zeros(epochs)
    loss_outer = np.zeros(epochs)
    best_loss = float('inf')
    for epoch in range(epochs):


        optimizer.zero_grad()

        loss_eqn_outer_left = criterion_outer_left(x_outer_left_reshape)
        loss_eqn_outer_right = criterion_outer_right(x_outer_right_reshape)
        loss_eqn_inner = criterion_inner(xi_inner)

        loss_bc_outer_left = Loss_BC_outer_left(xb_left)
        loss_bc_outer_right = Loss_BC_outer_right(xb_right)
        loss_bc_match = Loss_BC_match(x_0, x_20,-x_20)

        loss = (loss_eqn_outer_left+loss_eqn_outer_right+
                loss_eqn_inner+loss_bc_outer_left+loss_bc_outer_right+loss_bc_match)
        loss.backward()
        optimizer.step()
        loss_eqn[epoch] = loss.item()

        if loss < best_loss:
            best_loss=loss
            torch.save(net_outer_left.state_dict(),path_best_outer_left_model)
            torch.save(net_outer_right.state_dict(), path_best_outer_right_model)
            torch.save(net_inner.state_dict(),path_best_inner_model)

        if epoch % 1000 == 0:
            print('Train Epoch: {} \tLoss: {:.10f} Loss_eqn-inner: {:.10f} Loss_eqn-outer_left: {:.10f} '
                  .format(epoch, loss.item(), loss_eqn_inner.item(), loss_eqn_outer_left.item()))
            print('Loss_eqn-outer_right: {:.10f}  Loss_bc_outer_left: {:.8f}   Loss_bc_outer_left: {:.8f} Loss_bc_match: {:.8f}  '
                .format(loss_eqn_outer_right.item(), loss_bc_outer_left.item(), loss_bc_outer_right.item(),
                        loss_bc_match.item()))
            print('-----------------------------------------------------------------------------------')
    toc = time.time()
    elapseTime = toc - tic
    print("elapse time = ", elapseTime)
    ################

    plt.figure()
    plt.plot(loss_eqn, 'r', label='total loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(path + 'loss_{:.0f}.png'.format(lp + 1), dpi=600)

    net_outer_left.load_state_dict(torch.load(path_best_outer_left_model, weights_only=True))
    net_outer_right.load_state_dict(torch.load(path_best_outer_right_model, weights_only=True))
    net_inner.load_state_dict(torch.load(path_best_inner_model, weights_only=True))

    npt = 100
    z_inner = np.linspace(-20, 20, 10000)
    z_inner_reshape = z_inner.reshape(-1, 1)
    x_inner = z_inner * Diff + x0  # (0.48.0.52)

    x_inner_reshape = x_inner.reshape(-1, 1)
    x_inner_reshape = torch.tensor(x_inner_reshape, dtype=torch.float32)

    x_outer_left = np.linspace(0, 0.45, npt)
    x_outer_right = np.linspace(0.55, 1, npt)
    x_all = np.hstack((x_outer_left, x_inner, x_outer_right))

    x_outer_left_reshape = x_outer_left.reshape(-1, 1)
    x_outer_left_reshape = torch.tensor(x_outer_left_reshape, dtype=torch.float32)
    x_outer_right_reshape = x_outer_right.reshape(-1, 1)
    x_outer_right_reshape = torch.tensor(x_outer_right_reshape, dtype=torch.float32)

    x_all_reshape = x_all.reshape(-1, 1)
    x_all_reshape = torch.tensor(x_all_reshape, dtype=torch.float32)

    y_outer_left_5000 = net_outer_left(x_inner_reshape[:5000, :].to(device)).cpu().detach().numpy()
    y_outer_right_5000 = net_outer_right(x_inner_reshape[-5000:, :].to(device)).cpu().detach().numpy()

    y_outer_10000 = np.hstack((y_outer_left_5000.flatten(), y_outer_right_5000.flatten()))

    y_inner_10000 = net_inner((x_inner_reshape.to(device) - x0) / (inf_scale * Diff)).cpu().detach().numpy()

    y_left_5000 = np.minimum(y_inner_10000[:5000, :].flatten(), y_outer_10000[:5000])
    y_right_5000 = np.maximum(y_inner_10000[-5000:, :].flatten(), y_outer_10000[-5000:])
    y_10000 = np.hstack((y_left_5000, y_right_5000))

    y_outer_left_10100 = net_outer_left(x_all_reshape[:10100, :].to(device)).cpu().detach().numpy()
    y_outer_right_10100 = net_outer_right(x_all_reshape[-10100:, :].to(device)).cpu().detach().numpy()

    y_outer_left_10100 = y_outer_left_10100.flatten()
    y_outer_right_10100 = y_outer_right_10100.flatten()

    y_outer_left_100 = net_outer_left(x_outer_left_reshape.to(device)).cpu().detach().numpy()
    y_outer_right_100 = net_outer_right(x_outer_right_reshape.to(device)).cpu().detach().numpy()

    y_outer_left_100 = y_outer_left_100.flatten()
    y_outer_right_100 = y_outer_right_100.flatten()

    y_pred = np.hstack((y_outer_left_100, y_10000, y_outer_right_100))


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


    y_numerical = solve_singular_bvp(x_all)


    def mean_rel_l2(y_true, y_pred):
        return np.linalg.norm(y_true - y_pred, 2) / np.linalg.norm(y_true, 2)


    def max_error(y_true, y_pred):
        error = np.abs(y_true - y_pred)
        return np.max(error)


    idx = np.argmax(np.abs(y_numerical - y_pred))
    t_max = x_all[idx]
    max_error1 = np.abs(y_numerical[idx] - y_pred[idx])
    print(y_numerical[idx], y_pred[idx])

    print("最大误差:", max_error1)
    print("位置:", t_max)
    print("索引:", idx)

    inner_max = max_error(y_numerical[100:10100], y_10000)
    total_max = max_error(y_numerical, y_pred)
    inner_l2 = mean_rel_l2(y_numerical[100:10100], y_10000)
    total_l2 = mean_rel_l2(y_numerical, y_pred)

    with open('./baseline-BL-PINN-new-sample/test error', 'a') as f:
        f.write('\nloop_{:.0f}\n'.format(lp + 1))
        f.write('inner_rel_l2: ')
        f.write(str(inner_l2))
        f.write('\ninner max error:')
        f.write(str(inner_max))
        f.write('\ntotal_rel_l2: ')
        f.write(str(total_l2))
        f.write('\ntotal max error: ')
        f.write(str(total_max))
        # f.write('\ntotal time:')
        # f.write(str(elapseTime))

    # #### Plot ########

    fig, ax = plt.subplots(1, 1)
    ax.plot(x_all, y_numerical, '-', color='blue', label='Analytical solution', alpha=1.0, linewidth=3,
            zorder=0)  # analytical
    ax.plot(x_all, y_pred, '-', color='m', label='Analytical solution', alpha=1.0, linewidth=3,
            zorder=0)  # analytical
    # ax.plot(x_all[:10100], y_outer_left_10100, '--', label='BL-PINN left outer solution', alpha=1., linewidth=3,
    #          zorder=1, color='green')  # PINN
    # ax.plot(x_all[-10100:], y_outer_right_10100, '--', label='BL-PINN right outer solution', alpha=1., linewidth=3,
    #          zorder=1, color='orange')  # PINN
    # ax.plot(x_inner, y_inner_10000.flatten(), 'r--', label='BL-PINN inner solution', alpha=1., linewidth=3,
    #          zorder=1)  # PINN

    ax.set_xlim(0, 1)
    ax.set_ylim(-2, 2)
    ax.margins(0)

    axins = inset_axes(ax, width='40%', height='30%', loc="lower center", bbox_to_anchor=(0.005, 0.1, 0.6, 1),
                       bbox_transform=ax.transAxes)
    axins.plot(x_all, y_numerical, '-', color='blue', label='Analytical solution', alpha=1.0, linewidth=3,
               zorder=0)  # analytical
    axins.plot(x_all[:10100], y_outer_left_10100, '--', label='BL-PINN left outer solution', alpha=1., linewidth=3,
               zorder=1, color='green')  # PINN
    axins.plot(x_all[-10100:], y_outer_right_10100, '--', label='BL-PINN right outer solution', alpha=1., linewidth=3,
               zorder=1, color='orange')  # PINN
    axins.plot(x_inner, y_inner_10000.flatten(), 'r--', label='BL-PINN inner solution', alpha=1., linewidth=3,
               zorder=1)  # PINN
    axins.set_xlim(0.48, 0.52)
    axins.set_ylim(-1.8, 1.8)
    mark_inset(ax, axins, loc1=3, loc2=1, fc=(0.5, 0.5, 0.5, 0.3), ec=(0.5, 0.5, 0.5, 0.3), lw=1.0)
    plt.savefig(path + 'pred_{:.0f}.png'.format(lp + 1), dpi=600)
    plt.show()





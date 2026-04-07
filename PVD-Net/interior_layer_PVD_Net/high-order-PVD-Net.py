import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch.optim as optim
import torch.nn as nn
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import os
import time
from scipy.integrate import solve_bvp

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


## Parameters###
eps = 0.001
net_width =60
Pt_num = 201

learning_rate = 1e-4
num_point = 201
epochs = 100000
loop = 1
x0 = 1 / 2.

if not os.path.exists('./high-order-new-sample' + '/plots/'):
    os.makedirs('./high-order-new-sample' + '/plots/')

path = './high-order-new-sample' + '/plots/'


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


test_rel_l2_loop = np.zeros(loop)
test_max_error_loop = np.zeros(loop)
test_rel_l2_inner_loop = np.zeros(loop)
test_max_error_inner_loop = np.zeros(loop)
outer_max_point_error_loop = np.zeros(loop)
################################################################
for lp in range(loop):
    if not os.path.exists('./high-order-new-sample' + '/model/loop_{:.0f}/'.format(lp + 1)):
        os.makedirs('./high-order-new-sample' + '/model/loop_{:.0f}/'.format(lp + 1))
    path_best_outer_left0_model = './high-order-new-sample' + '/model/loop_{:.0f}/best_outer_left0_model.pth'.format(lp + 1)
    path_best_outer_left1_model = './high-order-new-sample' + '/model/loop_{:.0f}/best_outer_left1_model.pth'.format(lp + 1)
    path_best_outer_right0_model = './high-order-new-sample' + '/model/loop_{:.0f}/best_outer_right0_model.pth'.format(lp + 1)
    path_best_outer_right1_model = './high-order-new-sample' + '/model/loop_{:.0f}/best_outer_right1_model.pth'.format(lp + 1)
    path_best_inner0_model = './high-order-new-sample' + '/model/loop_{:.0f}/best_inner0_model.pth'.format(lp + 1)
    path_best_inner1_model = './high-order-new-sample' + '/model/loop_{:.0f}/best_inner1_model.pth'.format(lp + 1)
    path_best_inner_c_model = './high-order-new-sample' + '/model/loop_{:.0f}/best_inner_c_model.pth'.format(lp + 1)

    net_outer_left0 = Net(1, net_width).to(device)
    net_outer_left1 = Net(1, net_width).to(device)
    net_outer_right0 = Net(1, net_width).to(device)
    net_outer_right1 = Net(1, net_width).to(device)

    net_inner0 = Net(1, net_width).to(device)
    net_inner1 = Net(1, net_width).to(device)
    net_inner_c = Net(1, net_width).to(device)

    total_params = (
            sum(p.numel() for p in net_inner0.parameters())
            + sum(p.numel() for p in net_outer_left0.parameters())
            + sum(p.numel() for p in net_outer_right0.parameters())
            + sum(p.numel() for p in net_outer_left1.parameters())
            + sum(p.numel() for p in net_outer_right1.parameters()) +
            sum(p.numel() for p in net_inner1.parameters())
            + sum(p.numel() for p in net_inner_c.parameters()))
    print("Total parameters:", total_params)


    ###### Initialize the neural network using a standard method ##############
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

    optimizer = optim.Adam(list(net_outer_left0.parameters()) + list(net_outer_left1.parameters())
                           + list(net_outer_right0.parameters()) + list(net_outer_right1.parameters())
                           + list(net_inner0.parameters()) + list(net_inner1.parameters()) + list(net_inner_c.parameters()),
                           lr=learning_rate, betas=(0.9, 0.99), eps=10 ** -15)

    z_inner = np.random.uniform(-20, 20, num_point)
    z_inner_reshape = z_inner.reshape(-1, 1)
    x_outer_left = np.random.uniform(0, x0, Pt_num)
    x_outer_left_reshape = x_outer_left.reshape(-1, 1)
    x_outer_right = np.random.uniform(x0, 1, Pt_num)
    x_outer_right_reshape = x_outer_right.reshape(-1, 1)

    x_outer_left_bdy = np.array([0.], dtype=np.float32).reshape(-1, 1)
    x_outer_right_bdy = np.array([1.], dtype=np.float32).reshape(-1, 1)
    x_20 = np.array([20.], dtype=np.float32).reshape(-1, 1)
    x_0 = np.array([1 / 2.], dtype=np.float32).reshape(-1, 1)
    a = 1.
    b = -1.


    def loss_outer_left0(x):
        x = torch.Tensor(x).to(device)
        x.requires_grad = True
        net_in = x
        y = net_outer_left0(net_in)
        y = y.view(len(y), -1)
        y_x = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        loss_1 = y - y * y_x
        loss_f = nn.MSELoss()
        loss = loss_f(loss_1, torch.zeros_like(loss_1))
        return loss


    def loss_outer_left1(x):
        x = torch.Tensor(x).to(device)
        x.requires_grad = True
        net_in = x
        y0 = net_outer_left0(net_in)
        y1 = net_outer_left1(net_in)
        y0 = y0.view(len(y0), -1)
        y1 = y1.view(len(y1), -1)

        y0_x = torch.autograd.grad(y0, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        y0_xx = torch.autograd.grad(y0_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        y1_x = torch.autograd.grad(y1, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]

        loss_1 = y0_xx - y0 * y1_x - y0_x * y1 + y1

        loss_f = nn.MSELoss()
        loss = loss_f(loss_1, torch.zeros_like(loss_1))
        return loss


    def loss_outer_right0(x):
        x = torch.Tensor(x).to(device)
        x.requires_grad = True
        net_in = x
        y = net_outer_right0(net_in)
        y = y.view(len(y), -1)
        y_x = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        loss_1 = y - y * y_x
        loss_f = nn.MSELoss()
        loss = loss_f(loss_1, torch.zeros_like(loss_1))
        return loss


    def loss_outer_right1(x):
        x = torch.Tensor(x).to(device)
        x.requires_grad = True
        net_in = x
        y0 = net_outer_right0(net_in)
        y1 = net_outer_right1(net_in)
        y0 = y0.view(len(y0), -1)
        y1 = y1.view(len(y1), -1)

        y0_x = torch.autograd.grad(y0, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        y0_xx = torch.autograd.grad(y0_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        y1_x = torch.autograd.grad(y1, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]

        loss_1 = y0_xx - y0 * y1_x - y0_x * y1 + y1

        loss_f = nn.MSELoss()
        loss = loss_f(loss_1, torch.zeros_like(loss_1))
        return loss


    def loss_inner(x):
        x = torch.Tensor(x).to(device)
        x.requires_grad = True
        net_in = x
        Y0 = net_inner0(net_in)
        Y1 = net_inner_c(net_in)
        Y2 = net_inner1(net_in)

        Y0 = Y0.view(len(Y0), -1)
        Y1 = Y1.view(len(Y1), -1)
        Y2 = Y2.view(len(Y2), -1)
        Y = Y0 + eps * (Y2 + Y1 * x)
        Y_x = torch.autograd.grad(Y, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        Y_xx = torch.autograd.grad(Y_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        loss_1 = Y_xx - Y * Y_x + Y * eps
        loss_f = nn.MSELoss()
        loss = loss_f(loss_1, torch.zeros_like(loss_1))
        return loss


    def loss_bc_outer(x_0, x_1):
        x_0 = torch.FloatTensor(x_0).to(device)
        x_1 = torch.FloatTensor(x_1).to(device)
        y_left0 = net_outer_left0(x_0)
        y_left1 = net_outer_left1(x_0)
        y_right0 = net_outer_right0(x_1)
        y_right1 = net_outer_right1(x_1)

        y_left0 = y_left0.view(len(y_left0), -1)
        y_left1 = y_left1.view(len(y_left1), -1)
        y_right0 = y_right0.view(len(y_right0), -1)
        y_right1 = y_right1.view(len(y_right1), -1)

        loss_f = nn.MSELoss()
        loss_bc = (loss_f(y_left0, a * torch.ones_like(y_left0)) + loss_f(y_left1, 0 * torch.ones_like(y_left1))
                   + loss_f(y_right0, b * torch.ones_like(y_right0)) + loss_f(y_right1, 0 * torch.ones_like(y_right1)))
        return loss_bc


    def loss_bc_inner(x):
        x = torch.FloatTensor(x).to(device)
        y_inner0 = net_inner0(x)
        y_inner1 = net_inner1(x)
        y_inner0 = y_inner0.view(len(y_inner0), -1)
        y_inner1 = y_inner1.view(len(y_inner1), -1)
        loss_f = nn.MSELoss()
        loss_bc = loss_f(y_inner0, torch.zeros_like(y_inner0)) + loss_f(y_inner1, torch.zeros_like(y_inner1))
        return loss_bc


    def loss_Van_Dyke_match(xb, yb_plus, yb_minus):
        x = torch.FloatTensor(xb).to(device)
        yb_plus = torch.FloatTensor(yb_plus).to(device)
        yb_minus = torch.FloatTensor(yb_minus).to(device)
        x.requires_grad = True

        y_outer_left0 = net_outer_left0(x).view(len(x), -1)
        y_outer_left1 = net_outer_left1(x).view(len(x), -1)
        y_outer_right0 = net_outer_right0(x).view(len(x), -1)
        y_outer_right1 = net_outer_right1(x).view(len(x), -1)

        y_inner0_plus = net_inner0(yb_plus).view(len(yb_plus), -1)
        y_inner1_plus = net_inner1(yb_plus).view(len(yb_plus), -1)
        y_inner_c_plus = net_inner_c(yb_plus).view(len(yb_plus), -1)

        y_inner0_minus = net_inner0(yb_minus).view(len(yb_minus), -1)
        y_inner1_minus = net_inner1(yb_minus).view(len(yb_minus), -1)
        y_inner_c_minus = net_inner_c(yb_minus).view(len(yb_minus), -1)

        y_outer_left0_x = torch.autograd.grad(y_outer_left0, x, grad_outputs=torch.ones_like(x),
                                              create_graph=True, only_inputs=True)[0]
        y_outer_right0_x = torch.autograd.grad(y_outer_right0, x, grad_outputs=torch.ones_like(x),
                                               create_graph=True, only_inputs=True)[0]

        loss_f = nn.MSELoss()
        loss = (loss_f(y_outer_left0, y_inner0_minus) + loss_f(y_outer_left1, y_inner1_minus)
                + loss_f(y_outer_left0_x, y_inner_c_minus) + loss_f(y_outer_right0, y_inner0_plus)
                + loss_f(y_outer_right1, y_inner1_plus) + loss_f(y_outer_right0_x, y_inner_c_plus))
        return loss


    def minus_term_left(xb0, x):
        xb0 = torch.FloatTensor(xb0).to(device)
        x = torch.FloatTensor(x).to(device)
        xb0.requires_grad = True
        y_left0 = net_outer_left0(xb0).view(len(xb0), -1)
        y_left1 = net_outer_left1(xb0).view(len(xb0), -1)

        y_left0_x = \
        torch.autograd.grad(y_left0, xb0, grad_outputs=torch.ones_like(xb0), create_graph=True, only_inputs=True)[0]

        y = y_left0[0] + eps * y_left1[0] + y_left0_x[0] * (x - x0)
        return y


    def minus_term_right(xb0, x):
        xb0 = torch.FloatTensor(xb0).to(device)
        x = torch.FloatTensor(x).to(device)
        xb0.requires_grad = True
        y_right0 = net_outer_right0(xb0).view(len(xb0), -1)
        y_right1 = net_outer_right1(xb0).view(len(xb0), -1)

        y_right0_x = \
        torch.autograd.grad(y_right0, xb0, grad_outputs=torch.ones_like(xb0), create_graph=True, only_inputs=True)[0]

        y = y_right0[0] + eps * y_right1[0] + y_right0_x[0] * (x - x0)
        return y


    tic = time.time()
    loss_list = np.zeros(epochs)
    best_loss = float('inf')
    for epoch in range(epochs):

        net_outer_left0.zero_grad()
        net_outer_left1.zero_grad()
        net_outer_right0.zero_grad()
        net_outer_right1.zero_grad()
        net_inner0.zero_grad()
        net_inner1.zero_grad()
        net_inner_c.zero_grad()

        Loss_outer = (loss_outer_left0(x_outer_left_reshape) + loss_outer_left1(x_outer_left_reshape)
                      + loss_outer_right0(x_outer_right_reshape) + loss_outer_right1(x_outer_right_reshape))
        Loss_inner = loss_inner(z_inner_reshape)
        Loss_bc = loss_bc_outer(x_outer_left_bdy, x_outer_right_bdy)
        Loss_match = loss_Van_Dyke_match(x_0, x_20, -x_20)

        loss = Loss_outer + Loss_inner + Loss_bc + Loss_match
        loss.backward()

        optimizer.step()

        loss_list[epoch] = loss.item()
        if loss < best_loss:
            best_loss = loss
            torch.save(net_outer_left0.state_dict(), path_best_outer_left0_model)
            torch.save(net_outer_left1.state_dict(), path_best_outer_left1_model)
            torch.save(net_outer_right0.state_dict(), path_best_outer_right0_model)
            torch.save(net_outer_right1.state_dict(), path_best_outer_right1_model)
            torch.save(net_inner0.state_dict(), path_best_inner0_model)
            torch.save(net_inner1.state_dict(), path_best_inner1_model)
            torch.save(net_inner_c.state_dict(), path_best_inner_c_model)

            with open('./high-order-new-sample' + '/model/loop_{:.0f}/best epoch'.format(lp + 1), 'w') as f:
                f.write('best epoch: ')
                f.write(str(epoch + 1))

        if epoch % 1000 == 0:
            print('Train Epoch: {} \tLoss: {:.10f} Loss_outer: {:.10f} '
                  .format(epoch, loss.item(), Loss_outer.item()))
            print('Loss_inner: {:.8f}  Loss_match: {:.8f} Loss_bc: {:.8f} '
                  .format(Loss_inner.item(), Loss_match.item(), Loss_bc.item()))
            print('-----------------------------------------------------------------------------------------')
    toc = time.time()
    elapseTime = toc - tic
    net_outer_left0.load_state_dict(torch.load(path_best_outer_left0_model, weights_only=True))
    net_outer_left1.load_state_dict(torch.load(path_best_outer_left1_model, weights_only=True))
    net_outer_right0.load_state_dict(torch.load(path_best_outer_right0_model, weights_only=True))
    net_outer_right1.load_state_dict(torch.load(path_best_outer_right1_model, weights_only=True))
    net_inner0.load_state_dict(torch.load(path_best_inner0_model, weights_only=True))
    net_inner1.load_state_dict(torch.load(path_best_inner1_model, weights_only=True))
    net_inner_c.load_state_dict(torch.load(path_best_inner_c_model, weights_only=True))

    plt.figure()
    plt.plot(loss_list, 'r', label='total loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(path + 'loss_{:.0f}.png'.format(lp + 1), dpi=600)

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

    xi = torch.tensor(z_inner_reshape, dtype=torch.float32).to(device)
    y_inner = net_inner0(xi) + eps * (xi * net_inner_c(xi) + net_inner1(xi))
    y_inner = y_inner.reshape(-1)
    y_outer_left = net_outer_left0(x_left_inner_gpu) + eps * net_outer_left1(x_left_inner_gpu)
    y_outer_right = net_outer_right0(x_right_inner_gpu) + eps * net_outer_right1(x_right_inner_gpu)
    y_outer_left = y_outer_left.reshape(-1)
    y_outer_right = y_outer_right.reshape(-1)

    match_left = minus_term_left(x_0, x_left_inner)
    match_right = minus_term_right(x_0, x_right_inner)

    match_left_100 = minus_term_left(x_0, x_outer_left)
    match_right_100 = minus_term_right(x_0, x_outer_right)

    y_inner_left = np.hstack((match_left_100.cpu().detach().numpy(), y_inner[:5000].cpu().detach().numpy()))
    y_inner_right = np.hstack((y_inner[-5000:].cpu().detach().numpy(), match_right_100.cpu().detach().numpy()))

    y_left = y_inner_left + y_outer_left.cpu().detach().numpy() - match_left.cpu().detach().numpy()
    y_right = y_inner_right + y_outer_right.cpu().detach().numpy() - match_right.cpu().detach().numpy()

    y = np.hstack((y_left, y_right))
    print(y.shape)


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

    test_rel_l2 = mean_rel_l2_0(y_numerical, y)
    test_max = max_error_0(y_numerical, y)

    test_rel_l2_inner = mean_rel_l2_0(y_numerical[100:10100], y[100:10100])
    test_max_inner = max_error_0(y_numerical[100:10100], y[100:10100])

    test_rel_l2_loop[lp] = test_rel_l2
    test_max_error_loop[lp] = test_max
    test_rel_l2_inner_loop[lp] = test_rel_l2_inner
    test_max_error_inner_loop[lp] = test_max_inner

    with open('./high-order-new-sample/test error', 'a') as f:
        f.write('\nloop_{:.0f}\n'.format(lp + 1))
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
    ax.plot(x_all, y_numerical, 'b-', label='Numerical solution', alpha=0.8, linewidth=3)  # Numerical
    ax.plot(x_all, y, 'r--', label='High-order approximate solution', alpha=1., linewidth=3)
    ax.legend(loc='upper right')
    ax.set_xlim(0, 1)
    ax.set_ylim(-2, 2)
    ax.margins(0)

    axins = inset_axes(ax, width='40%', height='30%', loc="center right", bbox_to_anchor=[0, 0, 1, 1],
                       bbox_transform=ax.transAxes)
    axins.plot(x_all, y_numerical, 'b-', label='Numerical solution', alpha=0.8, linewidth=3)  # Numerical
    axins.plot(x_all, y, 'r--', label='High-order approximate solution', alpha=1.0, linewidth=3)  # PINN
    axins.set_xlim(0.48, 0.52)
    axins.set_ylim(-1.8, 1.8)

    mark_inset(ax, axins, loc1=3, loc2=1, fc=(0.5, 0.5, 0.5, 0.3), ec=(0.5, 0.5, 0.5, 0.3), lw=1.0)
    plt.savefig(path + 'pred_{:.0f}.png'.format(lp + 1), dpi=600)
    plt.show()



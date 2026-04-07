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

if not os.path.exists('./lambda_trainable-less_data' + '/plots/'):
    os.makedirs('./lambda_trainable-less_data' + '/plots/')
path = './lambda_trainable-less_data' + '/plots/'
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

## Parameters###
eps = 0.001
net_width = 150
Pt_num = 201
batchsize = 50
learning_rate = 1e-4
epochs = 50000
loop = 1
x0=1/2

for lp in range(loop):
    if not os.path.exists('./lambda_trainable-less_data' + '/model/loop_{:.0f}/'.format(lp + 1)):
        os.makedirs('./lambda_trainable-less_data' + '/model/loop_{:.0f}/'.format(lp + 1))
    path_best_inner_model='./lambda_trainable-less_data'+'/model/loop_{:.0f}/best_inner_model.pth'.format(lp + 1)
    path_best_outer_left_model='./lambda_trainable-less_data'+'/model/loop_{:.0f}/best_model.pth'.format(lp + 1)
    path_best_outer_right_model = './lambda_trainable-less_data' + '/model/loop_{:.0f}/best_outer_right_model.pth'.format(lp + 1)
    path_lambda = './lambda_trainable-less_data' + '/model/loop_{:.0f}/lambda.pth'.format(lp + 1)


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
    net_outer_left = Net(1, net_width).to(device)
    net_outer_right = Net(1, net_width).to(device)


    ###### Initialize the neural network using a standard method ##############
    def init_normal(m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight)


    scale_lambda = torch.nn.Parameter(torch.tensor(2.0))

    # use the modules apply function to recursively apply the initialization
    net_inner.apply(init_normal)
    net_outer_left.apply(init_normal)
    net_outer_right.apply(init_normal)
    ############################################################
    optimizer=optim.Adam(list(net_inner.parameters())+list(net_outer_left.parameters())
                         +list(net_outer_right.parameters())+[scale_lambda],
                         lr=learning_rate, betas=(0.9, 0.99), eps=10 ** -15)


    ###### Definte the PDE and physics loss here ##############
    z_inner = np.random.uniform(-20, 20, 200)
    z_inner_reshape = z_inner.reshape(-1, 1)

    z_inner_tensor = torch.tensor(z_inner, dtype=torch.float32, device=device)
    eps_tensor = torch.tensor(eps, dtype=torch.float32, device=device)

    x_data = np.linspace(0.49, 0.51, 10)
    x_data_reshape = x_data.reshape(-1, 1)
    x_data_tensor = torch.tensor(x_data_reshape, dtype=torch.float32, device=device)

    x_outer_left = np.random.uniform(0, x0, Pt_num)
    x_outer_left_reshape = x_outer_left.reshape(-1, 1)
    x_outer_right = np.random.uniform(x0, 1, Pt_num)
    x_outer_right_reshape = x_outer_right.reshape(-1, 1)

    x_outer_left_bdy = np.array([0.], dtype=np.float32).reshape(-1, 1)
    x_outer_right_bdy = np.array([1.], dtype=np.float32).reshape(-1, 1)
    x_20 = np.array([20.], dtype=np.float32).reshape(-1, 1)
    x_0 = np.array([1/2.], dtype=np.float32).reshape(-1, 1)
    a = 1.
    b = -1.



    def eqn_outer_left(x):
        x = torch.Tensor(x).to(device)
        x.requires_grad = True
        y_outer = net_outer_left(x)
        y_outer = y_outer.view(len(y_outer), -1)
        y_outer_x = \
        torch.autograd.grad(y_outer, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        tmp =  y_outer - y_outer*y_outer_x
        criterion = nn.MSELoss()
        return criterion(tmp, torch.zeros_like(tmp))


    def eqn_outer_right(x):
        x = torch.Tensor(x).to(device)
        x.requires_grad = True
        y_outer = net_outer_right(x)
        y_outer = y_outer.view(len(y_outer), -1)
        y_outer_x = \
            torch.autograd.grad(y_outer, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        tmp = y_outer - y_outer * y_outer_x
        criterion = nn.MSELoss()
        return criterion(tmp, torch.zeros_like(tmp))


    def eqn_inner(x):
        x = torch.Tensor(x).to(device)
        x.requires_grad = True
        y_inner = net_inner(x)
        y_inner = y_inner.view(len(y_inner), -1)
        y_inner_x = \
        torch.autograd.grad(y_inner, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        y_inner_xx = \
        torch.autograd.grad(y_inner_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        tmp = y_inner_xx-y_inner*y_inner_x*(eps**(scale_lambda-1))+(eps**(2*scale_lambda-1))*y_inner
        criterion = nn.MSELoss()
        return criterion(tmp, torch.zeros_like(tmp))

    def data_inner(x,xin):  #x_data
        xin = xin.to(device)
        y_inner = net_inner(xin)
        y_inner = y_inner.view(len(y_inner), -1)
        y_data = solve_singular_bvp(x,eps,a,b)
        y_data = tensor(y_data).reshape(-1,1).to(device)
        criterion = nn.MSELoss()
        return criterion(y_inner,y_data)

    ###### Define boundary conditions ##############

    ###################################################################

    def Loss_BC_outer_left(xb):
        xb = torch.FloatTensor(xb).to(device)
        out = net_outer_left(xb)
        cNN = out.view(len(out), -1)
        loss_f = nn.MSELoss()
        loss_bc = loss_f(cNN, a* torch.ones_like(cNN))
        return loss_bc

    def Loss_BC_outer_right(xb):
        xb = torch.FloatTensor(xb).to(device)
        out = net_outer_right(xb)
        cNN = out.view(len(out), -1)
        loss_f = nn.MSELoss()
        loss_bc = loss_f(cNN, b* torch.ones_like(cNN))
        return loss_bc





    def Loss_BC_match(xb, yb_plus,yb_minus):
        xb = torch.FloatTensor(xb).to(device)
        yb_plus = torch.FloatTensor(yb_plus).to(device)
        yb_minus = torch.FloatTensor(yb_minus).to(device)
        out = net_inner(yb_plus)
        output_inner_plus = out.view(len(out), -1)
        out = net_inner(yb_minus)
        output_inner_minus = out.view(len(out), -1)

        out = net_outer_left(xb)
        output_outer_left = out.view(len(out), -1)
        out = net_outer_right(xb)
        output_outer_right = out.view(len(out), -1)

        loss_f = nn.MSELoss()
        loss_bc = loss_f(output_inner_plus,output_outer_right)+loss_f(output_inner_minus,output_outer_left)
        return loss_bc


    ###### Main loop ###########

    tic = time.time()

    loss_eqn = np.zeros(epochs)
    loss_inner = np.zeros(epochs)
    loss_outer_left = np.zeros(epochs)
    loss_outer_right = np.zeros(epochs)
    lambda_record = np.zeros(epochs)
    best_loss = float('inf')
    lambda_init = float(tensor(2.0))
    x0_tensor = torch.tensor(x0, dtype=torch.float32, device=device)

    for epoch in range(epochs):
        xi = (x_data_tensor-x0_tensor) / (eps_tensor ** scale_lambda)
        optimizer.zero_grad()
        loss_eqn_outer_left = eqn_outer_left(x_outer_left_reshape)
        loss_eqn_outer_right = eqn_outer_right(x_outer_right_reshape)
        loss_eqn_inner = eqn_inner(z_inner_reshape)
        loss_bc_outer_left= Loss_BC_outer_left(x_outer_left_bdy)
        loss_bc_outer_right = Loss_BC_outer_right(x_outer_right_bdy)
        loss_bc_match = Loss_BC_match(x_0, x_20,-x_20)
        loss_data=data_inner(x_data,xi)
        loss = (loss_eqn_outer_left + loss_eqn_inner
                + loss_bc_outer_left + loss_bc_match
                + loss_eqn_outer_right + loss_bc_outer_right+0.1*loss_data)
        loss.backward()
        optimizer.step()

        loss_eqn[epoch] = loss.item()
        loss_inner[epoch] = loss_eqn_inner.item()
        loss_outer_left[epoch] = loss_eqn_outer_left.item()
        loss_outer_right[epoch] = loss_eqn_outer_right.item()
        lambda_record[epoch] = scale_lambda.item()
        if loss<best_loss:
            best_loss=loss
            torch.save(net_outer_left.state_dict(),path_best_outer_left_model)
            torch.save(net_outer_right.state_dict(), path_best_outer_right_model)
            torch.save(net_inner.state_dict(),path_best_inner_model)
        current_lambda = scale_lambda.item()
        if current_lambda < lambda_init:
            lambda_init = current_lambda
            torch.save(scale_lambda.item(), path_lambda)
            with open('./lambda_trainable-less_data' + '/model/loop_{:.0f}/best lambda'.format(lp + 1), 'w') as f:
                f.write('best lambda: ')
                f.write(str(current_lambda))
        if epoch % 1000 == 0:
            print('Train Epoch: {} \tLoss: {:.10f} Loss_eqn-inner: {:.10f} Loss_eqn-outer_left: {:.10f} '
                  .format(epoch, loss.item(), loss_eqn_inner.item(), loss_eqn_outer_left.item()))
            print('Loss_eqn-outer_right: {:.10f}  Loss_bc_outer_left: {:.8f}   Loss_bc_outer_left: {:.8f} Loss_bc_match: {:.8f}  '
                  .format(loss_eqn_outer_left.item(), loss_bc_outer_left.item(), loss_bc_outer_left.item(),loss_bc_match.item()))
            print('-----------------------------------------------------------------------------------')
            print('scale_lambda: {:.6f}'.format(scale_lambda.item()))
            print('-----------------------------------------------------------------------------------')

    toc = time.time()
    elapseTime = toc - tic
    print("elapse time = ", elapseTime)
    with open('./lambda_trainable-less_data' + '/model/loop_{:.0f}/best lambda'.format(lp + 1), 'a') as f:
        f.write('\nlast lambda: ')
        f.write(str(scale_lambda.item()))

    plt.figure()
    plt.plot(loss_eqn, 'r', label='total loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(path + 'loss_{:.0f}.png'.format(lp + 1), dpi=600)

    plt.figure()
    plt.plot(lambda_record, 'r', linewidth=3, label='trainable $\lambda$')
    plt.axhline(y=1.0, color='b', linestyle='--', linewidth=3, label='Exact: $\lambda$=1')
    plt.ylim(0.5, 2.1)
    plt.legend()
    plt.savefig(path + 'lambda_{:.0f}.png'.format(lp+1), dpi=600)

    net_outer_left.load_state_dict(torch.load(path_best_outer_left_model,weights_only=True))
    net_outer_right.load_state_dict(torch.load(path_best_outer_right_model, weights_only=True))
    net_inner.load_state_dict(torch.load(path_best_inner_model,weights_only=True))


    z_inner = np.linspace(-20, 20, 10000)
    z_inner_reshape = z_inner.reshape(-1, 1)
    x_inner = z_inner * eps+x0 # (0.48.0.52)
    npt = 100
    x_outer_left = np.linspace(0, 0.45, npt)
    x_outer_right = np.linspace(0.55, 1, npt)
    x_all = np.hstack((x_outer_left, x_inner, x_outer_right))

    x_left_inner = np.hstack((x_outer_left, x_inner[:5000])) #5100
    x_right_inner = np.hstack((x_inner[-5000:], x_outer_right)) #5100

    y_inner = net_inner(torch.tensor(z_inner_reshape, dtype=torch.float32).to(device)).reshape(-1)
    y_outer_left = net_outer_left(torch.tensor(x_left_inner.reshape(-1, 1), dtype=torch.float32).to(device)).reshape(-1) #5100
    y_outer_right = net_outer_right(torch.tensor(x_right_inner.reshape(-1, 1), dtype=torch.float32).to(device)).reshape(-1) #5100

    x_0 = tensor(x_0)
    left = net_outer_left(x_0.to(device))[0]
    right = net_outer_right(x_0.to(device))[0]

    supp_left = left * torch.ones(npt).to(device)
    supp_right = right * torch.ones(npt).to(device)

    y_inner_left = np.hstack((supp_left.cpu().detach().numpy(), y_inner[:5000].cpu().detach().numpy()))
    y_inner_right = np.hstack((y_inner[-5000:].cpu().detach().numpy(),supp_right.cpu().detach().numpy()))

    y_left = y_inner_left + y_outer_left.cpu().detach().numpy() - left.cpu().detach().numpy()
    y_right = y_inner_right + y_outer_right.cpu().detach().numpy() - right.cpu().detach().numpy()

    y=np.hstack((y_left,y_right))


    y_numerical=solve_singular_bvp(x_all)

    idx = np.argmax(np.abs(y_numerical - y))
    t_max = x_all[idx]
    max_error = np.abs(y_numerical[idx] - y[idx])


    print("最大误差:", max_error)
    print("位置:", t_max)
    print("索引:", idx)
    print(y_inner[5000])

    test_rel_l2 = mean_rel_l2_0(y_numerical, y)
    test_max = max_error_0(y_numerical, y)
    #
    test_rel_l2_inner = mean_rel_l2_0(y_numerical[100:10100], y[100:10100])
    test_max_inner = max_error_0(y_numerical[100:10100], y[100:10100])

    with open('trash/lambda_trainable-less_data/test error', 'a') as f:
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
    ax.plot(x_all, y_numerical, 'b--', label='Numerical solution', alpha=1.0)  # Numerical
    ax.plot(x_inner[:5000], y_inner[:5000].cpu().detach().numpy(), 'r-', label='0-order approximate solution', alpha=1.)  # PINN

    ax.legend(loc='best')
    ax.set_xlim(0, 1)
    ax.set_ylim(-2,2)
    ax.margins(0)
    bbox = [0, 0, 1, 1]
    axins = inset_axes(ax, width='10%', height='60%', loc="lower right", bbox_to_anchor=bbox, bbox_transform=ax.transAxes)
    axins.plot(x_all, y_numerical, 'b-', label='Numerical solution', alpha=1.0)  # Numerical
    axins.plot(x_all, y, 'r--', label='0-order approximate solution', alpha=1.)  # PINN
    # axins.legend(loc='best')
    axins.set_xlim(0.48, 0.55)
    axins.set_ylim(-1.8,1.8)

    mark_inset(ax, axins, loc1=3, loc2=1, fc=(0.5, 0.5, 0.5, 0.3), ec=(0.5, 0.5, 0.5, 0.3), lw=1.0)
    plt.savefig(path + 'pred_{:.0f}.png'.format(lp + 1), dpi=600)
    plt.show()







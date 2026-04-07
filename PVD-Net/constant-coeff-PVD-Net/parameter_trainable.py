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
#seed-1

torch.manual_seed(42)
np.random.seed(42)

if not os.path.exists('./lambda_trainable_less_point' + '/plots/'):
    os.makedirs('./lambda_trainable_less_point' + '/plots/')
path = './lambda_trainable_less_point' + '/plots/'
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

def exact_sol(eps, a, b, xf):
    eps = torch.tensor(eps, dtype=torch.float32, device=device)
    a = torch.tensor(a, dtype=torch.float32, device=device)
    b = torch.tensor(b, dtype=torch.float32, device=device)

    lambda1 = (-1 + torch.sqrt(1 - 4 * eps)) / (2 * eps)
    lambda2 = (-1 - torch.sqrt(1 - 4 * eps)) / (2 * eps)

    c1 = (-a * torch.exp(lambda2) + b) / (torch.exp(lambda1) - torch.exp(lambda2))
    c2 = (a * torch.exp(lambda1) - b) / (torch.exp(lambda1) - torch.exp(lambda2))

    sol = c1 * torch.exp(lambda1 * xf) + c2 * torch.exp(lambda2 * xf)

    return sol

## Parameters###
eps = 0.001
net_width = 100
Pt_num = 201
batchsize = 50
learning_rate = 1e-4
epochs = 100000
loop = 1
test_rel_l2_loop=np.zeros(loop)
test_max_error_loop=np.zeros(loop)
test_rel_l2_inner_loop=np.zeros(loop)
test_max_error_inner_loop=np.zeros(loop)
outer_max_point_error_loop=np.zeros(loop)

for lp in range(loop):
    if not os.path.exists('./lambda_trainable_less_point' + '/model/loop_{:.0f}/'.format(lp + 1)):
        os.makedirs('./lambda_trainable_less_point' + '/model/loop_{:.0f}/'.format(lp + 1))
    path_best_inner_model='./lambda_trainable_less_point'+'/model/loop_{:.0f}/best_inner_model.pth'.format(lp + 1)
    path_best_outer_model='./lambda_trainable_less_point'+'/model/loop_{:.0f}/best_outer_model.pth'.format(lp + 1)

    path_lambda='./lambda_trainable_less_point'+'/model/loop_{:.0f}/lambda.pth'.format(lp + 1)
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

    #mu=torch.nn.Parameter(torch.tensor(0.0))

    scale_lambda = torch.nn.Parameter(torch.tensor(2.0))

    # use the modules apply function to recursively apply the initialization
    net_inner.apply(init_normal)
    net_outer.apply(init_normal)
    ############################################################
    optimizer=optim.Adam(list(net_inner.parameters())+list(net_outer.parameters())+ [scale_lambda],
                        lr=learning_rate, betas=(0.9, 0.99), eps=10 ** -15)

    ###### Definte the PDE and physics loss here ##############
    z_inner = np.random.uniform(0, 20, Pt_num)
    z_inner_reshape = z_inner.reshape(-1, 1)

    z_inner_tensor = torch.tensor(z_inner, dtype=torch.float32, device=device)
    eps_tensor= torch.tensor(eps, dtype=torch.float32, device=device)
    x_inner = z_inner_tensor * eps_tensor ** (scale_lambda) # 0.02

    x_outer = np.random.uniform(0, 1, Pt_num)
    x_outer_reshape = x_outer.reshape(-1, 1)

    x_data=np.linspace(0,0.01,10)
    x_data_reshape=x_data.reshape(-1,1)
    x_data_tensor=torch.tensor(x_data_reshape, dtype=torch.float32, device=device)

    x_inner_bdy = np.array([0.], dtype=np.float32).reshape(-1, 1)
    x_outer_bdy = np.array([1.], dtype=np.float32).reshape(-1, 1)
    x_20 = np.array([20.], dtype=np.float32).reshape(-1, 1)
    a = 1.
    b = 2.



    def eqn_outer(x):
        x = torch.Tensor(x).to(device)
        x.requires_grad = True
        y_outer = net_outer(x)
        y_outer = y_outer.view(len(y_outer), -1)
        y_outer_x = \
        torch.autograd.grad(y_outer, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        tmp = y_outer_x + y_outer
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
        tmp = y_inner_xx  + eps**(scale_lambda-1)* y_inner_x+eps**(2*scale_lambda-1)*y_inner
        criterion = nn.MSELoss()
        return criterion(tmp, torch.zeros_like(tmp))

    def data_inner(x,xin):
        xin = xin.to(device)
        y_inner = net_inner(xin)
        y_inner = y_inner.view(len(y_inner), -1)
        y_data= exact_sol(eps,a,b,x).reshape(-1,1)
        y_data = y_data.to(device)
        criterion = nn.MSELoss()
        return criterion(y_inner,y_data)

    ###### Define boundary conditions ##############

    ###################################################################

    def Loss_BC_outer(xb):
        xb = torch.FloatTensor(xb).to(device)
        out = net_outer(xb)
        cNN = out.view(len(out), -1)
        loss_f = nn.MSELoss()
        loss_bc = loss_f(cNN, b * torch.ones_like(cNN))
        return loss_bc


    def Loss_BC_inner(xb):
        xb = torch.FloatTensor(xb).to(device)
        out = net_inner(xb)
        cNN = out.view(len(out), -1)
        loss_f = nn.MSELoss()
        loss_bc = loss_f(cNN, a * torch.ones_like(cNN))
        return loss_bc


    def Loss_BC_match(xb, yb):
        xb = torch.FloatTensor(xb).to(device)
        yb = torch.FloatTensor(yb).to(device)
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
    lambda_record=np.zeros(epochs)
    best_loss=float('inf')
    lambda_init=float(tensor(2.0))
    for epoch in range(epochs):
        xi = x_data_tensor / (eps_tensor ** scale_lambda)
        optimizer.zero_grad()
        loss_eqn_outer = eqn_outer(x_outer_reshape)
        loss_eqn_inner = eqn_inner(z_inner_reshape)
        loss_bc_inner = Loss_BC_inner(x_inner_bdy)
        loss_bc_outer = Loss_BC_outer(x_outer_bdy)
        loss_bc_match = Loss_BC_match(x_inner_bdy, x_20)
        loss_data=data_inner(x_data_tensor,xi)
        loss = (loss_eqn_outer + loss_eqn_inner + loss_bc_inner
                + loss_bc_outer + loss_bc_match + 30*loss_data) #30
        loss.backward()

        optimizer.step()

        loss_eqn[epoch] = loss.item()
        loss_inner[epoch] = loss_eqn_inner.item()
        loss_outer[epoch] = loss_eqn_outer.item()
        lambda_record[epoch]=scale_lambda.item()

        if loss<best_loss:
            best_loss=loss
            torch.save(net_outer.state_dict(),path_best_outer_model)
            torch.save(net_inner.state_dict(),path_best_inner_model)
        current_lambda = scale_lambda.item()
        if current_lambda < lambda_init:
            lambda_init = current_lambda
            torch.save(scale_lambda.item(), path_lambda)
            with open('./lambda_trainable_less_point' + '/model/loop_{:.0f}/best lambda'.format(lp + 1), 'w') as f:
                f.write('best lambda: ')
                f.write(str(current_lambda))


        if epoch % 100 == 0:
            print('Train Epoch: {} \tLoss: {:.10f} Loss_eqn-inner: {:.10f} Loss_eqn-outer: {:.10f}'
                  .format(epoch, loss.item(), loss_eqn_inner.item(), loss_eqn_outer.item()))

            print('Loss_bc_inner: {:.8f} Loss_bc_outer: {:.8f}  Loss_bc_match: {:.8f}'
                  .format(loss_bc_inner.item(), loss_bc_outer.item(), loss_bc_match.item()))

            print('scale_lambda: {:.6f}'.format(scale_lambda.item()))
            print('-----------------------------------------------------------------------------------')

    toc = time.time()
    elapseTime = toc - tic

    with open('./lambda_trainable_less_point' + '/model/loop_{:.0f}/best lambda'.format(lp + 1), 'a') as f:
        f.write('\nlast lambda: ')
        f.write(str(scale_lambda.item()))

    plt.figure()
    plt.plot(loss_eqn, 'r', label='total loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(path + 'loss_{:.0f}.png'.format(lp+1), dpi=600)


    plt.figure()
    plt.plot(lambda_record, 'r', linewidth=3, label='trainable $\lambda$')
    plt.axhline(y=1.0, color='b', linestyle='--', linewidth=3, label='Exact: $\lambda$=1')
    plt.ylim(0.5, 2.1)
    plt.legend()
    plt.savefig(path + 'lambda_{:.0f}.png'.format(100000), dpi=600)


    net_outer.load_state_dict(torch.load(path_best_outer_model,weights_only=True))
    net_inner.load_state_dict(torch.load(path_best_inner_model,weights_only=True))

    x_max = np.array([0.005])
    z_max = x_max / eps
    z_max_reshape = z_max.reshape(-1, 1)
    x_max_reshape = x_max.reshape(-1, 1)
    y_inner_max = net_inner(torch.tensor(z_max_reshape, dtype=torch.float32).to(device)).reshape(-1)
    y_outer_max = net_outer(torch.tensor(x_max_reshape, dtype=torch.float32).to(device)).reshape(-1)
    lambda_value = torch.load(path_lambda, weights_only=True)

    z_inner = np.linspace(0, 20, 10000)
    z_inner_reshape = z_inner.reshape(-1, 1)
    x_inner = z_inner * (eps**lambda_value)
    print(x_inner[-1])
    npt = 100
    x_outer = np.linspace(0.03, 1, npt)
    x_all = np.hstack((x_inner, x_outer))
    y_inner = net_inner(torch.tensor(z_inner_reshape, dtype=torch.float32).to(device)).reshape(-1)
    y_outer = net_outer(torch.tensor(x_all.reshape(-1, 1), dtype=torch.float32).to(device)).reshape(-1)
    x_0 = tensor(x_inner_bdy)
    betae = net_outer(x_0.to(device))[0]
    supp_0 = betae * torch.ones(npt).to(device)
    y_inner_all = np.hstack((y_inner.cpu().detach().numpy(), supp_0.cpu().detach().numpy()))
    y = y_inner_all + y_outer.cpu().detach().numpy() - betae.cpu().detach().numpy()
    y_junction_pred = y_inner_max.cpu().detach().numpy() + y_outer_max.cpu().detach().numpy() - betae.cpu().detach().numpy()


    def exact_sol(eps, a, b, xf):
        lambda1 = (-1 + np.sqrt(1 - 4 * eps)) / (2 * eps)
        lambda2 = (-1 - np.sqrt(1 - 4 * eps)) / (2 * eps)
        alpha = a
        beta = b
        c1 = (-alpha * np.exp(lambda2) + beta) / (np.exp(lambda1) - np.exp(lambda2))
        c2 = (alpha * np.exp(lambda1) - beta) / (np.exp(lambda1) - np.exp(lambda2))
        sol = c1 * np.exp(lambda1 * xf) + c2 * np.exp(lambda2 * xf)
        return sol


    y_analytical = exact_sol(eps, a, b, x_all)
    y_junction_anal = exact_sol(eps, a, b, x_max)
    y_anal_10101 = np.hstack((y_analytical, y_junction_anal))
    y_pred_10101 = np.hstack((y, y_junction_pred))

    test_rel_l2 = mean_rel_l2_0(y_anal_10101, y_pred_10101)
    test_max = max_error_0(y_anal_10101, y_pred_10101)

    test_rel_l2_inner = mean_rel_l2_0(y_analytical[:10000], y[:10000])
    test_max_inner = max_error_0(y_analytical[:10000], y[:10000])
    test_junction_point = np.abs(y_junction_pred - y_junction_anal)

    test_rel_l2_loop[lp]=test_rel_l2
    test_max_error_loop[lp]=test_max
    test_rel_l2_inner_loop[lp]=test_rel_l2_inner
    test_max_error_inner_loop[lp]=test_max_inner
    outer_max_point_error_loop[lp]=test_junction_point[0]

    with open('lambda_trainable_less_point/test error', 'a') as f:
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
        f.write(str(test_junction_point))
        # f.write('\ntotal time:')
        # f.write(str(elapseTime))

    fig, ax = plt.subplots(1, 1)
    ax.plot(x_all, y_analytical[:], 'b-', label='Analytical solution', alpha=0.8,linewidth=2)  # analytical
    ax.plot(x_all, y, 'r--', label='0-order approximate solution', alpha=1.,linewidth=2)  # PINN
    ax.legend(loc='best')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 7)
    ax.margins(0)

    axins = inset_axes(ax, width='40%', height='50%', loc="lower center", bbox_to_anchor=(0.005, 0.1, 0.6, 1),
                       bbox_transform=ax.transAxes)
    axins.plot(x_all, y_analytical[:], 'b-', label='Analytical solution', alpha=0.8,linewidth=2)  # analytical
    axins.plot(x_all, y, 'r--', label='0-order approximate solution', alpha=1.,linewidth=2)  # PINN
    # axins.legend(loc='best')
    axins.set_xlim(0, 0.015)
    axins.set_ylim(5.2, 5.5)

    mark_inset(ax, axins, loc1=3, loc2=1, fc=(0.5, 0.5, 0.5, 0.3), ec=(0.5, 0.5, 0.5, 0.3), lw=1.0)
    plt.savefig(path + 'pred_{:.0f}.png'.format(lp + 1), dpi=600)
    plt.show()



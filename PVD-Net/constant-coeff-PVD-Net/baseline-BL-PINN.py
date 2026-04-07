import torch
import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim
import time
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

if not os.path.exists('./baseline-BL-PINN-new-sample' + '/plots/'):
    os.makedirs('./baseline-BL-PINN-new-sample' + '/plots/')
path = './baseline-BL-PINN-new-sample' + '/plots/'

torch.manual_seed(42)
np.random.seed(42)

h_n = 100
input_n = 1
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
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
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
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

Lambda_bc = 1.
## Parameters###
Diff = 0.001
nPt = 200


C_BC1 = 1.
C_BC2 = 2.
xx = np.random.uniform(0, 1, nPt).astype(np.float32)
xx = np.reshape(xx, (nPt, 1))

xb = np.array([0., 1.], dtype=np.float32)
cb = np.array([C_BC1, C_BC2], dtype=np.float32)
xb = xb.reshape(-1, 1)
cb = cb.reshape(-1, 1)

xb_inner = np.array([0.], dtype=np.float32)
xb_inner = xb_inner.reshape(-1, 1)
xb_outer = np.array([1.], dtype=np.float32)
xb_outer = xb_outer.reshape(-1, 1)

batchsize = 50
learning_rate = 1e-4

inf_scale = 20.
U_scale = 1.
loop = 1
epochs=100000
def exact_sol(eps, a, b, xf):
    lambda1 = (-1 + np.sqrt(1 - 4 * eps)) / (2 * eps)
    lambda2 = (-1 - np.sqrt(1 - 4 * eps)) / (2 * eps)
    alpha = a
    beta = b
    c1 = (-alpha * np.exp(lambda2) + beta) / (np.exp(lambda1) - np.exp(lambda2))
    c2 = (alpha * np.exp(lambda1) - beta) / (np.exp(lambda1) - np.exp(lambda2))
    sol = c1 * np.exp(lambda1 * xf) + c2 * np.exp(lambda2 * xf)
    return sol

for lp in range(loop):
    if not os.path.exists('./baseline-BL-PINN-new-sample' + '/model/loop_{:.0f}/'.format(lp + 1)):
        os.makedirs('./baseline-BL-PINN-new-sample' + '/model/loop_{:.0f}/'.format(lp + 1))
    path_best_inner_model='./baseline-BL-PINN-new-sample'+'/model/loop_{:.0f}/best_inner_model.pth'.format(lp + 1)
    path_best_outer_model='./baseline-BL-PINN-new-sample'+'/model/loop_{:.0f}/best_outer_model.pth'.format(lp + 1)

    net_inner = Net1().to(device)
    net_outer = Net2().to(device)
    

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
    def criterion_outer(x):
        x = torch.tensor(x).to(device)
        x.requires_grad = True
        net_in = x
        C = net_outer(net_in)
        C = C.view(len(C), -1)

        c_x = torch.autograd.grad(C, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        loss_1 = c_x + C
        # MSE LOSS
        loss_f = nn.MSELoss()
        # Note our target is zero. It is residual so we use zeros_like
        loss = loss_f(loss_1, torch.zeros_like(loss_1))
        return loss

    def criterion_inner(x):
        x = torch.tensor(x).to(device)
        x.requires_grad = True
        net_in = x
        C = net_inner(net_in)
        C = C.view(len(C), -1)
        c_x = torch.autograd.grad(C, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        c_xx = torch.autograd.grad(c_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        loss_1 = c_xx / inf_scale + c_x
        # MSE LOSS
        loss_f = nn.MSELoss()
        # Note our target is zero. It is residual so we use zeros_like
        loss = loss_f(loss_1, torch.zeros_like(loss_1))
        return loss

    ###### Define boundary conditions ##############
    def Loss_BC_outer(xb):
        xb = torch.tensor(xb).to(device)
        out = net_outer(xb)
        cNN = out.view(len(out), -1)
        loss_f = nn.MSELoss()
        loss_bc = loss_f(cNN, 2.0 / U_scale * torch.ones_like(cNN))
        return loss_bc

    def Loss_BC_inner(xb):
        xb = torch.tensor(xb).to(device)
        out = net_inner(xb)
        cNN = out.view(len(out), -1)
        loss_f = nn.MSELoss()
        loss_bc = loss_f(cNN, torch.ones_like(cNN))
        return loss_bc

    def Loss_BC_match(xb, yb):
        xb = torch.tensor(xb).to(device)
        yb = torch.tensor(yb).to(device)
        out = net_inner(yb)
        output_inner = out.view(len(out), -1)

        out = net_outer(xb)
        output_outer = out.view(len(out), -1)
        loss_f = nn.MSELoss()
        loss_bc = loss_f(output_inner, output_outer)
        return loss_bc

    tic = time.time()
    loss_eqn = np.zeros(epochs)
    loss_inner = np.zeros(epochs)
    loss_outer = np.zeros(epochs)
    best_loss = float('inf')
    for epoch in range(epochs):

        net_inner.zero_grad()
        net_outer.zero_grad()

        loss_eqn_outer = criterion_outer(xx)
        loss_eqn_inner = criterion_inner(xx)

        loss_bc_inner = Loss_BC_inner(xb_inner)
        loss_bc_outer = Loss_BC_outer(xb_outer)
        loss_bc_match = Loss_BC_match(xb_inner, xb_outer)

        loss = loss_eqn_outer + loss_eqn_inner + loss_bc_inner + loss_bc_outer + loss_bc_match
        loss.backward()
        optimizer.step()
        loss_eqn[epoch] = loss.item()
        loss_inner[epoch] = loss_eqn_inner.item()
        loss_outer[epoch] = loss_eqn_outer.item()
        if loss < best_loss:
            best_loss=loss
            torch.save(net_outer.state_dict(),path_best_outer_model)
            torch.save(net_inner.state_dict(),path_best_inner_model)

        if epoch % 10000 == 0:
            print('Train Epoch: {} \tLoss: {:.10f} Loss_eqn-inner: {:.10f} Loss_eqn-outer: {:.10f}'
                  .format(epoch,loss.item(),loss_eqn_inner.item(),loss_eqn_outer.item()))
            print('Loss_bc_inner: {:.8f} Loss_bc_outer: {:.8f}  Loss_bc_match: {:.8f}'
                  .format(loss_bc_inner.item(),loss_bc_outer.item(),loss_bc_match.item()))
            print('---------------------------------------------------------------------------------------------------')
    toc = time.time()
    elapseTime = toc - tic
    print("elapse time = ", elapseTime)
    ################

    plt.figure()
    plt.plot(loss_eqn, 'r', label='total loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(path + 'loss_{:.0f}.png'.format(lp + 1), dpi=600)

    net_outer.load_state_dict(torch.load(path_best_outer_model, weights_only=True))
    net_inner.load_state_dict(torch.load(path_best_inner_model, weights_only=True))

    npt = 100
    x_inner=np.linspace(0,0.02,10000)
    x_inner_reshape=x_inner.reshape(-1,1)
    x_inner_reshape=torch.tensor(x_inner_reshape, dtype=torch.float32)
    x_outer = np.linspace(0.03, 1, npt)
    x_outer_reshape = x_outer.reshape(-1,1)
    x_outer_reshape = torch.tensor(x_outer_reshape, dtype=torch.float32)

    x_all=np.hstack((x_inner,x_outer))
    x_all_reshape=x_all.reshape(-1,1)
    x_all_reshape = torch.tensor(x_all_reshape, dtype=torch.float32)

    y_outer_10000 = net_outer(x_inner_reshape.to(device)).cpu().detach().numpy()
    y_inner_10000 = net_inner(x_inner_reshape.to(device) / (inf_scale * Diff)).cpu().detach().numpy()
    y_10000 = np.minimum(y_inner_10000.flatten(),y_outer_10000.flatten())

    y_outer_10100 = net_outer(x_all_reshape.to(device)).cpu().detach().numpy()
    y_outer_10100 = y_outer_10100.flatten()

    y_outer_100=net_outer(x_outer_reshape.to(device)).cpu().detach().numpy()
    y_100=y_outer_100.flatten()
    y_pred=np.hstack((y_10000,y_100))
    y_anal = exact_sol(Diff, C_BC1, C_BC2, x_all)

    x_max = np.array([0.005])
    x_max_reshape = x_max.reshape(-1, 1)
    x_max_reshape = torch.tensor(x_max_reshape, dtype=torch.float32)
    max = net_inner(x_max_reshape.to(device) / (inf_scale * Diff))
    C_inner_max = max.cpu().data.numpy()
    outer_max = net_outer(x_max_reshape.to(device))
    C_outer_max =  outer_max.cpu().data.numpy()

    y_junction_anal = exact_sol(Diff, C_BC1, C_BC2, x_max)

    y_junction_pred=np.min([C_inner_max,C_outer_max])
    
    junction_error = np.abs(y_junction_pred - y_junction_anal)
   
    def mean_rel_l2(y_true, y_pred):
        return np.linalg.norm(y_true - y_pred, 2) / np.linalg.norm(y_true, 2)

    y_pred_10101=np.hstack((y_pred,y_junction_pred))
    y_anal_10101=np.hstack((y_anal,y_junction_anal))
    
    inner_rel_l2 = mean_rel_l2(y_anal[:10000], y_10000)
    total_rel_l2 = mean_rel_l2(y_anal_10101, y_pred_10101)
    def max_error(y_true, y_pred):
        error = np.abs(y_true - y_pred)
        return np.max(error)

    inner_max = max_error(y_anal[:10000], y_10000)
    total_max = max_error(y_anal_10101,y_pred_10101)

    with open('./baseline-BL-PINN-new-sample/test error', 'w') as f:
        f.write('\nloop_{:.0f}\n'.format(lp + 1))
        f.write('inner_rel_l2: ')
        f.write(str(inner_rel_l2))
        f.write('\ninner max error:')
        f.write(str(inner_max))
        f.write('\nJunction point error:')
        f.write(str(junction_error))
        f.write('\ntotal_rel_l2: ')
        f.write(str(total_rel_l2))
        f.write('\ntotal max error: ')
        f.write(str(total_max))
    # #### Plot ########
    plt.figure()
    plt.plot(x_all, y_anal, '-', label='Analytical solution', alpha=1.0, linewidth=5,zorder=0)  # analytical
    plt.plot(x_all, y_outer_10100, '--', label='BL-PINN outer solution', alpha=1.,linewidth=5,
             zorder=1, color='green')  # PINN
    plt.plot(x_inner, y_inner_10000, 'r--', label='BL-PINN inner solution', alpha=1.,linewidth=5,
             zorder=1)  # PINN
    ##plt.legend(loc='best')
    plt.margins(0)

    fig, ax = plt.subplots(1, 1)
    ax.plot(x_all, y_anal, 'b-', label='Analytical solution', alpha=0.8, zorder=0)  # analytical
    ax.plot(x_all, y_outer_10100, '--', label='BL-PINN outer solution', alpha=1., markersize=6,
            zorder=10, color='green')  # PINN
    ax.plot(x_inner, y_inner_10000, 'r--', label='BL-PINN inner solution', alpha=1., markersize=6,
            zorder=10)  # PINN
    #ax.legend(loc='best')

    ax.set_ylim(0, 5.5)
    ax.margins(0)

    axins = inset_axes(ax, width='40%', height='30%', loc="lower center", bbox_to_anchor=(0.005, 0.1, 0.6, 1),
                       bbox_transform=ax.transAxes)
    axins.plot(x_all, y_anal, '-', label='Analytical solution', alpha=0.8, zorder=0)  # analytical
    axins.plot(x_all, y_outer_10100, '--', label='BL-PINN outer solution', alpha=1., markersize=6,
               zorder=10, color='green')  # PINN
    axins.plot(x_inner, y_inner_10000, 'r--', label='BL-PINN inner solution', alpha=1., markersize=6,
               zorder=10)  # PINN
    axins.set_xlim(0, 0.01)
    axins.set_ylim(5, 5.5)
    mark_inset(ax, axins, loc1=3, loc2=1, fc=(0.5, 0.5, 0.5, 0.3), ec=(0.5, 0.5, 0.5, 0.3), lw=1.0)
    plt.savefig(path + 'pred_{:.0f}.png'.format(lp + 1), dpi=600)
    plt.show()





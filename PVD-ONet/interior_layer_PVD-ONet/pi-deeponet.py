import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib
from scipy.integrate import solve_bvp
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes,mark_inset
import torch.optim as optim

if torch.cuda.is_available():
    print(f"CUDA is available. Using device {torch.cuda.current_device()}: {torch.cuda.get_device_name()}")
else:
    print("CUDA is not available. Using CPU.")

class deeponet(nn.Module):
    def __init__(self,branch_size,trunk_size):
        super(deeponet,self).__init__()
        self.branch_size=branch_size
        self.trunk_size=trunk_size
        self.branch_net_ls=self.branch_net()
        self.trunk_net_ls=self.trunk_net()
    def branch_net(self):
        layers=[]
        prev_size = self.branch_size[0]
        for hidden_size in self.branch_size[1:-1]:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.SiLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, self.branch_size[-1]))
        ls = nn.Sequential(*layers)
        return ls
    def trunk_net(self):
        layers=[]
        prev_size = self.trunk_size[0]
        for hidden_size in self.trunk_size[1:-1]:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.SiLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, self.trunk_size[-1]))
        ls = nn.Sequential(*layers)
        return ls

    def forward(self,x_func,x_loc):
        y_func=x_func
        y_loc=x_loc
        y_func=self.branch_net_ls(y_func)
        y_loc=self.trunk_net_ls(y_loc)
        Y=torch.einsum('bi,bi->b',y_func,y_loc) #unaligned
        #Y=torch.einsum('bi,ni->bn',y_func,y_loc) #aligned
        return Y
    def loss(self,y_pred,y_true): #mse
        train_loss=torch.mean(torch.square(y_true-y_pred))
        return train_loss

def tensor(x):
    return torch.tensor(x,dtype=torch.float)

def mean_rel_l2(y_true,y_pred):
    return np.mean(np.linalg.norm(y_true-y_pred,2,axis=1)/
                      np.linalg.norm(y_true,2,axis=1))

def max_error(y_true,y_pred):
    error=np.max(np.abs(y_true-y_pred),axis=1)
    return np.mean(error)

torch.manual_seed(0)
np.random.seed(0)

if not os.path.exists('./pi-deeponet'+ '/plots/'):
    os.makedirs('./pi-deeponet'  + '/plots/')
if not os.path.exists('./pi-deeponet'+ '/model/'):
    os.makedirs('./pi-deeponet'  + '/model/')

path='./pi-deeponet' + '/plots/'

path_best_model = './pi-deeponet' + '/model/best_model.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num=600
ntrain=500
ntest=100

net_width = 260
## Parameters###
eps= 0.001
nPt = 1100
batchsize = 50  # 50
learning_rate = 1e-4
epochs=100000
mm=2

net= deeponet([mm,net_width,net_width,net_width,net_width,net_width],
                     [1,net_width,net_width,net_width,net_width,net_width]).to(device)

num_params = sum(p.numel() for p in net.parameters())
print("Total parameters:", num_params)

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)

# use the modules apply function to recursively apply the initialization
net.apply(init_normal)

############################################################
optimizer = optim.Adam(list(net.parameters()),
                       lr=learning_rate, betas=(0.9, 0.99), eps=10 ** -15)




a=np.random.uniform(0.6,1.0,(num,1))
b=-a
ab=np.hstack((a,b))
ab_train=ab[:ntrain,:]
ab_test=ab[-ntest:,:]


ab_train=tensor(ab_train)
ab_test=tensor(ab_test)




x_outer=np.random.uniform(0,1,nPt)
x_outer_reshape=x_outer.reshape(-1,1)
x_outer_reshape_train=tensor(x_outer_reshape[:ntrain,:])
x_outer_reshape_test=tensor(x_outer_reshape[-ntest:,:])

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset
                                               (ab_train,x_outer_reshape_train),
                                               batch_size=batchsize,
                                               shuffle=True)

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset
                                              (ab_test,x_outer_reshape_test),
                                              batch_size=batchsize,
                                              shuffle=False)

xb_0 = np.array([0.], dtype=np.float32)
xb_0 = xb_0.reshape(-1, 1)
xb_1 = np.array([1.], dtype=np.float32)
xb_1 = xb_1.reshape(-1, 1)
xb_20= np.array([20.], dtype=np.float32)
xb_20 = xb_20.reshape(-1, 1)
x_bd= np.array([1/2.], dtype=np.float32)
x_bd = x_bd.reshape(-1, 1)




def loss_eqn(ab,x):
    ab = ab.cuda()
    ab = ab.view(-1, 2)
    x = x.to(device)
    x.requires_grad = True
    net_in = x
    y = net(ab,net_in)
    y = y.reshape(-1,1)
    y_x = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
    y_xx = torch.autograd.grad(y_x, x, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]

    loss_1 = eps*y_xx-y*y_x+y
    loss_f = nn.MSELoss()
    loss = loss_f(loss_1, torch.zeros_like(loss_1))
    return loss


def loss_bc(ab,x_0,x_1):
    ab = ab.cuda()
    ab = ab.view(-1, 2)
    x_0 = np.tile(x_0,(ab.shape[0],1))
    x_0 = torch.FloatTensor(x_0).to(device)
    x_1 = np.tile(x_1, (ab.shape[0], 1))
    x_1 = torch.FloatTensor(x_1).to(device)
    y_out_left  = net(ab,x_0).reshape(-1,1)
    y_out_right = net(ab,x_1).reshape(-1,1)

    loss_f = nn.MSELoss()
    loss = loss_f(y_out_right,ab[:,1].reshape(-1,1))+loss_f(y_out_left,ab[:,0].reshape(-1,1))
    return loss


train_loss_ls=np.zeros(epochs)
test_loss_ls=np.zeros(epochs)
best_loss = float('inf')
tic = time.time()
for ep in range(epochs):
    net.train()
    train_mse = 0
    train_loss_eqn = 0
    train_loss_bc = 0
    for ab, x_out_train in train_loader:
        optimizer.zero_grad()
        Loss_eqn = loss_eqn(ab,x_out_train)
        Loss_bc = loss_bc(ab,xb_0,xb_1)
        loss = Loss_eqn+Loss_bc
        loss.backward()
        optimizer.step()
        train_mse += loss.item()
        train_loss_eqn += Loss_eqn.item()
        train_loss_bc += Loss_bc.item()

    net.eval()


    test_mse = 0
    test_loss_eqn= 0
    test_loss_bc = 0


    for ab,  x_out_test in test_loader:
        test_eqn = loss_eqn(ab, x_out_test)
        test_bc = loss_bc(ab, xb_0, xb_1)

        test_Loss = test_eqn+ test_bc

        test_mse += test_Loss.item()
        test_loss_eqn += test_eqn.item()
        test_loss_bc += test_bc.item()


    train_mse /= len(train_loader)
    train_loss_eqn /= len(train_loader)
    train_loss_bc /= len(train_loader)

    train_loss_ls[ep] = train_mse

    test_mse /= len(test_loader)
    test_loss_eqn /= len(test_loader)
    test_loss_bc /= len(test_loader)

    test_loss_ls[ep] = test_mse
    if test_mse < best_loss:
        best_loss = test_mse
        torch.save(net.state_dict(), path_best_model)

    if ep % 1000 == 0:
        print('Train Epoch: {} \tLoss: {:.10f} train_eqn: {:.10f} '
              .format(ep, train_mse, train_loss_eqn))
        print('train_bc: {:.8f} '
              .format( train_loss_bc))
        print('                                                                                         ')
        print('test Loss: {:.10f} test_eqn: {:.10f} '
              .format(test_mse, test_loss_eqn))
        print('test_bc: {:.8f}  '
              .format( test_loss_bc))
        print('-------------------------------------------------------------------------------------')
toc = time.time()
elapseTime = toc - tic
print("elapse time = ", elapseTime)

plt.figure()
plt.plot(train_loss_ls, 'r', label='train loss')
plt.plot(test_loss_ls, 'g', label='test loss')
plt.yscale('log')
plt.legend()
plt.savefig(path + 'loss.png', dpi=600)

net.load_state_dict(torch.load(path_best_model,weights_only=True))



x0=1/2
z_inner = np.linspace(-20, 20, 10000)
in_pt=10000
x_inner = z_inner * eps+x0 # (0.48.0.52)
npt = 100
x_outer_left = np.linspace(0, 0.45, npt)
x_outer_right = np.linspace(0.55, 1, npt)
x_all = np.hstack((x_outer_left, x_inner, x_outer_right))
x_all_reshape=torch.tensor(x_all[:,None],dtype=torch.float).to(device)


def solve_singular_bvp(t_eval,ab, eps=0.001):
    ya = ab[0]
    yb = ab[1]
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


y_numerical=np.array(list(map(lambda ab: solve_singular_bvp(x_all,ab,eps=0.001), ab_test.numpy()))) #(100,10200)



ab_test = ab_test.cuda()
y = torch.vstack(list(map(lambda ab:net(torch.tile(ab.view(-1,2),(10200,1)),x_all_reshape),
                                 ab_test))) #(100,10200)
y=y.cpu().detach().numpy()


test_rel_l2=mean_rel_l2(y_numerical,y)
test_max=max_error(y_numerical,y)

test_rel_l2_inner = mean_rel_l2(y_numerical[:,100:10100],y[:,100:10100])
test_max_inner = max_error(y_numerical[:,100:10100],y[:,100:10100])

with open('./pi-deeponet/test error', 'a') as f:
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

for i in range(ntest):
    if i % 31== 0:
        ax.plot(x_all, y_numerical[i, :], 'b-',label='Numerical solution',linewidth=2, alpha=0.8, zorder=0)  # analytical
        ax.plot(x_all,y[i,:],'r--',label='Leading-order PVD-ONet',linewidth=2, alpha=1.0, zorder=1)
ax.legend(['Numerical solution', 'Leading-order PVD-ONet'], loc='best')

ax.set_xlim(0, 1)
ax.set_ylim(-2, 2)
ax.margins(0)
bbox = (0.005, 0.1, 0.6, 1)
axins = inset_axes(ax, width='40%', height='50%', loc="lower center", bbox_to_anchor=bbox,
                           bbox_transform=ax.transAxes)

for i in range(ntest):
    if i % 31 == 0:
        axins.plot(x_all, y_numerical[i,:], 'b-', label='Numerical solution',linewidth=2, alpha=0.8,zorder=0)  # analytical
        axins.plot(x_all, y[i,:], 'r--', label='Leading-order PVD-ONet',
                   linewidth=2, alpha=1.0, zorder=1)
axins.set_xlim(0.48, 0.52)
axins.set_ylim(-1.8, 1.8)

mark_inset(ax, axins, loc1=3, loc2=1, fc=(0.5, 0.5, 0.5, 0.3), ec=(0.5, 0.5, 0.5, 0.3), lw=1.0)
plt.savefig(path + 'pred.png', dpi=600)

plt.show()




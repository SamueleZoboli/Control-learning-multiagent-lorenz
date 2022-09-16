'''Train a NN to approximate the nonlinear metric P(x),
Author: Samuele Zoboli'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd.functional import jacobian
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse


def create_mlp(in_dim, out_dim, arch, activ, out_activ):
    '''Multilayer Perceptron network generation'''
    layers = [nn.Linear(in_dim, arch[0])]
    layers.append(activ)
    if len(arch) > 1:
        for i in range(len(arch) - 1):
            hidden = nn.Linear(arch[i], arch[i + 1])
            layers.append(hidden)
            layers.append(activ)
    out = nn.Linear(arch[-1], out_dim)
    layers.append(out)
    layers.append(out_activ)
    net = nn.Sequential(*layers)
    return net


class System():
    '''Lorenz attractor'''

    def __init__(self):
        self.sigma = 10
        self.rho = 28
        self.beta = 2.667
        self.n = 3
        self.m = 6
        self.learning_metric = True
        self.learning_policy = False

    def learn_metric(self):
        self.learning_metric = True
        self.learning_policy = False
        self.m = 6
        return

    def learn_policy(self):
        self.learning_metric = True
        self.learning_policy = False
        self.m = 1
        return

    def calc_dfdx(self, s):
        '''Compute Jacobian'''
        x, y, z = s[:, 0], s[:, 1], s[:, 2]

        dfdx = torch.zeros((s.shape[0], s.shape[1], s.shape[1]))

        dfdx[:, 0, 0] = -self.sigma
        dfdx[:, 0, 1] = self.sigma
        dfdx[:, 0, 2] = 0
        dfdx[:, 1, 0] = self.rho - z.squeeze(1)
        dfdx[:, 1, 1] = -1
        dfdx[:, 1, 2] = -x.squeeze(1)
        dfdx[:, 2, 0] = y.squeeze(1)
        dfdx[:, 2, 1] = x.squeeze(1)
        dfdx[:, 2, 2] = -self.beta

        return dfdx

    def calc_f(self, s):
        '''Compute continuous dynamics without input'''
        x, y, z = s[:, 0], s[:, 1], s[:, 2]
        x_dot = self.sigma * (y - x)
        y_dot = self.rho * x - y - x * z
        z_dot = x * y - self.beta * z

        f = torch.cat((x_dot, y_dot, z_dot), dim=1).unsqueeze(2)

        return f

    def calc_g(self, s):
        '''Compute continuous dynamics input contribution'''
        x, y, z = s[:, 0], s[:, 1], s[:, 2]
        x_dot = torch.ones(x.shape)
        y_dot = 2 + torch.sin(x)
        z_dot = torch.zeros(x.shape)

        g = torch.cat((x_dot, y_dot, z_dot), dim=1).unsqueeze(2)

        return g

    def calc_dgdx(self, s):
        '''Compute Jacobian'''
        x, y, z = s[:, 0], s[:, 1], s[:, 2]

        dgdx = torch.zeros((s.shape[0], s.shape[1], s.shape[1]))

        dgdx[:, 1, 0] = torch.cos(x).squeeze(1)

        return dgdx


def calc_P(actions, n, device):
    '''Generate P(x) from NN output
    '''
    P = torch.zeros((actions.shape[0], n, n)).to(device)

    P[:, 0, 0] = actions[:, 0]
    P[:, 0, 1] = actions[:, 1]
    P[:, 1, 0] = P[:, 0, 1]
    P[:, 0, 2] = actions[:, 2]
    P[:, 2, 0] = P[:, 0, 2]
    P[:, 1, 1] = actions[:, 3]
    P[:, 1, 2] = actions[:, 4]
    P[:, 2, 1] = P[:, 1, 2]
    P[:, 2, 2] = actions[:, 5]

    return P.to(device)


def calc_dPdx(net, s, f, g, high, m, n, device):
    '''Calculate the metric derivative passing through the NN jacobian'''
    state = s.squeeze(2)

    # action gradient
    dadx = torch.zeros((state.shape[0], m, n)).to(device)
    for i in range(state.shape[0]):
        dadx[i] = high * jacobian(net, state[i, :], create_graph=True)

    # return directly the multiplication by f and g
    dPdxg_vec = dadx @ g
    dPdxg = torch.zeros((s.shape[0], n, n)).to(device)

    dPdxg[:, 0, 0] = dPdxg_vec[:, 0, 0]
    dPdxg[:, 0, 1] = dPdxg_vec[:, 1, 0]
    dPdxg[:, 1, 0] = dPdxg[:, 0, 1]
    dPdxg[:, 0, 2] = dPdxg_vec[:, 2, 0]
    dPdxg[:, 2, 0] = dPdxg[:, 0, 2]
    dPdxg[:, 1, 1] = dPdxg_vec[:, 3, 0]
    dPdxg[:, 1, 2] = dPdxg_vec[:, 4, 0]
    dPdxg[:, 2, 1] = dPdxg[:, 1, 2]
    dPdxg[:, 2, 2] = dPdxg_vec[:, 5, 0]

    dPdxf_vec = dadx @ f
    dPdxf = torch.zeros((s.shape[0], n, n)).to(device)

    dPdxf[:, 0, 0] = dPdxf_vec[:, 0, 0]
    dPdxf[:, 0, 1] = dPdxf_vec[:, 1, 0]
    dPdxf[:, 1, 0] = dPdxf[:, 0, 1]
    dPdxf[:, 0, 2] = dPdxf_vec[:, 2, 0]
    dPdxf[:, 2, 0] = dPdxf[:, 0, 2]
    dPdxf[:, 1, 1] = dPdxf_vec[:, 3, 0]
    dPdxf[:, 1, 2] = dPdxf_vec[:, 4, 0]
    dPdxf[:, 2, 1] = dPdxf[:, 1, 2]
    dPdxf[:, 2, 2] = dPdxf_vec[:, 5, 0]

    return dPdxg.to(device), dPdxf.to(device)


def criterion(net, x, f, g, dfdx, dgdx, actions, n, m, q, e, s, lowp, high, device):
    '''Loss function'''
    mat_elem = actions
    P = calc_P(mat_elem, n, device=device)
    dfdxT = torch.transpose(dfdx, dim0=1, dim1=2).to(device)
    dgdxT = torch.transpose(dgdx, dim0=1, dim1=2).to(device)
    gT = torch.transpose(g, dim0=1, dim1=2).to(device)
    dPdxg, dPdxf = calc_dPdx(net, x, f, g, high, m, n, device=device)
    I_n = torch.eye(n).unsqueeze(0).repeat(x.shape[0], 1, 1).to(device)
    M1 = dfdxT @ P + P @ dfdx + dPdxf - s * (P @ g @ gT @ P) + q * I_n
    M2_up = dgdxT @ P + P @ dgdx + dPdxg - e * I_n
    M2_low = - (dgdxT @ P + P @ dgdx + dPdxg) - e * I_n
    M3 = - P + lowp * I_n

    # cost to satisfy conditions
    max_M1 = torch.real(torch.linalg.eigvalsh(M1)).max()
    max_M2_up = torch.real(torch.linalg.eigvalsh(M2_up)).max()
    max_M2_low = torch.real(torch.linalg.eigvalsh(M2_low)).max()
    max_M3 = torch.real(torch.linalg.eigvalsh(M3)).max()

    cost1 = torch.log(F.relu(max_M1) + 1)
    cost2_up = torch.log(F.relu(max_M2_up) + 1)
    cost2_low = torch.log(F.relu(max_M2_low) + 1)
    cost3 = torch.log(F.relu(max_M3) + 1)
    cost = cost1 + 10 * cost2_up + 10 * cost2_low + 20 * cost3

    return cost, {'c1': cost1, 'c2_up': cost2_up, 'c2_low': cost2_low, 'c3': cost3}


class Param_estimator(nn.Module):
    '''Estimate the cost parameters'''

    def __init__(self, min, max):
        super(Param_estimator, self).__init__()
        self.par = nn.Parameter(torch.tensor([0.5,0.5,30,0.001]))#torch.rand(4, ))
        self.min = min
        self.max = max

    def get_par(self):
        e = F.hardtanh(self.par[0], min_val=self.min[0], max_val=self.max[0])
        q = F.hardtanh(self.par[1], min_val=e.item(), max_val=self.max[1])
        s = F.hardtanh(self.par[2], min_val=self.min[2], max_val=self.max[2])
        lowp = F.hardtanh(self.par[3], min_val=self.min[3], max_val=self.max[3])
        return q, e, s, lowp


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='find_P',
                                     description='Look for a metric satisfying the contraction conditions',
                                     allow_abbrev=False)

    parser.add_argument('--net', type=str, help='hidden layers dimensions separated by comma', required=True)
    parser.add_argument('--activ', type=str, help='activ function for hidden layers (relu or tanh)', required=True)
    parser.add_argument('--dataset_size', type=float, help='total number of samples', required=True)
    parser.add_argument('--batch_size', type=int, help='batch size', required=True)
    parser.add_argument('--n_epochs', type=int, help='number of epochs', required=True)
    parser.add_argument('--learning_rate', type=float, help='initial learning rate, scheduling follows cosineannealing',
                        required=True)
    parser.add_argument('--log_name', type=str, help='name of log folder', required=True)

    args = parser.parse_args()

    arch = list(map(int, args.net.split(',')))
    a_fn = args.activ
    n_samples = int(args.dataset_size)
    batch_size = int(args.batch_size)
    n_epochs = int(args.n_epochs)
    lr = args.learning_rate
    log_name = args.log_name

    dyn_sys = System()
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # env parameters
    max_mat_elem_val = 100
    max_param_val = torch.tensor([10, 10, 100, 100], requires_grad=False).to(device)
    min_param_val = torch.tensor([0.001, 0.001, 0.001, 0.001], requires_grad=False).to(device)
    low = -max_mat_elem_val * torch.ones((dyn_sys.m,), requires_grad=False).to(device)
    high = max_mat_elem_val * torch.ones((dyn_sys.m,), requires_grad=False).to(device)

    # net parameters
    in_dim = dyn_sys.n
    out_dim = dyn_sys.m
    assert a_fn in ('tanh', 'relu'), 'Not supported activ'
    if a_fn == 'tanh':
        activ = {'name': 'tanh', 'layer': nn.Tanh()}
    else:
        activ = {'name': 'relu', 'layer': nn.ReLU()}

    out_activ = {'name': 'sat', 'layer': nn.Hardtanh()}

    # learning parameters
    train_size = int(0.8 * n_samples)
    test_size = n_samples - train_size
    mean = 0
    std = 10

    log_dir = 'runs/net_and_par/' + log_name
    comment = 'net= ' + ','.join(str(l) for l in arch) + ', act= ' + activ['name'] + ', out_activ= ' + out_activ[
        'name'] + ', lr= ' + str(lr)
    suff = log_name
    writer = SummaryWriter(log_dir=log_dir, comment=comment, filename_suffix=suff)

    # create models folder if needed
    model_dir = log_dir + '/models/'
    exists = os.path.exists(model_dir)
    if not exists:
        os.makedirs(model_dir)

    # model and training
    model = create_mlp(in_dim=in_dim, out_dim=out_dim, arch=arch, activ=activ['layer'],
                       out_activ=out_activ['layer']).to(device)
    estimator = Param_estimator(min=min_param_val, max=max_param_val).to(device)

    writer.add_graph(model, torch.randn(1, dyn_sys.n).to(device))
    net_optimizer = optim.Adam(model.parameters(), lr=lr)
    par_optimizer = optim.Adam(estimator.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(net_optimizer, n_epochs)

    # generate dataset
    x = torch.normal(mean=mean, std=std, size=(n_samples, dyn_sys.n, 1))

    with torch.no_grad():
        dfdx = dyn_sys.calc_dfdx(x)
        dgdx = dyn_sys.calc_dgdx(x)
        f = dyn_sys.calc_f(x)
        g = dyn_sys.calc_g(x)
    dataset = TensorDataset(x, f, g, dfdx, dgdx)

    train_data, test_data = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=len(test_data))

    for epoch in range(n_epochs):
        train_losses = []
        c1_buffer = []
        c2_up_buffer = []
        c2_low_buffer = []
        c3_buffer = []
        q_buffer = []
        e_buffer = []
        s_buffer = []
        lowp_buffer = []

        model.train()
        start_time = time.time()
        for x, f, g, dfdx, dgdx in train_loader:
            x = x.to(device)
            f = f.to(device)
            g = g.to(device)
            dfdx = dfdx.to(device)
            dgdx = dgdx.to(device)
            net_optimizer.zero_grad()
            par_optimizer.zero_grad()
            q, e, s, lowp = estimator.get_par()
            scaled_actions = model(x.squeeze(2))
            actions = low + (0.5 * (scaled_actions + 1.0) * (high - low))
            loss, info = criterion(net=model, x=x, f=f, g=g, dfdx=dfdx, dgdx=dgdx, actions=actions, n=in_dim, m=out_dim,
                                   q=q, e=e, s=s, lowp=lowp,
                                   high=max_mat_elem_val, device=device)
            loss.backward()
            net_optimizer.step()
            par_optimizer.step()
            train_losses.append(loss.item())
            c1_buffer.append(info['c1'].item())
            c2_up_buffer.append(info['c2_up'].item())
            c2_low_buffer.append(info['c2_low'].item())
            c3_buffer.append(info['c3'].item())
            q_buffer.append(q.item())
            e_buffer.append(e.item())
            s_buffer.append(s.item())
            lowp_buffer.append(lowp.item())

        # save model for checkpoint
        torch.save(model, model_dir +  'epoch_' + str(epoch) + '.pt')

        # logs
        writer.add_scalar("lr/train", scheduler.get_last_lr()[0], epoch)
        writer.add_scalar("q/train", np.mean(q_buffer), epoch)
        writer.add_scalar("e/train", np.mean(e_buffer), epoch)
        writer.add_scalar("s/train", np.mean(s_buffer), epoch)
        writer.add_scalar("lowp/train", np.mean(lowp_buffer), epoch)

        writer.add_scalar("Loss/train", np.mean(train_losses), epoch)
        writer.add_scalar("Cost_Lf/train", np.mean(c1_buffer), epoch)
        writer.add_scalar("Cost_Lg_up/train", np.mean(c2_up_buffer), epoch)
        writer.add_scalar("Cost_Lg_low/train", np.mean(c2_low_buffer), epoch)
        writer.add_scalar("Cost_P/train", np.mean(c3_buffer), epoch)

        model.eval()

        with torch.no_grad():
            for i, (x, f, g, dfdx, dgdx) in enumerate(test_loader):
                assert (i == 0)  # ensure I pick all the test at once dataset
                x = x.to(device)
                f = f.to(device)
                g = g.to(device)
                dfdx = dfdx.to(device)
                dgdx = dgdx.to(device)
                q, e, s, lowp = estimator.get_par()
                scaled_actions = model(x.squeeze(2))
                actions = low + (0.5 * (scaled_actions + 1.0) * (high - low))
                test_loss, test_info = criterion(net=model, x=x, f=f, g=g, dfdx=dfdx, dgdx=dgdx, actions=actions,
                                                 n=in_dim, m=out_dim,
                                                 q=q, e=e, s=s, lowp=lowp,
                                                 high=max_mat_elem_val, device=device)
                writer.add_scalar("Loss/test", test_loss.item(), epoch)
                writer.add_scalar("Cost_Lf/test", test_info['c1'].item(), epoch)
                writer.add_scalar("Cost_Lg_up/test", test_info['c2_up'].item(), epoch)
                writer.add_scalar("Cost_Lg_low/test", test_info['c2_low'].item(), epoch)
                writer.add_scalar("Cost_P/test", test_info['c3'].item(), epoch)
                writer.add_scalar("q/test", q.item(), epoch)
                writer.add_scalar("e/test", e.item(), epoch)
                writer.add_scalar("s/test", s.item(), epoch)
                writer.add_scalar("lowp/test", lowp.item(), epoch)

        # save best model
        if epoch == 0:
            best_score = test_loss.item()
            torch.save(model, model_dir + '/best.pt')
            with open(model_dir + '/parameters.txt', 'w') as file:
                file.write(
                    'q= ' + str(q.item()) + 'e= ' + str(e.item()) + 's= ' + str(s.item()) + 'lowp= ' + str(lowp.item()))
        else:
            if test_loss <= best_score:
                best_score = test_loss.item()
                torch.save(model, model_dir + '/best.pt')
                with open(model_dir + '/parameters.txt', 'w') as file:
                    file.write('q= ' + str(q.item()) + 'e= ' + str(e.item()) + 's= ' + str(s.item()) + 'lowp= ' + str(
                        lowp.item()))

        print('epoch %i train loss %.2e test loss %.2e lr %.2e' % (
        epoch, np.mean(train_losses).item(), test_loss, scheduler.get_last_lr()[0]))
        # next learning rate
        scheduler.step()

    writer.flush()
    writer.close()

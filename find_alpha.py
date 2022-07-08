'''Learn a primitive of g^T P for the control action, with P coming from a trained DNN
Author: Samuele Zoboli'''

import torch
import torch.nn as nn
import numpy as np
from torch.autograd.functional import jacobian
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse
from find_P import System, calc_P, create_mlp


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='find_alpha',
                                     description='Look for a function satisfying the integrability condition',
                                     allow_abbrev=False)

    parser.add_argument('--net', type=str, help='hidden layers dimensions separated by comma', required=True)
    parser.add_argument('--activ', type=str, help='activ function for hidden layers (relu or tanh)', required=True)
    parser.add_argument('--dataset_size', type=float, help='total number of samples', required=True)
    parser.add_argument('--batch_size', type=int, help='batch size', required=True)
    parser.add_argument('--n_epochs', type=int, help='number of epochs', required=True)
    parser.add_argument('--learning_rate', type=float, help='initial learning rate, scheduling follows cosineannealing',
                        required=True)
    parser.add_argument('--log_name', type=str, help='name of log folder', required=True)
    parser.add_argument('--metric_path', type=str, help='path to the metric network', required=True)

    args = parser.parse_args()

    arch = list(map(int, args.net.split(',')))
    a_fn = args.activ
    n_samples = int(args.dataset_size)
    batch_size = int(args.batch_size)
    n_epochs = int(args.n_epochs)
    lr = args.learning_rate
    log_name = args.log_name
    metric_path = args.metric_path

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # env parameters
    max_mat_elem_val = 100
    dyn_sys = System()
    low = -max_mat_elem_val * torch.ones((dyn_sys.m,), requires_grad=False).to(device)
    high = max_mat_elem_val * torch.ones((dyn_sys.m,), requires_grad=False).to(device)
    dyn_sys.learn_policy()

    # net parameters
    in_dim = dyn_sys.n
    out_dim = dyn_sys.m
    assert a_fn in ('tanh', 'relu'), 'Not supported activ'
    if a_fn == 'tanh':
        activ = {'name': 'tanh', 'layer': nn.Tanh()}
    else:
        activ = {'name': 'relu', 'layer': nn.ReLU()}

    out_activ = {'name': 'id', 'layer': nn.Identity()}

    # learning parameters
    train_size = int(0.8 * n_samples)
    test_size = n_samples - train_size
    mean = 0
    std = 10

    log_dir = 'runs/alpha/' + log_name
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

    metric = torch.load(metric_path)

    writer.add_graph(model, torch.randn(1, dyn_sys.n).to(device))
    net_optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(net_optimizer, n_epochs)

    # generate dataset
    x = torch.normal(mean=mean, std=std, size=(n_samples, dyn_sys.n,1))
    with torch.no_grad():
        scaled_elem = metric(x.squeeze(2))
        elem = low + (0.5 * (scaled_elem + 1.0) * (high - low))
        P = calc_P(elem, dyn_sys.n, device)
        g = dyn_sys.calc_g(x)
        gT = torch.transpose(g, dim0=1, dim1=2)
        gTP = gT@P
    dataset = TensorDataset(x,gTP)

    train_data, test_data = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=len(test_data))
    criterion = nn.MSELoss()

    for epoch in range(n_epochs):
        train_losses = []

        model.train()
        start_time = time.time()
        for x, dadx in train_loader:
            x = x.to(device)
            dadx = dadx.to(device)
            net_optimizer.zero_grad()
            jac = torch.zeros((x.shape[0], dyn_sys.m, dyn_sys.n)).to(device)
            for i in range(x.shape[0]):
                jac[i] =  jacobian(model, x[i, :,0], create_graph=True)
            loss = criterion(jac, dadx)
            loss.backward()
            net_optimizer.step()
            train_losses.append(loss.item())

        # save model for checkpoint
        torch.save(model, model_dir +  'epoch_' + str(epoch) + '.pt')

        # logs
        writer.add_scalar("lr/train", scheduler.get_last_lr()[0], epoch)
        writer.add_scalar("Loss/train", np.mean(train_losses), epoch)

        model.eval()

        with torch.no_grad():
            for i, (x, dadx) in enumerate(test_loader):
                assert (i == 0)  # ensure I pick all the test dataset at once
                x = x.to(device)
                dadx = dadx.to(device)
                jac = torch.zeros((x.shape[0], dyn_sys.m, dyn_sys.n)).to(device)
                for i in range(x.shape[0]):
                    jac[i] = jacobian(model, x[i, :, 0])
                test_loss = criterion(jac, dadx)
                writer.add_scalar("Loss/test", test_loss.item(), epoch)

        # save best model
        if epoch == 0:
            best_score = test_loss.item()
            torch.save(model, model_dir + '/best.pt')
        else:
            if test_loss <= best_score:
                best_score = test_loss.item()
                torch.save(model, model_dir + '/best.pt')

        print('epoch %i train loss %.2e test loss %.2e lr %.2e' % (
        epoch, np.mean(train_losses).item(), test_loss, scheduler.get_last_lr()[0]))
        # next learning rate
        scheduler.step()

    writer.flush()
    writer.close()

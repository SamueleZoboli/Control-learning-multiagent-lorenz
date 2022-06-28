'''Test the NN-based controller on a system of Lorenz attractors,
Author: Samuele Zoboli'''

import matplotlib.pyplot as plt
import torch
import numpy as np
import argparse


class MultiSys():
    '''Multiagent system of lorenz attractors'''
    def __init__(self, n_agents: int):
        self.N = n_agents
        self.n = 3
        self.m = 5
        self.sigma = 10
        self.rho = 28
        self.beta = 2.667
        self.dt = .01
        self.g = np.array([[0.], [1.], [0.]])
        self.state = self.reset()
        self.k = 4
        self.L = np.array([[ 0, 0, 0, 0, 0, 0],
                           [-1, 3, -1, 0, -1, 0],
                           [0, -1, 3, -1, -1, 0],
                           [0, 0, -1, 2, 0 , -1],
                           [0, -1, -1, 0, 2, 0],
                           [0, 0, 0, -1, 0, 1],])

    def reset(self):
        '''Set a random initial condition'''
        mu = 0
        sigma = 20
        x = mu + sigma * np.random.randn(self.N, self.n)
        return x.astype(np.float32)

    def get_obs(self):
        return self.state

    def step(self, u):
        '''Compute the next state from the dynamics'''

        assert (u.shape == (self.N, 1)), 'Input dimension is not correct'

        x, y, z = self.state[:, 0], self.state[:, 1], self.state[:, 2]
        x_dot_f = self.sigma * (y - x)
        y_dot_f = self.rho * x - y - x * z
        z_dot_f = x * y - self.beta * z

        f = np.concatenate(
            (np.expand_dims(x_dot_f, axis=1), np.expand_dims(y_dot_f, axis=1), np.expand_dims(z_dot_f, axis=1)), axis=1)

        x_dot_g = np.ones(x.shape)
        y_dot_g = 2 + np.sin(x)
        z_dot_g = np.zeros(x.shape)

        g = np.concatenate(
            (np.expand_dims(x_dot_g, axis=1), np.expand_dims(y_dot_g, axis=1), np.expand_dims(z_dot_g, axis=1)), axis=1)

        dyn = f + (np.expand_dims(g, axis = 2) @ np.expand_dims(u, axis = 1)).squeeze(2)

        self.state = self.state + dyn * self.dt

        return self.get_obs()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='test_multiagent',
                                     description='Test the law on a network of 6 agents',
                                     allow_abbrev=False)

    parser.add_argument('--alpha_path', type=str, help='path to the alpha network', required=True)

    args = parser.parse_args()

    model_path = args.alpha_path

    n_agents = 6
    n_steps = 800

    model = torch.load(model_path)
    env = MultiSys(n_agents = n_agents)

    traj = []
    u_tr = []
    noise_tr = []

    x = env.reset()

    with torch.no_grad():
        for i in range(n_steps):
            # add noise
            noise =  0.5 * np.random.randn(x.shape[0], x.shape[1])
            x_noise = x + noise
            alpha = model(torch.from_numpy(x_noise.astype(np.float32))).numpy()
            u = -env.k * env.L @ alpha
            x_plus = env.step(u)
            x = x_plus
            traj.append(x)
            u_tr.append(u)
            noise_tr.append(noise)

    traj = np.array(traj)
    u_tr = np.array(u_tr)
    noise_tr = np.array(noise_tr)

    t =np.arange(n_steps)*env.dt
    fig, (ax1, ax2, ax3) = plt.subplots(nrows = 3, ncols = 1, sharex = True, figsize = (8,8))
    color = ['coral', 'darkviolet', 'limegreen', 'firebrick',  'b', 'c' ]
    linewidth = 1.2
    for i in range(n_agents):
        ax1.plot(t, traj[:,i,0],label = r'$a_'+str(i+1)+'$', color = color[i], linewidth = linewidth)
    for i in range(n_agents):
        ax2.plot(t, traj[:,i,1],label = r'$a_'+str(i+1)+'$', color = color[i], linewidth = linewidth)
    for i in range(n_agents):
        ax3.plot(t, traj[:,i,2],label = r'$a_'+str(i+1)+'$', color = color[i], linewidth = linewidth)

    fontsize = 20

    ax1.set_ylabel(r'$x_1$', fontsize = fontsize)
    ax2.set_ylabel(r'$x_2$', fontsize = fontsize)
    ax3.set_ylabel(r'$x_3$', fontsize = fontsize)
    ax3.set_xlabel(r'$t \; [s]$', fontsize = fontsize)

    ax1.grid(visible=True,which = 'major', alpha= 0.5, ls='--')
    ax2.grid(visible=True,which = 'major',  alpha= 0.5, ls='--')
    ax3.grid(visible=True,which = 'major', alpha= 0.5, ls='--')

    # error plot
    plt.figure(figsize = (8,6))
    error_full = np.zeros((traj.shape[0],traj.shape[1]))
    for i in range(traj.shape[1]):
        error_full[:,i] = np.linalg.norm(traj[:,0,:]-traj[:,i,:], axis=1)
    error_mean, error_std = np.mean(error_full[:,1:], axis = 1), np.std(error_full[:,1:], axis = 1)
    plt.fill_between(t, error_mean-error_std, error_mean+ error_std,  color = 'lightblue', label = 'std(error norm)')
    plt.plot(t, error_mean, label='mean(error norm)')
    plt.grid(visible=True,which = 'major', alpha= 0.5, ls='--')
    plt.xlabel(r'$t \; [s]$', fontsize=fontsize)
    plt.legend(loc = 'upper right', ncol = 1, fontsize = int(fontsize/1.2))
    plt.show()
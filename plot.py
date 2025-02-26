import os
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from models.lyapunov import QuadraticLyapunov
from utils import logger


plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 20})


def plot_V_heatmap(
    fig, V, rho, lower_limit, upper_limit, nx, x_boundary=None,
    V_baseline=None, plot_idx=[0, 1], mode=0.0, V_color="k", V_lqr=None,
    device=None,
):
    x_ticks = torch.linspace(
        lower_limit[plot_idx[0]], upper_limit[plot_idx[0]], 500, device=device
    )
    y_ticks = torch.linspace(
        lower_limit[plot_idx[1]], upper_limit[plot_idx[1]], 500, device=device
    )
    grid_x, grid_y = torch.meshgrid(x_ticks, y_ticks)
    X = torch.ones(250000, nx, device=device) * upper_limit * mode
    X[:, plot_idx[0]] = grid_x.flatten()
    X[:, plot_idx[1]] = grid_y.flatten()

    with torch.no_grad():
        V_val = V(X)

    V_val = V_val.cpu().reshape(500, 500)
    grid_x = grid_x.cpu()
    grid_y = grid_y.cpu()

    ax = fig.add_subplot(111)
    im = ax.pcolor(grid_x, grid_y, V_val, cmap=cm.coolwarm)
    ct = ax.contour(grid_x, grid_y, V_val, [rho], colors=V_color, linewidths=2.5, label='Ours')

    if V_baseline is not None:
        with torch.no_grad():
            V_val_baseline = V_baseline['V'](X).cpu().reshape(500, 500)
        ct = ax.contour(grid_x, grid_y, V_val_baseline, [V_baseline['rho']],
                   colors='orange', linewidths=2.5, label='Yang et al. (2024)')

    lower_limit = lower_limit.cpu()
    upper_limit = upper_limit.cpu()
    ax.set_xlim(lower_limit[plot_idx[0]], upper_limit[plot_idx[0]])
    ax.set_ylim(lower_limit[plot_idx[1]], upper_limit[plot_idx[1]])
    ax.set_xlabel(plot_idx[2])
    ax.set_ylabel(plot_idx[3])
    cbar = fig.colorbar(im, ax=ax)
    return ax, cbar


def plot_V(model, lower_limit, upper_limit, rho, V_baseline=None,
           save_path=None, args=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    lower_limit = lower_limit.to(args.device)
    upper_limit = upper_limit.to(args.device)
    nx = lower_limit.shape[-1]
    if V_baseline is not None:
        V_baseline['V'] = V_baseline['V'].to(args.device)
    if args.dynamics == 'quadrotor2d':
        labels = ['x (m)', 'y (m)', '$\\theta$ (rad)',
                  '$\dot{x}$ (m)', '$\dot{y}$ (m)', '$\dot{\\theta}$ (rad)',
                  ]
        plot_idxs = [
            (i, j, labels[i], labels[j])
            for (i, j) in [(0, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 3)]
            # for i in range(lower_limit.shape[-1])
            # for j in range(i + 1, lower_limit.shape[-1])
        ]
    else:
        plot_idxs = [
            (i, j, '', '')
            for i in range(lower_limit.shape[-1])
            for j in range(i + 1, lower_limit.shape[-1])
        ]
    for plot_idx in plot_idxs:
        logger.info('Plotting: %s', plot_idx)
        fig = plt.figure()
        plot_V_heatmap(
            fig, model.lyapunov, rho, lower_limit, upper_limit, nx,
            V_baseline=V_baseline, plot_idx=plot_idx, device=args.device)
        fig.tight_layout()
        fig.savefig(os.path.join(save_path, f'{plot_idx[0]}_{plot_idx[1]}.png'), dpi=1000)


def load_V_baseline(args):
    if args.dynamics == 'quadrotor2d':
        checkpoint = torch.load('baselines/models/quadrotor2d_state_feedback.pth')
        lyapunov = QuadraticLyapunov(
            goal_state=torch.zeros(6), x_dim=6, R_rows=6, R_scale=1.0, eps=0.01,
            R=checkpoint['state_dict']['lyapunov.R']
        )
        return {
            'V': lyapunov,
            'rho': 1.3392
        }
    else:
        return None

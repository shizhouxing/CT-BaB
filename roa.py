import torch
import torch.nn as nn
import argparse
import yaml
import os
from tqdm import tqdm


def eval_roa(V, lower_limit, upper_limit, hole_lower, hole_upper, rho,
             sample_batch_size=int(1e7), sample_batches=1000, num_ticks=20,
             scale_input=1.0, scale_decay=0.95, scale_iters=100):
    lower_limit = lower_limit.cuda()
    upper_limit = upper_limit.cuda()
    hole_lower = hole_lower.cuda()
    hole_upper = hole_upper.cuda()

    roa = torch.tensor(0., device='cuda')
    roa_scaled = torch.tensor(0., device='cuda')
    scale = 1.0
    for t in range(scale_iters):
        cnt_inside = cnt_all = 0
        ticks = [
            torch.linspace(lower_limit[i], upper_limit[i], num_ticks, device='cuda')
            for i in range(lower_limit.shape[-1])
        ]
        grid = torch.meshgrid(*ticks)
        grid = [item.reshape(-1, 1) for item in grid]
        grid = torch.concat(grid, dim=-1)
        y = V(grid * scale)
        inside = (y <= rho).float().mean()
        roa_scaled_ = inside * (scale / scale_input) ** (lower_limit.shape[-1])
        roa_ = roa_scaled_ * torch.prod((upper_limit-lower_limit))
        print('scale', scale, 'inside', inside, 'roa', roa_scaled_, roa_)
        roa = max(roa, roa_)
        roa_scaled = max(roa_scaled, roa_scaled_)
        scale *= scale_decay

        if inside == 1.0:
            break

    print('ROA:', 'scaled', roa_scaled, 'real', roa)

    return roa


def eval_roa_yang(model):
    from neural_lyapunov_training.lyapunov import NeuralNetworkQuadraticLyapunov, NeuralNetworkLyapunov

    def load_lyapunov_state_dict(lyapunov, path):
        model = torch.load(path)
        if 'state_dict' in model:
            model = model['state_dict']
        if isinstance(lyapunov, NeuralNetworkQuadraticLyapunov):
            lyapunov.R.data = model['lyapunov.R']
        else:
            state_dict = {
                k[len('lyapunov.'):]: v
                for k, v in model.items() if k.startswith('lyapunov.')
            }
            lyapunov.load_state_dict(state_dict)

    num_ticks = 20
    if model.endswith('quadrotor2d_state_feedback.pth'):
        lyapunov = NeuralNetworkQuadraticLyapunov(
            x_dim=6, goal_state=torch.zeros(6).cuda(),
            R_rows=6, eps=0.01)
        rho = 1.3392
        upper_limit = torch.tensor([0.75, 0.75, 1.57, 4, 4, 3])
    elif model.endswith('quadrotor2d_output_feedback.pth'):
        lyapunov = NeuralNetworkQuadraticLyapunov(
            x_dim=8, goal_state=torch.zeros(8).cuda(),
            R_rows=8, eps=0.01)
        rho = 0.045
        upper_limit = torch.tensor([
            0.1, 0.6283185307179586, 0.2, 0.6283185307179586,
            0.05, 0.3141592653589793, 0.1, 0.3141592653589793])
        num_ticks = 10
    elif (model.endswith('pendulum_state_feedback.pth')
          or model.endswith('pendulum_state_feedback_small_torque.pth')):
        # kappa=0.001
        lyapunov = NeuralNetworkLyapunov(
            x_dim=2, goal_state=torch.zeros(2).cuda(),
            hidden_widths=[16,16,8], R_rows=3, absolute_output=True, eps=0.01,
            activation=torch.nn.LeakyReLU, V_psd_form="L1"
        )
        if model.endswith('pendulum_state_feedback_small_torque.pth'):
            rho = 1.15516
        elif model.endswith('pendulum_state_feedback.pth'):
            rho = 672
        else:
            raise NotImplementedError
        upper_limit = torch.tensor([12., 12.])
    elif (model.endswith('path_tracking_state_feedback.pth')
          or model.endswith('path_tracking_state_feedback_small_torque.pth')):
        # kappa=0.001
        lyapunov = NeuralNetworkLyapunov(
            x_dim=2, goal_state=torch.zeros(2).cuda(),
            hidden_widths=[16,16,8], R_rows=3, absolute_output=True, eps=0.01,
            activation=torch.nn.LeakyReLU, V_psd_form="L1"
        )
        if model.endswith('path_tracking_state_feedback_small_torque.pth'):
            rho = 22.8664703369
        elif model.endswith('path_tracking_state_feedback.pth'):
            rho = 36.48387
        else:
            raise NotImplementedError
        upper_limit = torch.tensor([3., 3.])
    elif 'nlc_discrete' in model and model.endswith('path_tracking_lyapunov.pth'):
        lyapunov = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        rho = 60.99#72.425
        upper_limit = torch.tensor([3., 3.])
    elif 'nlc_discrete' in model and model.endswith('pendulum_lyapunov.pth'):
        lyapunov = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        rho = 22.74517822265625 #25.58
        upper_limit = torch.tensor([12., 12.])
    else:
        raise NotImplementedError

    lower_limit = -upper_limit
    hole_lower = lower_limit * 0.001
    hole_upper = upper_limit * 0.001
    load_lyapunov_state_dict(lyapunov, model)
    lyapunov = lyapunov.cuda()

    eval_roa(lyapunov, lower_limit, upper_limit, hole_lower, hole_upper, rho,
             num_ticks=num_ticks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    args = parser.parse_args()

    eval_roa_yang(args.model)

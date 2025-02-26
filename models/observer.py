import torch
import torch.nn as nn
from models.nn import FFNet


class NeuralNetworkLuenbergerObserver(nn.Module):
    """
    Neural network observer that takes vectors as observations.
    The observer dynamics is z[t+1] = f(z[t], u[t]) + nn(z[t], y[t]-h(z[t]))
    """

    def __init__(self, z_dim, y_dim, dynamics, h, zero_obs_error, depth, width,
                 activation='leaky_relu', scale_input=1.0):
        """
        z_dim: state estimate dimension, same as state dimension.
        y_dim: output dimension.
        h: observation function.
        zero_obs_error: y[t] - h(x[t]). when y[t] = h(z[t]), nn(z[t], y[t]-h(z[t])) = 0.
        """
        super().__init__()
        self.z_dim = z_dim
        self.dynamics = dynamics
        self.h = h
        self.zero_obs_error = zero_obs_error
        self.scale_input = scale_input
        self.net = FFNet(z_dim + y_dim, z_dim, depth, width, activation=activation)

    def forward(self, z, u, y):
        self.zero_obs_error = self.zero_obs_error.to(z.device)
        zero_obs_error = torch.ones((z.shape[0], 1), device=z.device) * self.zero_obs_error
        z_scale_back = z / self.scale_input
        z_nominal, _ = self.dynamics(z_scale_back, u)
        z_nominal = z_nominal * self.scale_input
        obs_error = y - self.h(z_scale_back) * self.scale_input
        Le = self.net(torch.cat((z, obs_error), 1))
        L0 = self.net(torch.cat((z, zero_obs_error), 1))
        unclipped_z = z_nominal + Le - L0
        return unclipped_z

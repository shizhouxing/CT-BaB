import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.nn import FFNet, ResNet


class NeuralNetworkController(nn.Module):
    def __init__(
        self,
        nlayer=3,
        in_dim=2,
        out_dim=1,
        hidden_dim=64,
        u_lo=None,
        u_up=None,
        x_equilibrium=None,
        u_equilibrium=None,
        activation='relu',
        arch='ff',
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if u_lo is not None:
            assert u_lo.shape == (out_dim,)
        self.u_lo = u_lo
        if u_up is not None:
            assert u_up.shape == (out_dim,)
        self.u_up = u_up
        if x_equilibrium is not None:
            assert x_equilibrium.shape == (in_dim,)
        self.x_equilibrium = x_equilibrium
        if u_equilibrium is not None:
            assert u_equilibrium.shape == (out_dim,)
            if self.u_lo is not None:
                assert (u_equilibrium >= self.u_lo).all()
            if self.u_up is not None:
                assert (u_equilibrium <= self.u_up).all()
        self.u_equilibrium = u_equilibrium
        if arch == 'ff':
            self.net = FFNet(in_dim, out_dim, nlayer, hidden_dim, activation=activation)
        elif arch == 'resnet':
            self.net = ResNet(in_dim, out_dim, nlayer, hidden_dim, activation=activation)
        else:
            raise NotImplementedError

    def _unclipped_output(self, x: torch.Tensor) -> torch.Tensor:
        if self.x_equilibrium is not None and self.u_equilibrium is not None:
            unclipped_output = (
                self.net(x) - self.net(self.x_equilibrium) + self.u_equilibrium
            )
        else:
            unclipped_output = self.net(x)
        return unclipped_output

    def forward(self, x):
        unclipped_output = self._unclipped_output(x)
        # Instead of calling clamp direct, we use relu twice.
        # Currently auto_LIRPA doesn't handle clamp.
        f1 = F.relu(unclipped_output - self.u_lo) + self.u_lo
        f = -(F.relu(self.u_up - f1) - self.u_up)
        return f

    def _apply(self, fn):
        """Handles CPU/GPU transfer and type conversion."""
        super()._apply(fn)
        self.x_equilibrium = fn(self.x_equilibrium)
        self.u_equilibrium = fn(self.u_equilibrium)
        if self.u_lo is not None:
            self.u_lo = fn(self.u_lo)
        if self.u_up is not None:
            self.u_up = fn(self.u_up)
        return self

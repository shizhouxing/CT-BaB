import math
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.nn import FFNet


class QuadraticLyapunov(nn.Module):
    """
    A quadratic Lyapunov function.
    This neural network output is V(x) = (x-x*)^T(εI+RᵀR)(x-x*),
    R is the parameters to be optimized.
    """

    def __init__(self, goal_state: torch.Tensor, x_dim: int, R_rows: int,
                 R_scale: float, eps: float,
                 R: typing.Optional[torch.Tensor] = None,
                 *args, **kwargs):
        """
        Args:
          x_dim: The dimension of state
          R_rows: The number of rows in matrix R.
          V(x) = (x-x*)^T(εI+RᵀR)(x-x*)
        """
        super().__init__(*args, **kwargs)
        self.goal_state = goal_state
        self.x_dim = x_dim
        self.R_rows = R_rows
        self.R_scale = R_scale
        self.eps = eps
        assert self.goal_state.shape == (self.x_dim,)
        assert isinstance(eps, float) and eps >= 0
        assert eps >= 0
        # Rt is the transpose of R

        if R is None:
            R = (torch.rand((R_rows, self.x_dim)) - 0.5) * R_scale
        # assert R_rows % self.x_dim == 0
        # R = torch.eye(self.x_dim).repeat(R_rows // self.x_dim, 1) * R_scale

        self.register_parameter(name="R", param=torch.nn.Parameter(R))

    def forward(self, x):
        x0 = x - self.goal_state
        Q = (
            self.eps * torch.eye(self.x_dim, device=x.device)
            + self.R.transpose(0, 1) @ self.R
        )
        lyap = torch.sum(x0 * (x0 @ Q), axis=1, keepdim=True)
        return lyap

    def diff(self, x, x_next, kappa=0, lyapunov_x=None):
        # V(x) = (x_t - x_*)^T Q (x_t - x_*)
        # V(x_next) = (x_next - x_*)^T Q (x_next - x_*)
        # dV = (x_next - x_*)^T Q (x_next - x_*) - (1-kappa) (x_t - x_*)^T Q (x_t - x_*)
        #    = x_next^T Q x_next
        #        - (1-kappa) x_t^T Q x_t
        #        - 2 (x_next - (1-kappa) x_t)^T Q x_*
        #        + kappa * x_*^T Q x_*
        #    = (x_next - sqrt(1-kappa) x_t)^T Q (x_next + sqrt(1-kappa) x_t)
        #        - 2 (x_next - (1-kappa)x_t)^T Q x_*
        #        + kappa * x_*^T Q x_*
        if kappa == 0:
            x_d1 = x_d2 = x_next - x
            x_s = x_next + x
        else:
            sqrt_1_minus_kappa = math.sqrt(1 - kappa)
            x_d1 = x_next - sqrt_1_minus_kappa * x
            x_d2 = x_next - (1 - kappa) * x
            x_s = x_next + sqrt_1_minus_kappa * x
        Q = (
            self.eps * torch.eye(self.x_dim, device=x.device)
            + (self.R.transpose(0, 1) @ self.R)
        )
        dV = (
            torch.sum(x_d1 * (x_s @ Q), axis=-1, keepdim=True)
            - 2 * torch.sum(x_d2 * (self.goal_state @ Q), axis=-1, keepdim=True)
        )
        if kappa != 0:
            dV = dV + kappa * torch.sum(
                self.goal_state * (self.goal_state @ Q), axis=-1, keepdim=True)
        return dV

    def _apply(self, fn):
        """Handles CPU/GPU transfer and type conversion."""
        super()._apply(fn)
        self.goal_state = fn(self.goal_state)
        return self


class NeuralNetworkLyapunov(QuadraticLyapunov):

    def __init__(self, goal_state: torch.Tensor, x_dim: int,
                 R_rows: int, R_scale: float, eps: float,
                 width: int, depth: int, V_psd_form: str = "L1",
                 scale=1.0, activation='relu', nn_scale=1.0,
                 *args, **kwargs
    ):
        super().__init__(goal_state, x_dim, R_rows, R_scale, eps)

        self.V_psd_form = V_psd_form
        self.scale = scale
        self.nn_scale = nn_scale

        if self.V_psd_form == 'new':
            self.net = FFNet(
                self.x_dim, self.R_rows, depth, width,
                activation=activation, last_bias=False)
        else:
            self.net = FFNet(self.x_dim, 1, depth, width, activation=activation)

    def forward(self, x):
        if self.V_psd_form == 'new':
            x = x - self.goal_state
            y = self.net(x)

            V = (y * y).sum(dim=-1, keepdim=True)
            V = V + 1e-8 * x.abs().sum(dim=-1, keepdim=True)
        else:
            V_psd = self._psd(x)
            V_net1 = self._net1(x)
            V = V_psd + V_net1 * self.nn_scale
            V = V * self.scale
        return V

    def _psd(self, x):
        if self.V_psd_form == 'quadratic':
            return super().forward(x)
        elif self.V_psd_form == 'l1_simple':
            return self._psd_l1_simple(x)
        else:
            return self._psd_l1(x, self.R)

    def _net1(self, x):
        """The first network with bias."""
        V_net1 = self.net(x) - self.net(self.goal_state)
        V_net1 = F.relu(V_net1) + F.relu(-V_net1)
        return V_net1

    def _psd_l1_simple(self, x):
        """|(εI+RᵀR)(x-x*)|₁"""
        output = self.eps * (F.relu(x) + F.relu(-x)).sum(dim=-1, keepdim=True)
        return output

    def _psd_l1(self, x, R):
        """|(εI+RᵀR)(x-x*)|₁"""
        Q = self.eps * torch.eye(self.x_dim, device=x.device) + (R.t() @ R)
        x = x - self.goal_state
        Rx = x @ Q
        output = (F.relu(Rx) + F.relu(-Rx)).sum(dim=-1, keepdim=True)
        return output

    def V_and_dV(self, x, x_next, dx, kappa=0):
        if self.V_psd_form == "quadratic":
            V_psd = super().forward(x)
            dV_psd = super().diff(x, x_next, kappa)

            V_net = self._net1(x) * self.nn_scale
            V_net_next = self._net1(x_next) * self.nn_scale
            dV = V_net_next - V_net * (1 - kappa)

            V = V_net + V_psd
            dV = dV + dV_psd

        elif self.V_psd_form == 'new':
            x = x - self.goal_state
            x_next = x_next - self.goal_state

            y = self.net(x)
            y_next = self.net(x_next)
            V = (y * y).sum(dim=-1, keepdim=True)

            if self.net.depth == 1:
                dV = (self.net(dx) * (self.net(x + x_next))).sum(dim=-1,keepdim=True)
            else:
                dV = ((y_next - y) * (y_next + y)).sum(dim=-1, keepdim=True)

            x_abs_sum = x.abs().sum(dim=-1, keepdim=True)
            x_next_abs_sum = x_next.abs().sum(dim=-1, keepdim=True)

            V = V + 1e-8 * x_abs_sum
            dV = dV + 1e-8 * (x_next_abs_sum - x_abs_sum)
            dV = dV + V * kappa

        else:
            V = self.forward(x)
            V_next = self.forward(x_next)
            dV = V_next - V * (1 - kappa)

        return V, dV

    def _apply(self, fn):
        """Handles CPU/GPU transfer and type conversion."""
        super()._apply(fn)
        self.goal_state = fn(self.goal_state)
        return self

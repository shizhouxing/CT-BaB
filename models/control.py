import torch
import torch.nn as nn
from models.lyapunov import NeuralNetworkLyapunov, QuadraticLyapunov
from models.controllers import NeuralNetworkController
from models.dynamic_system import FirstOrderDiscreteTimeSystem, SecondOrderDiscreteTimeSystem, IntegrationMethod
from models.pendulum import PendulumDynamics
from models.path_tracking import PathTrackingDynamics
from models.quadrotor2d import Quadrotor2DDynamics, Quadrotor2DLidarDynamics
from models.quadrotor3d import Quadrotor3DDynamics
from models.observer import NeuralNetworkLuenbergerObserver
from huggingface_hub import PyTorchModelHubMixin


class ModelControl(nn.Module, PyTorchModelHubMixin):
    def __init__(self, valid_lower=None, valid_upper=None,
                 controller_depth=4, controller_width=8,
                 controller_arch='ff', controller_act='relu',
                 lyapunov_func="quadratic", lyapunov_R_rows=3,
                 lyapunov_R_scale=1.0, lyapunov_nn_scale=1.0,
                 lyapunov_width=16, lyapunov_depth=3,
                 lyapunov_scale=1.0, lyapunov_act='relu',
                 lyapunov_eps=1e-8, lyapunov_psd_form="quadratic",
                 rho=1.0, kappa=0, dynamics="pendulum", dynamics_version="default",
                 observer_width=8, observer_depth=2,
                 verification=False, scale_input=1.0):
        super().__init__()
        self.verification = verification
        self.valid_lower = valid_lower
        self.valid_upper = valid_upper
        self.kappa = kappa
        self.rho = rho
        self.scale_input = scale_input
        self.output_feedback = False
        if dynamics == "pendulum":
            self.dynamics = SecondOrderDiscreteTimeSystem(
                PendulumDynamics(m=0.15, l=0.5, beta=0.1),
                dt=0.05,
                position_integration=IntegrationMethod.ExplicitEuler,
                velocity_integration=IntegrationMethod.ExplicitEuler,
            )
        elif dynamics == "pendulum_small_torque":
            self.dynamics = SecondOrderDiscreteTimeSystem(
                PendulumDynamics(m=0.15, l=0.5, beta=0.1, max_u=0.75),
                dt=0.05,
                position_integration=IntegrationMethod.ExplicitEuler,
                velocity_integration=IntegrationMethod.ExplicitEuler,
            )
        elif dynamics == "path_tracking":
            self.dynamics = FirstOrderDiscreteTimeSystem(
                PathTrackingDynamics(speed=2.0, length=1.0, radius=10.0),
                dt=0.05,
            )
        elif dynamics == "path_tracking_small_torque":
            self.dynamics = FirstOrderDiscreteTimeSystem(
                PathTrackingDynamics(speed=2.0, length=1.0, radius=10.0, max_u=0.5),
                dt=0.05,
            )
        elif dynamics == "quadrotor2d":
            self.dynamics = SecondOrderDiscreteTimeSystem(
                Quadrotor2DDynamics(), dt=0.01)
        elif dynamics == "quadrotor2d_output_feedback":
            self.dynamics = SecondOrderDiscreteTimeSystem(
                Quadrotor2DLidarDynamics(), dt=0.01)
            self.output_feedback = True
        elif dynamics == "quadrotor3d":
            # self.dynamics = SecondOrderDiscreteTimeSystem(
            self.dynamics = FirstOrderDiscreteTimeSystem(
                Quadrotor3DDynamics(version=dynamics_version), dt=0.01)
        else:
            raise NameError(dynamics)
        self.x_dim = self.dynamics.continuous_time_system.x_dim
        if lyapunov_func == 'nn':
            self.lyapunov = NeuralNetworkLyapunov(
                x_dim=self.x_dim,
                goal_state=torch.zeros(self.x_dim),
                width=lyapunov_width,
                depth=lyapunov_depth,
                R_rows=lyapunov_R_rows,
                R_scale=lyapunov_R_scale,
                nn_scale=lyapunov_nn_scale,
                scale=lyapunov_scale,
                eps=lyapunov_eps,
                V_psd_form=lyapunov_psd_form,
                activation=lyapunov_act,
            )
        elif lyapunov_func == 'quadratic':
            # TODO add scale
            self.lyapunov = QuadraticLyapunov(
                x_dim=self.x_dim,
                goal_state=torch.zeros(self.x_dim),
                R_rows=lyapunov_R_rows,
                R_scale=lyapunov_R_scale,
                eps=lyapunov_eps,
            )
        else:
            raise NameError(lyapunov_func)
        self.controller = NeuralNetworkController(
            in_dim=self.dynamics.x_equilibrium.size(0),
            out_dim=self.dynamics.u_equilibrium.size(0),
            x_equilibrium=self.dynamics.x_equilibrium * self.scale_input,
            u_equilibrium=self.dynamics.u_equilibrium,
            nlayer=controller_depth,
            hidden_dim=controller_width,
            u_lo=self.dynamics.u_lo,
            u_up=self.dynamics.u_up,
            activation=controller_act,
            arch=controller_arch,
        )
        if self.output_feedback:
            self.observer = NeuralNetworkLuenbergerObserver(
                self.dynamics.continuous_time_system.nx,
                self.dynamics.continuous_time_system.ny,
                self.dynamics,
                self.dynamics.continuous_time_system.h,
                torch.zeros(1, self.dynamics.continuous_time_system.ny),
                depth=observer_depth,
                width=observer_width,
                scale_input=scale_input,
            )

    def forward(self, x, sim=False):
        if self.verification:
            # At verification time, the input boxes are not scaled.
            # Do the scaling here.
            x = x * self.scale_input

        if self.output_feedback:
            xe = x
            x = xe[:, :self.dynamics.nx]
            e = xe[:, self.dynamics.nx:]
            z = x - e
            y = self.observer.h(x / self.scale_input) * self.scale_input
            ey = y - self.observer.h(z / self.scale_input) * self.scale_input
            u = self.controller.forward(torch.cat((z, ey), dim=1))
            new_x, _ = self.dynamics.forward(x / self.scale_input, u)
            new_x = new_x * self.scale_input
            new_z = self.observer.forward(z, u, y)
            new_xe = torch.cat((new_x, new_x - new_z), dim=1)
            x = xe
            x_next = new_xe
            dx = x_next - x
        else:
            u = self.controller(x)
            x_next, dx = self.dynamics.forward(x / self.scale_input, u)
            dx = dx * self.scale_input
            x_next = x + dx

        if isinstance(self.lyapunov, NeuralNetworkLyapunov):
            lyapunov_x, d_lyapunov = self.lyapunov.V_and_dV(
                x, x_next, dx, self.kappa)
        else:
            lyapunov_x = self.lyapunov(x)
            d_lyapunov = self.lyapunov.diff(
                x, x_next, lyapunov_x=lyapunov_x, kappa=self.kappa)

        if sim:
            return x_next, self.lyapunov(x_next)

        if self.verification:
            return torch.concat([
                d_lyapunov,
                lyapunov_x,
                x_next / self.scale_input,
            ], dim=-1)
        else:
            # First output: lyapunov_x
            # Starting from the second output: we want the upper bound to be negative

            valid_lower = self.valid_lower.to(x)
            valid_upper = self.valid_upper.to(x)
            x_next_violation_lower = valid_lower - x_next
            x_next_violation_upper = x_next - valid_upper

            violation = torch.concat([
                d_lyapunov,
                x_next_violation_lower,
                x_next_violation_upper,
            ], dim=-1)

            return torch.concat([
                -lyapunov_x,
                violation,
            ], dim=-1)


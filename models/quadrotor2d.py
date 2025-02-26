import torch
from torch.nn import functional as F


class Quadrotor2DDynamics:
    """
    2D Quadrotor dynamics, based on https://github.com/StanfordASL/neural-network-lyapunov/blob/master/neural_network_lyapunov/examples/quadrotor2d/quadrotor_2d.py
    """

    def __init__(
        self, length=0.25, mass=0.486, inertia=0.00383, gravity=9.81, *args, **kwargs
    ):
        self.nx = 6
        self.nq = 3
        self.nu = 2
        # length of the rotor arm.
        self.length = length
        # mass of the quadrotor.
        self.mass = mass
        # moment of inertia
        self.inertia = inertia
        # gravity.
        self.gravity = gravity
        self.x_dim = 6

    def forward(self, x, u):
        """
        Compute the continuous-time dynamics (batched, pytorch).
        This is the actual computation that will be bounded using auto_LiRPA.
        """
        q = x[:, :3]
        qdot = x[:, 3:]
        qddot1 = (-1.0 / self.mass) * (torch.sin(q[:, 2:]) * (u[:, :1] + u[:, 1:]))
        qddot2 = (1.0 / self.mass) * (
            torch.cos(q[:, 2:]) * (u[:, :1] + u[:, 1:])
        ) - self.gravity
        qddot3 = (self.length / self.inertia) * (u[:, :1] - u[:, 1:])
        return torch.cat((qddot1, qddot2, qddot3), dim=1)

    @property
    def x_equilibrium(self):
        return torch.zeros((6,))

    @property
    def u_equilibrium(self):
        return torch.full((2,), (self.mass * self.gravity) / 2)

    @property
    def u_lo(self):
        return torch.tensor([0., 0.])

    @property
    def u_up(self):
        return self.u_equilibrium * 2.5


class Quadrotor2DLidarDynamics:
    """
    (y, theta, ydot, thetadot)
    2D Quadrotor dynamics, based on https://github.com/StanfordASL/neural-network-lyapunov/blob/master/neural_network_lyapunov/examples/quadrotor2d/quadrotor_2d.py
    """

    def __init__(
        self, length=0.25, mass=0.486, inertia=0.00383, gravity=9.81, b=0,
        *args,
        **kwargs
    ):
        self.nx = 4
        self.nq = 2
        self.nu = 2
        self.ny = 4
        # length of the rotor arm.
        self.length = length
        # mass of the quadrotor.
        self.mass = mass
        # moment of inertia
        self.inertia = inertia
        # gravity.
        self.gravity = gravity
        self.H = 5
        self.angle_max = 0.149 * torch.pi
        self.origin_height = 1
        self.x_dim = 8

    def forward(self, x, u):
        """
        Compute the continuous-time dynamics (batched, pytorch).
        This is the actual computation that will be bounded using auto_LiRPA.
        """
        qddot1 = (
            (1.0 / self.mass) * (torch.cos(x[:, 1:2]) * (u[:, :1] + u[:, 1:]))
            - self.gravity
        )
        qddot2 = (self.length / self.inertia) * (u[:, :1] - u[:, 1:])
        return torch.cat((qddot1, qddot2), dim=1)

    def h(self, x):
        y = (torch.ones(x.shape[0], self.ny, device=x.device) * x[:, :1]
             + self.origin_height)
        theta = x[:, 1:2]
        phi = theta - torch.linspace(
            -self.angle_max, self.angle_max, self.ny, device=x.device
        )
        lidar_rays = y / torch.cos(phi)
        lidar_rays = F.relu(lidar_rays)
        lidar_rays = -F.relu(self.H - lidar_rays) + self.H
        return lidar_rays

    @property
    def x_equilibrium(self):
        return torch.zeros((8,))

    @property
    def u_equilibrium(self):
        return torch.full((2,), (self.mass * self.gravity) / 2)

    @property
    def u_lo(self):
        return torch.tensor([0., 0.])

    @property
    def u_up(self):
        return self.u_equilibrium * 3

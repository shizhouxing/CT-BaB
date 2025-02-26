import torch

class Quadrotor3DDynamics:
    """
    3D Quadrotor dynamics, based on https://raw.githubusercontent.com/StanfordASL/neural-network-lyapunov/master/neural_network_lyapunov/examples/quadrotor3d/quadrotor.py

    Reference:
    https://github.com/huanzhang12/neural_lyapunov_training/blob/32665a946a6edae25be0a819fba07d011df6f9c7/quadrotor3d_training.py
    https://github.com/huanzhang12/neural_lyapunov_training/blob/32665a946a6edae25be0a819fba07d011df6f9c7/models.py

    A quadrotor that directly commands the thrusts.
    The state is [pos_x, pos_y, pos_z, roll, pitch, yaw, pos_xdot, pos_ydot,
    pos_zdot, angular_vel_x, angular_vel_y, angular_vel_z], where
    angular_vel_x/y/z are the angular velocity measured in the body frame.
    Notice that unlike many models where uses the linear velocity in the body
    frame as the state, we use the linear velocit in the world frame as the
    state. The reason is that the update from linear velocity to next position
    is a linear constraint, and we don't need to use a neural network to encode
    this update.
    """

    def __init__(
        self, length=0.225, mass=0.486, gravity=9.81,
        z_torque_to_force_factor=1.1 / 29, version='default',
        *args, **kwargs
    ):
        self.version = version
        self.nx = 12
        self.nq = 0
        self.nu = 4
        self.arm_length = 0.225
        self.mass = 0.486
        self.gravity = 9.81
        self.z_torque_to_force_factor = 1.1 / 29
        self.hover_thrust = self.mass * self.gravity / 4
        self.inertia = torch.tensor([4.9E-3, 4.9E-3, 8.8E-3])
        self.plant_input_w = torch.tensor(
            [
                [1, 1, 1, 1],
                [0, self.arm_length, 0, -self.arm_length],
                [-self.arm_length, 0, self.arm_length, 0],
                [
                    self.z_torque_to_force_factor,
                    -self.z_torque_to_force_factor,
                    self.z_torque_to_force_factor,
                    -self.z_torque_to_force_factor
                ]
            ]
        )
        self.pos_ddot_bias = torch.tensor([0, 0, -self.gravity])
        self.x_dim = 12

    def forward(self, x, u):
        self.inertia = self.inertia.to(x)
        self.plant_input_w = self.plant_input_w.to(x)
        self.pos_ddot_bias = self.pos_ddot_bias.to(x)

        rpy = x[:, 3:6]
        pos_dot = x[:, 6:9]
        omega = x[:, 9:12]

        if self.version == 'default':
            cos_rpy = torch.cos(rpy)
            sin_rpy = torch.sin(rpy)
            cos_roll = cos_rpy[:, 0:1]
            sin_roll = sin_rpy[:, 0:1]
            cos_pitch = cos_rpy[:, 1:2]
            sin_pitch = sin_rpy[:, 1:2]
            tan_pitch = torch.tan(rpy[:, 1:2])
            cos_yaw = cos_rpy[:, 2:3]
            sin_yaw = sin_rpy[:, 2:3]

            R_last_col = torch.concat([
                sin_yaw * sin_roll + cos_yaw * sin_pitch * cos_roll,
                -cos_yaw * sin_roll + sin_yaw * sin_pitch * cos_roll,
                cos_pitch * cos_roll,
            ], dim=-1)

            omega_0 = omega[:, 0:1]
            omega_1 = omega[:, 1:2]
            omega_2 = omega[:, 2:3]
            sin_cos_roll_omega = sin_roll * omega_1 + cos_roll * omega_2
            rpy_dot_0 = omega_0 + tan_pitch * sin_cos_roll_omega
            rpy_dot_1 = cos_roll * omega_1 - sin_roll * omega_2
            rpy_dot_2 = sin_cos_roll_omega / cos_pitch
        elif self.version == '0915-v1':
            # https://zh.wikipedia.org/wiki/%E4%B8%89%E8%A7%92%E6%81%92%E7%AD%89%E5%BC%8F
            # sin(a)sin(b)=1/2 (cos(a-b) - cos(a+b))
            # cos(a)cos(b)=1/2 (cos(a-b) + cos(a+b))
            # sin(a)cos(b)=1/2 (sin(a+b) + sin(a-b))

            cos_rpy = torch.cos(rpy)
            sin_rpy = torch.sin(rpy)
            cos_roll = cos_rpy[:, 0:1]
            sin_roll = sin_rpy[:, 0:1]
            cos_pitch = cos_rpy[:, 1:2]
            sin_pitch = sin_rpy[:, 1:2]
            tan_pitch = torch.tan(rpy[:, 1:2])
            cos_yaw = cos_rpy[:, 2:3]
            sin_yaw = sin_rpy[:, 2:3]

            roll = rpy[:, 0:1]
            pitch = rpy[:, 1:2]
            yaw = rpy[:, 2:3]

            yaw_minus_roll = yaw - roll
            cos_yaw_minus_roll = torch.cos(yaw_minus_roll)
            sin_yaw_minus_roll = torch.sin(yaw_minus_roll)

            yaw_plus_roll = yaw + roll
            cos_yaw_plus_roll = torch.cos(yaw_plus_roll)
            sin_yaw_plus_roll = torch.sin(yaw_plus_roll)

            pitch_minus_roll = pitch - roll
            cos_pitch_minus_roll = torch.cos(pitch_minus_roll)

            pitch_plus_roll = pitch + roll
            cos_pitch_plus_roll = torch.cos(pitch_plus_roll)

            R_last_col = torch.concat([
                # sin_yaw * sin_roll
                0.5 * (cos_yaw_minus_roll - cos_yaw_plus_roll)
                # cos_yaw * sin_pitch * cos_roll
                + 0.5 * (cos_yaw_minus_roll + cos_yaw_plus_roll) * sin_pitch,

                # sin(a)cos(b)=1/2 (sin(a+b) + sin(a-b))
                # -sin_roll * cos_yaw
                -0.5 * (sin_yaw_plus_roll - sin_yaw_minus_roll)
                # sin_yaw * cos_roll * sin_pitch
                + 0.5 * (sin_yaw_plus_roll + sin_yaw_minus_roll) * sin_pitch,

                # cos_pitch * cos_roll
                0.5 * (cos_pitch_minus_roll + cos_pitch_plus_roll),
            ], dim=-1)

            omega_0 = omega[:, 0:1]
            omega_1 = omega[:, 1:2]
            omega_2 = omega[:, 2:3]
            sin_cos_roll_omega = sin_roll * omega_1 + cos_roll * omega_2
            rpy_dot_0 = omega_0 + tan_pitch * sin_cos_roll_omega
            rpy_dot_1 = cos_roll * omega_1 - sin_roll * omega_2
            rpy_dot_2 = sin_cos_roll_omega / cos_pitch
        else:
            raise NotImplementedError

        # plant_input is [total_thrust, torque_x, torque_y, torque_z]
        plant_input = u.matmul(self.plant_input_w.t())
        pos_ddot = (self.pos_ddot_bias
                    + R_last_col * plant_input[:, 0:1] / self.mass)
        # Here we exploit the fact that the inertia matrix is diagonal.
        omega_dot = (torch.cross(-omega, self.inertia * omega) +
                     plant_input[:, 1:]) / self.inertia

        ret = torch.cat([
            pos_dot,
            rpy_dot_0, rpy_dot_1, rpy_dot_2,
            pos_ddot,
            omega_dot
        ], dim=-1)

        return ret

    @property
    def x_equilibrium(self):
        return torch.zeros((12,))

    @property
    def u_equilibrium(self):
        return torch.full((4,), self.hover_thrust)

    @property
    def u_lo(self):
        return torch.zeros(4)

    @property
    def u_up(self):
        return self.u_equilibrium * 3
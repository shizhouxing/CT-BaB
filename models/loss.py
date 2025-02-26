import torch
import torch.nn as nn
from torch.nn import functional as F


class LyapunovLoss(nn.Module):
    def __init__(self, valid_lower, valid_upper,
                 rho_penalty=1.0, margin=0., margin_adv=0., margin_rho=1.0,
                 kappa=0, kappa_adv=0, rho=1.0, loss_rho_size=1.0,
                 rho_ratio=1.0, loss_rho_clip=True, loss_x_next_weight=1.0,
                 hole_size=0.0, border_size=None,
                 loss_rho_sorted=False, loss_max=False, normalize=False, device=None):
        super().__init__()
        self.valid_lower = valid_lower.to(device)
        self.valid_upper = valid_upper.to(device)
        self.x_dim = self.valid_lower.shape[-1]
        self.kappa = kappa
        self.kappa_adv = kappa_adv
        self.loss_rho_size = loss_rho_size
        self.loss_rho_sorted = loss_rho_sorted
        self.rho_penalty = rho_penalty
        self.rho_ratio = rho_ratio
        self.loss_rho_clip = loss_rho_clip
        self.rho = rho
        self.margin = margin
        self.margin_adv = margin_adv
        self.margin_rho = margin_rho
        self.hole_size = hole_size
        self.border_size = border_size
        self.loss_x_next_weight = loss_x_next_weight
        self.normalize = normalize
        self.loss_max = loss_max

        self.adv_dim = 2 * self.valid_lower.shape[-1] + 1
        self.adv_mapping = torch.zeros(self.adv_dim, self.adv_dim, self.adv_dim, device=device)
        for i in range(self.adv_dim):
            self.adv_mapping[i, i, i] = 1

    def get_adv_loss(self, y):
        lyapunov_x, violation = -y[:, 0], y[:, 1:]
        batch_size = y.shape[0] // self.adv_dim
        lyapunov_x = lyapunov_x.view(batch_size, self.adv_dim)
        violation = violation.view(batch_size, self.adv_dim, self.adv_dim)
        violation = torch.diagonal(violation, dim1=1, dim2=2)
        violation[:, 0] = violation[:, 0] + self.kappa_adv * lyapunov_x[:, 0]

        inside = lyapunov_x <= self.rho
        loss_adv = torch.where(
            inside,
            violation + self.margin_adv,
            self.rho - lyapunov_x,
        ).view(-1)

        return loss_adv

    def _get_rho_loss(self, y):
        lyapunov_x = -y[:, 0]
        inside = lyapunov_x <= self.rho
        rho_violation = F.relu(lyapunov_x - self.rho)
        if self.loss_rho_clip and inside.float().mean() >= self.rho_ratio:
            loss_rho = torch.tensor(0.).to(lyapunov_x)
        elif self.loss_rho_sorted:
            sorted_rho_violation = torch.sort(rho_violation)[0]
            num_rho_samples = int(rho_violation.shape[0] * self.rho_ratio)
            # loss_rho = sorted_rho_violation[:num_rho_samples].sum() / rho_violation.shape[0]
            loss_rho = sorted_rho_violation[:num_rho_samples].mean()
        else:
            loss_rho = rho_violation.mean()
        return {
            'loss': loss_rho,
            'inside': inside
        }

    def _get_empirical_loss(self, y):
        lyapunov_x = -y[:, :, 0]
        violation = y[:, :, 1:]

        inside = lyapunov_x <= self.rho
        safe_violation = ((violation <= 0).all(dim=-1))
        safe = ((~inside) | safe_violation)
        safe_violation = safe_violation.all(dim=-1)
        safe = safe.all(dim=-1)
        ret = {
            'safe': safe,
            'safe_violation': safe_violation,
        }

        inside = lyapunov_x <= self.rho * (1 + self.margin_rho)
        violation_V = violation[:, :, 0]
        violation_next = F.relu(violation[:, :, 1:]).sum(dim=-1)
        violation_V = violation_V + lyapunov_x * self.kappa_adv
        if self.kappa_adv > 0:
            violation_V = F.relu(violation_V)# + self.margin)
        else:
            if True:#self.normalize:
                scale = lyapunov_x.clamp(min=1e-8, max=1.0)
                violation_V = F.relu(violation_V + self.margin_adv * scale) / scale
            else:
                violation_V = F.relu(violation_V + self.margin_adv)

        next_in = violation_next <= 0
        # if self.loss_max:
        violation_V = torch.where(next_in, violation_V, torch.max(violation_V, self.rho - lyapunov_x))
        # else:
        #     violation_V = next_in * violation_V + (~next_in) * self.rho
        loss = inside * (violation_V + violation_next * self.loss_x_next_weight)

        inside = inside.any(dim=-1)
        loss = loss.sum(dim=-1)

        ret.update({
            'loss': loss,
            'inside': inside,
        })

        return ret

    def _get_verified_loss(self, y, y_M):
        lyapunov_x = -y[:, 0]
        violation = y[:, 1:]

        inside = lyapunov_x <= self.rho
        safe_violation = (violation <= 0).all(dim=-1)
        safe = (~inside) | safe_violation
        if violation.numel() > 0:
            max_violation = (inside * F.relu(violation).sum(dim=-1)).max()
        else:
            max_violation = 0.

        ret = {
            'safe': safe,
            'safe_violation': safe_violation,
            'max_violation': max_violation,
        }

        inside = lyapunov_x <= self.rho * (1 + self.margin_rho)
        violation_V = violation[:, 0]
        violation_next = F.relu(violation[:, 1:]).sum(dim=-1)

        if self.normalize:
            scale = (-y_M[:, 0]).clamp(min=1e-8, max=1.0)
            violation_V = F.relu(violation_V + self.margin * scale) / scale
        else:
            violation_V = F.relu(violation_V + self.margin)

        next_in = violation_next <= 0

        if self.loss_max:
            violation_V = torch.where(next_in, violation_V, torch.max(violation_V, self.rho - lyapunov_x))
        else:
            violation_V = next_in * violation_V + (~next_in) * self.rho
        loss = inside * (violation_V + violation_next * self.loss_x_next_weight)

        ret.update({
            'loss': loss,
            'inside': inside,
        })
        return ret

    def forward(self, loss_term, y=None, y_M=None):
        if loss_term == 'empirical':
            return self._get_empirical_loss(y)
        elif loss_term == 'rho':
            return self._get_rho_loss(y)
        elif loss_term == 'verified':
            return self._get_verified_loss(y, y_M)
        else:
            return NameError(loss_term)

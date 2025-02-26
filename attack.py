import torch
from auto_LiRPA import BoundedTensor, PerturbationLpNorm


class PGD:
    def __init__(self, model, loss_func, alpha, restarts, steps):
        self.model = model
        self.loss_func = loss_func
        self.alpha = alpha
        self.restarts = restarts
        self.steps = steps

    def attack(self, x_L, x_U, loss_func=None):
        if loss_func is None:
            loss_func = self.loss_func.get_adv_loss
            adv_dim = self.loss_func.adv_dim
        else:
            adv_dim = 1

        grad_status = {}
        for p in self.model.parameters():
            grad_status[p] = p.requires_grad
            p.requires_grad_(False)

        batch_size = x_L.shape[0]
        x_L_ = x_L.unsqueeze(1).unsqueeze(1).expand(
            batch_size, self.restarts, adv_dim, -1)
        x_U_ = x_U.unsqueeze(1).unsqueeze(1).expand(
            batch_size, self.restarts, adv_dim, -1)
        x = (x_L_ + torch.rand_like(x_L_) * (x_U_ - x_L_)).requires_grad_()
        alpha = (x_U_ - x_L_) / 2 * self.alpha

        for _ in range(self.steps):
            y = self.model(x.view(-1, x.shape[-1]))
            loss = loss_func(y).mean()
            loss.backward()
            grad = x.grad.detach()
            x.data = x + alpha * torch.sign(grad)
            x.data = torch.max(x_L_, torch.min(x, x_U_))
            x.grad.zero_()

        x_adv = x.detach().view(-1, x.shape[-1])

        for p in self.model.parameters():
            p.requires_grad_(grad_status[p])
            p.grad = None

        return x_adv

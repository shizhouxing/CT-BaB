import torch


def bf_split(x_L, x_U, ratio=0.5, flat=False, three_branch=False):
    input_dim = x_L.shape[-1]
    num_branches = 3 if three_branch else 2
    new_x_L = x_L.expand(num_branches, input_dim, -1, -1).clone()
    new_x_U = x_U.expand(num_branches, input_dim, -1, -1).clone()
    if three_branch:
        x_M1 = x_L * (1 - ratio) + x_U * ratio
        x_M2 = x_L * ratio + x_U * (1 - ratio)
        for i in range(input_dim):
            new_x_U[0, i, :, i] = x_M1[:, i]
            new_x_L[1, i, :, i] = x_M1[:, i]
            new_x_U[1, i, :, i] = x_M2[:, i]
            new_x_L[2, i, :, i] = x_M2[:, i]
    else:
        x_M = x_L * (1 - ratio) + x_U * ratio
        for i in range(input_dim):
            new_x_U[0, i, :, i] = x_M[:, i]
            new_x_L[1, i, :, i] = x_M[:, i]
    if flat:
        new_x_L = new_x_L.view(-1, input_dim)
        new_x_U = new_x_U.view(-1, input_dim)
    return new_x_L, new_x_U


def split_with_decision(x_L, x_U, decision, ratio=0.5, lower_limit=None, upper_limit=None):
    if decision.ndim == 1:
        decision = decision.unsqueeze(-1)

    x_M = (x_L + x_U) / 2
    x_M = torch.gather(x_M, dim=-1, index=decision)

    if ratio != 0.5:
        # FIXME this requires all the dimensions to be equal
        lower_limit, upper_limit = lower_limit.min(), upper_limit.max()

        x_L_dec = torch.gather(x_L, dim=-1, index=decision)
        x_U_dec = torch.gather(x_U, dim=-1, index=decision)
        mask_neg = (x_L_dec == lower_limit) & (x_U_dec == 0)
        point_neg = x_L_dec * ratio + x_U_dec * (1 - ratio)
        x_M[mask_neg] = point_neg[mask_neg]
        mask_pos = (x_L_dec == 0) & (x_U_dec == upper_limit)
        point_pos = x_L_dec * (1 - ratio) + x_U_dec * ratio
        x_M[mask_pos] = point_pos[mask_pos]

    x_U_left = x_U.clone()
    x_L_right = x_L.clone()
    x_U_left.scatter_(dim=-1, index=decision, src=x_M)
    x_L_right.scatter_(dim=-1, index=decision, src=x_M)
    return x_L_right, x_U_left

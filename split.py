import time
import torch
from torch.nn import functional as F
from tqdm import tqdm
from dataset import get_data_loader
from bound_utils import get_verified_result, get_bounds
from auto_LiRPA import BoundedTensor, PerturbationLpNorm
from branching_domains import UnsortedInputDomainList
from utils import logger
from split_utils import bf_split, split_with_decision


@torch.no_grad()
def bf_split_and_bound(x_L, x_U, model, loss_func, args):
    input_dim = x_L.shape[-1]
    new_x_L, new_x_U = bf_split(x_L, x_U)
    new_x_L_flat = new_x_L.view(-1, input_dim)
    new_x_U_flat = new_x_U.view(-1, input_dim)

    new_ub_all = []
    for i in range((new_x_L_flat.shape[0] + args.split_batch_size - 1) // args.split_batch_size):
        new_x_L_ = new_x_L_flat[i*args.split_batch_size:(i+1)*args.split_batch_size]
        new_x_U_ = new_x_U_flat[i*args.split_batch_size:(i+1)*args.split_batch_size]
        new_x = BoundedTensor(
            new_x_L_, ptb=PerturbationLpNorm(x_L=new_x_L_, x_U=new_x_U_))
        new_output = model(new_x)
        new_ub = get_bounds(model, 'full-CROWN', args=args)
        # new_output = model(new_x)
        # new_ub_ibp = get_bounds(model, 'IBP', args=args)
        # new_ub = torch.min(new_ub, new_ub_ibp)
        new_ub_all.append(new_ub)
    new_ub = torch.concat(new_ub_all, dim=0)

    new_margin = torch.concat([
        (-new_ub[:, 0] - loss_func.rho).unsqueeze(-1),
        -new_ub[:, 1:].amax(dim=-1, keepdim=True)
    ], dim=-1)
    new_margin = new_margin.view(2, input_dim, -1, 2)#.amax(dim=-1)

    return new_x_L, new_x_U, new_margin


def get_branching_decision_eval(old_lb, x_L, x_U, model, loss_func, args):
    ori_size = x_L.shape[0]
    input_dim = x_L.shape[-1]
    new_x_L, new_x_U, new_margin = bf_split_and_bound(x_L, x_U, model, loss_func, args)
    new_margin_amax = new_margin.amax(dim=-1)

    # Prune domains
    both_branch_verified = new_margin_amax.amin(dim=0).amax(dim=0) > 0
    # print('Both branch verified after a single split: '
    #       f'{both_branch_verified.sum()}/{both_branch_verified.shape[0]}')
    for i in range(input_dim):
        left_verified = new_margin_amax[0, i] > 0
        x_L[left_verified, i] = new_x_L[1, i, left_verified, i]
        right_verified = new_margin_amax[1, i] > 0
        x_U[right_verified, i] = new_x_U[0, i, right_verified, i]
        # print(f'Prune dim {i}: '
        #       f'left {(left_verified & ~both_branch_verified).sum()} '
        #       f'right {(right_verified & ~both_branch_verified).sum()}')
    x_L = x_L[~both_branch_verified]
    x_U = x_U[~both_branch_verified]
    old_lb = old_lb[~both_branch_verified]
    new_margin = new_margin[:, :, ~both_branch_verified]
    decision = new_margin.sum(dim=-1).mean(dim=0).argmax(dim=0)
    gain = new_margin.amax(dim=-1).amin(dim=0).amax(dim=0) - old_lb.squeeze(-1)

    decision_naive = get_branching_decision_naive(x_L, x_U)
    too_bad = gain < 1e-3
    try:
        decision[too_bad] = decision_naive[too_bad]
    except:
        breakpoint()

    print(f'Pruned: {ori_size} -> {x_L.shape[0]}')
    if x_L.shape[0] > 0:
        # worst_idx = old_lb.amax(dim=-1).argmin(dim=0)
        margin_with_decision = torch.gather(
            new_margin,
            index=decision.view(1, 1, -1, 1).expand(
                new_margin.shape[0], 1, -1, new_margin.shape[-1]),
            dim=1
        ).squeeze(1)
        worst_idx = margin_with_decision.amax(dim=-1).amin(dim=0).argmin(dim=0)
        print(f'  Worst {worst_idx}: {old_lb[worst_idx]} '
              f'-> {new_margin[:, decision[worst_idx], worst_idx]}')
        for i in range(x_L[worst_idx].shape[-1]):
            print('  ', end='')
            if i == decision[worst_idx]:
                print('->', end='')
            print(f'dim {i}: {x_L[worst_idx][i].item()} {x_U[worst_idx][i].item()}')

    return x_L, x_U, decision, new_margin


def get_branching_decision_sb(x_L, x_U, A, lb):
    perturb = (x_U - x_L).unsqueeze(-2)
    score = A.abs().clamp(min=1e-3) * perturb / 2
    score = score.sum(dim=-2)
    return torch.topk(score, 1, -1).indices


def get_branching_decision(x_L, x_U, model, loss_func, args, A=None, lb=None):
    if args.split_heuristic == 'sb':
        return get_branching_decision_sb(x_L, x_U, A, lb)
    elif args.split_heuristic == 'naive' or x_L.shape[0] > args.split_max_bf_domains:
        return get_branching_decision_naive(x_L, x_U)

    # new_x_L, new_x_U, new_margin = bf_split_and_bound(x_L, x_U, model, loss_func, args)
    # new_margin_amax = new_margin.amax(dim=-1)

    # decision = new_margin.sum(dim=-1).mean(dim=0).argmax(dim=0)
    # gain = new_margin.amax(dim=-1).amin(dim=0).amax(dim=0) - lb.amax(dim=-1)
    # decision_naive = get_branching_decision_naive(x_L, x_U)
    # too_bad = gain < 1e-3
    # decision[too_bad] = decision_naive[too_bad]

    input_dim = x_L.shape[-1]
    new_x_L = x_L.expand(2, input_dim, -1, -1).clone()
    new_x_U = x_U.expand(2, input_dim, -1, -1).clone()
    x_M = (x_L + x_U) / 2
    for i in range(input_dim):
        new_x_U[0, i, :, i] = x_M[:, i]
        new_x_L[1, i, :, i] = x_M[:, i]
    new_x_L = new_x_L.view(-1, new_x_L.shape[-1])
    new_x_U = new_x_U.view(-1, new_x_U.shape[-1])

    loss = []
    lb = []
    if args.split_batch_size is None:
        args.split_batch_size = args.batch_size
    for i in range((new_x_L.shape[0] + args.split_batch_size - 1) // args.split_batch_size):
        new_x_L_ = new_x_L[i*args.split_batch_size:(i+1)*args.split_batch_size]
        new_x_U_ = new_x_U[i*args.split_batch_size:(i+1)*args.split_batch_size]
        ret_branched = get_verified_result(
            x_L=new_x_L_, x_U=new_x_U_, model=model,
            loss_func=loss_func, args=args,
        )
        lb.append(-F.relu(ret_branched['ub'][:, 1:]).sum(keepdim=True, dim=-1))
        loss.append(ret_branched['loss'])
    loss = torch.concat(loss, dim=0).reshape(2, input_dim, -1)
    lb = torch.concat(lb).reshape(2, input_dim, -1)
    score = loss.sum(dim=0)
    if args.split_heuristic == 'sum_with_min':
        score = score - 10000 * (loss.amin(dim=0) == 0)
    decision = score.argmin(dim=0)

    return decision


def get_branching_decision_naive(x_L, x_U, hint=True):
    if hint:
        return (x_U - x_L + 10000 * ((x_L < 0) & (x_U > 0))).argmax(dim=-1)
    else:
        return (x_U - x_L).argmax(dim=-1)


@torch.no_grad()
def input_split(dataset, unsafe, x_L, x_U, weight, idx, model,
                loss_func, ret, step, args=None):
    x_L, x_U, weight, idx = x_L[unsafe], x_U[unsafe], weight[unsafe], idx[unsafe]
    if step >= args.max_split_domains_start_steps:
        violation = ret['verified']['ub'][unsafe][:, 1:].clamp(min=0).sum(dim=-1)
        _, indices = torch.sort(violation, descending=True)
        mask = torch.zeros(indices.shape[0], dtype=torch.bool)
        mask[:args.max_split_domains] = True
        mask[violation > args.split_ub_thresh] = True
        indices = indices[mask]
        x_L, x_U, weight, idx = x_L[indices], x_U[indices], weight[indices], idx[indices]

    if args.split_heuristic == 'sb': #!= 'naive':
        ub = ret['verified']['ub'][unsafe]
        uA = ret['verified']['uA'][unsafe]
        lb = torch.concat([
            (-ub[:, 0] - loss_func.rho).unsqueeze(-1),
            -ub[:, 1:].amax(dim=-1, keepdim=True)
        ], dim=-1)
        lA = -uA
    else:
        lA = lb = None

    decision = get_branching_decision(
        x_L, x_U, A=lA, lb=lb, model=model, loss_func=loss_func, args=args)

    x_L_right, x_U_left = split_with_decision(
        x_L, x_U, decision, ratio=args.split_ratio,
        lower_limit=loss_func.valid_lower,
        upper_limit=loss_func.valid_upper)
    dataset.add_split(idx.cpu(), x_L_right.cpu(), x_U_left.cpu(), (weight / 2).cpu())

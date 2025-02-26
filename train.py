"""Robust training framework."""

import copy
import os
import time
import math
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from huggingface_hub import HFSummaryWriter

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from auto_LiRPA.utils import MultiAverageMeter

from models import *
from models.loss import LyapunovLoss
from dataset import InputDomainDataset, get_data_loader, prepare_dataset
from attack import PGD
from utils import (load_checkpoint, load_checkpoint_controller_only,
                   save_checkpoint, set_seed, get_optimizer,
                   print_selected_meters, set_file_handler, logger,
                   write_tensorboard, scale_grad_norm)
from arguments import parse_args
from bound_utils import get_verified_result, get_bounds
from split import input_split
from plot import plot_V, load_V_baseline
from roa import eval_roa
from generate_vnnlib import generate_instances
from project import project_params
from init import initialize


args = parse_args()
args.eval = args.eval or args.plot

if not args.eval:
    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

    hf_writer = HFSummaryWriter(args.hf, repo_private=True) if args.hf else None
    writer = SummaryWriter(os.path.join(args.dir, 'log'), flush_secs=10)
    set_file_handler(logger, args.dir)

logger.info('Arguments: %s', args)


def get_rho_loss(model, loss_func, x_L):
    box_radius = (loss_func.valid_upper - loss_func.valid_lower) / 2
    box_center = (loss_func.valid_upper + loss_func.valid_lower) / 2
    x_ref = (torch.rand_like(x_L) * 2 - 1) * args.loss_rho_size * box_radius + box_center
    output_ref = model(x_ref)
    ret_rho = loss_func(loss_term='rho', y=output_ref)
    return ret_rho


def get_observer_loss(model, model_ori, loss_func, x_L, x_U):
    x = torch.rand_like(x_L) * (x_U - x_L) + x_L
    output = model(x)
    x_next_vio = output[:, -x.shape[-1]:]
    x_next = x_next_vio + loss_func.valid_upper
    error = x_next[:, model_ori.dynamics.nx:]
    loss = (error**2).sum(dim=-1).mean(dim=0)
    return loss


def get_empirical_result(model, loss_func, attacker, dataset, x_L, x_U, meter):
    model.eval()
    x_adv = attacker.attack(x_L, x_U)
    model.train()
    output_adv = model(x_adv)
    ret_empirical = loss_func(loss_term='empirical', y=output_adv)
    return ret_empirical


def get_new_adv_guided_result(x_L, x_U, model, model_ori, attacker, dataset, loss_func, args, meter, step):
    # Initial verification to identify which domains are unsafe
    model.train()
    x_M = (x_L + x_U) / 2
    x = BoundedTensor(x_L, PerturbationLpNorm(x_L=x_L, x_U=x_U))
    output_M = model(x) if loss_func.normalize else None
    ub, uA = get_bounds(model, return_A=True, args=args)

    ret_verified = loss_func(loss_term='verified', y=ub, y_M=output_M)
    ret_verified.update({'ub': ub, 'uA': uA})

    model.eval()
    # Try to find adversarial examples within the levelset
    batch_size_attack = x_L.shape[0] #unsafe.sum()
    # x_adv = attacker.attack(x_L[unsafe], x_U[unsafe])

    x_adv = attacker.attack(x_L, x_U)

    model.train()
    output_adv = model(x_adv)
    output_adv = output_adv.reshape(batch_size_attack, -1, output_adv.shape[-1])
    ret_empirical = loss_func(loss_term='empirical', y=output_adv)

    if args.loss_adv_guided_weight > 0:
        # TODO
        # # For unsafe domains, also try to empically bound the levelset value
        unsafe = ~ret_verified['safe']
        x_L_unsafe, x_U_unsafe = x_L[unsafe], x_U[unsafe]
        x_adv_V = attacker.attack(x_L_unsafe, x_U_unsafe, loss_func=lambda y: y[:, 0])
        output_adv_V = model(x_adv_V)[:, 0]

        if args.adv_guided_version in ['v4', 'v4.1']:
            x_adv_V_2 = torch.where(x_U_unsafe < 0, x_U_unsafe, F.relu(x_L_unsafe))
            output_adv_V_2 = model(x_adv_V_2)[:, 0]
            output_adv_V = torch.max(output_adv_V, output_adv_V_2)

        if args.adv_guided_version == 'v2':
            mask_out = -output_adv_V > loss_func.rho * args.adv_guided_thresh
        elif args.adv_guided_version in ['v3', 'v4', 'v4.1']:
            mask_out = -output_adv_V > loss_func.rho * args.adv_guided_thresh
        else:
            raise NotImplementedError
        if mask_out.any():
            guide_index = unsafe.nonzero().squeeze(-1)[mask_out]
            output_adv_V = output_adv_V[mask_out]
            adv_guided_loss = ub[guide_index, 0] - output_adv_V.detach()
            meter.update('verified/adv_guided_loss', adv_guided_loss.sum())
            meter.update('verified/adv_guided_ratio', mask_out.float().mean(), mask_out.shape[0])
            if args.adv_guided_version in ['v2', 'v4.1']:
                ret_verified['loss'][guide_index] += adv_guided_loss * args.loss_adv_guided_weight
            elif args.adv_guided_version in ['v3', 'v4']:
                # Adv guided loss only
                ret_verified['loss'][guide_index] = adv_guided_loss * args.loss_adv_guided_weight
            else:
                raise NotImplementedError

    return ret_empirical, ret_verified


def train_step(model, model_ori, attacker, x_L, x_U, weight,
               loss_func, meter, opt=None, dataset=None, epoch=None, step=None):
    ret_rho = get_rho_loss(model, loss_func, x_L)
    if args.tune_rho and step >= args.split_start_steps:
        assert args.tune_rho_weight >= 1.0
        inside = ret_rho['inside'].float().mean()
        if inside > args.rho_ratio * 2:
            loss_func.rho /= args.tune_rho_weight
            ret_rho['loss'] = 0
        elif args.tune_rho_enlarge and inside < args.rho_ratio * 0.5:
            loss_func.rho *= args.tune_rho_weight
        meter.update('main/rho', loss_func.rho)

    if args.adv_only:
        ret_empirical = get_empirical_result(
            model, loss_func, attacker, dataset, x_L, x_U, meter)
        ret_verified = {k:v for k, v in ret_empirical.items()}
    else:
        ret_empirical, ret_verified = get_new_adv_guided_result(
            x_L=x_L, x_U=x_U, model=model, model_ori=model_ori,
            attacker=attacker, dataset=dataset, loss_func=loss_func,
            args=args, meter=meter, step=step,
        )

    included_adv = ret_empirical['inside']
    num_included_adv = included_adv.int().sum()
    included_verified = ret_verified['inside']
    num_included_verified = included_verified.int().sum()

    if args.loss_for_included_only:
        if included_adv.any():
            loss_empirical = ret_empirical['loss'][included_adv].mean()
        else:
            loss_empirical = 0.
        if included_verified.any():
            loss_verified = (ret_verified['loss'][included_verified]).mean()
        else:
            loss_verified = 0.
    else:
        loss_empirical = ret_empirical['loss'].mean()
        loss_verified = ret_verified['loss'].mean()
    loss = loss_verified * args.loss_verified_weight

    if (~ret_empirical['safe']).sum() >= args.focus_emp:
        loss = loss * args.focus_emp_weight

    loss = loss + loss_empirical * args.loss_empirical_weight

    loss = loss + ret_rho['loss'] * args.rho_penalty

    if args.sim:
        # Run simulation
        x = x_L + (x_U - x_L) * torch.rand_like(x_L)
        lyap = model_ori.lyapunov(x)
        mask = (lyap <= args.rho).squeeze(-1)
        if not mask.any():
            logger.warning('No simulation')
        else:
            x = x[mask]
            lyap = lyap[mask]
            loss_sim = 0.
            safe_sim = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
            for step in range(args.sim_steps):
                x_next, lyap_next = model_ori(x, sim=True)
                risk_roi = (
                    F.relu(loss_func.valid_lower - x_next).sum(dim=-1, keepdim=True)
                    + F.relu(x_next - loss_func.valid_upper).sum(dim=-1, keepdim=True)
                )
                risk_lyap = F.relu(lyap_next - (1 - args.kappa) * lyap)
                loss_ = (risk_roi + risk_lyap).squeeze(-1) * safe_sim
                safe_ = (
                    (x_next >= loss_func.valid_lower).all(dim=-1, keepdim=True)
                    & (x_next <= loss_func.valid_upper).all(dim=-1, keepdim=True)
                    & (lyap_next <= (1 - args.kappa) * lyap)
                ).squeeze(-1)
                safe_sim = safe_sim & safe_
                loss_sim = loss_sim + loss_
                x = x_next
                lyap = lyap_next
            loss_sim = loss_sim / args.sim_steps
            loss_sim = loss_sim.mean()
            loss = loss + loss_sim * args.loss_sim_weight
            meter.update('sim/loss', loss_sim, x.shape[0])
            meter.update('sim/safe', safe_sim.float().mean(), x.shape[0])

    if model_ori.output_feedback:
        loss_observer = get_observer_loss(model, model_ori, loss_func, x_L, x_U)
        meter.update('main/observer', loss_observer)
        loss = loss + loss_observer * args.loss_observer_weight

    loss = loss * args.loss_scale

    if meter is not None:
        meter.update('main/inside', ret_rho['inside'].float().mean())
        meter.update('main/loss', loss)

        meter.update('adv/loss', loss_empirical)
        meter.update('adv/included', included_adv.float().mean(),
                     included_adv.shape[0])
        meter.update('verified/loss', loss_verified)
        meter.update('verified/included', included_verified.float().mean(),
                     included_verified.shape[0])

        meter.update('main/loss_rho', ret_rho['loss'])
        meter.update('main/unsafe', (~ret_empirical['safe']).sum())
        if num_included_adv > 0:
            for k in ['safe', 'safe_violation']:
                target_k = 'main/adv_safe' if k == 'safe' else f'adv/{k}'
                if isinstance(ret_empirical[k], torch.Tensor) and ret_empirical[k].numel() > 1:
                    meter.update(target_k,
                                 ret_empirical[k].float().mean(),
                                 ret_empirical[k].shape[0])
                else:
                    meter.update(target_k, ret_empirical[k])
        if not args.adv_only and num_included_verified > 0:
            for k in ['safe', 'safe_violation', 'max_violation']:
                if k not in ret_verified:
                    continue
                if ret_verified[k] is None:
                    continue
                target_k = 'main/ver_safe' if k == 'safe' else f'verified/{k}'
                if isinstance(ret_verified[k], torch.Tensor) and ret_verified[k].numel() > 1:
                    meter.update(target_k,
                                 ret_verified[k].float().mean(),
                                 ret_verified[k].shape[0])
                else:
                    meter.update(target_k, ret_verified[k].float())

        meter.update('main/ver_safe_in',
                     ret_verified['safe'][included_verified].float().mean(),
                     num_included_verified)

    loss.backward()
    if scale_grad_norm:
        grad_norm = scale_grad_norm(
            model_ori.parameters(), args.grad_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model_ori.parameters(), max_norm=args.grad_norm)

    opt.step()
    if args.proj_params:
        project_params(model_ori)
    opt.zero_grad()

    if meter is not None:
        meter.update('train/grad_norm', grad_norm, 1)
        meter.update('train/lr', opt.param_groups[0]['lr'], 1)

    return {
        'safe': ret_verified['safe'],
        'inside': included_verified,
        'empirical': ret_empirical,
        'verified': ret_verified,
    }


def main(args):
    set_seed(args.seed)

    dataset = prepare_dataset(args)
    logger.info('Dataset: %s', dataset)

    model_ori = ModelControl(
        valid_lower=dataset.lower_limit,
        valid_upper=dataset.upper_limit,
        controller_width=args.controller_width,
        controller_depth=args.controller_depth,
        controller_act=args.controller_act,
        controller_arch=args.controller_arch,
        lyapunov_func=args.lyapunov_func,
        lyapunov_R_rows=args.lyapunov_R_rows,
        lyapunov_R_scale=args.lyapunov_R_scale,
        lyapunov_nn_scale=args.lyapunov_nn_scale,
        lyapunov_width=args.lyapunov_width,
        lyapunov_depth=args.lyapunov_depth,
        lyapunov_psd_form=args.lyapunov_psd_form,
        lyapunov_eps=args.lyapunov_eps,
        lyapunov_scale=args.lyapunov_scale,
        lyapunov_act=args.lyapunov_act,
        observer_width=args.observer_width,
        observer_depth=args.observer_depth,
        dynamics=args.dynamics,
        dynamics_version=args.dynamics_version,
        kappa=args.kappa,
        scale_input=args.scale_input,
    ).to(args.device)
    model_ori.eval()
    logger.info('Model: %s', model_ori)

    if args.init:
        initialize(model_ori, dataset, scale=args.init_scale, args=args)

    model = BoundedModule(
        model_ori,
        (torch.zeros(2, model_ori.x_dim, device=args.device),),
        bound_opts={
            'relu': args.relu_relaxation,
            'drelu': args.drelu_relaxation,

            # 'sparse_intermediate_bounds': False,
            # 'sparse_conv_intermediate_bounds': False,
            # 'sparse_intermediate_bounds_with_ibp': False,
            # 'sparse_features_alpha': False,
            # 'sparse_spec_alpha': False,

            'mul': {'middle': args.mul_middle},
            'compare_crown_with_ibp': args.compare_crown_with_ibp,
            'conv_mode': 'matrix',

            'disable_optimization': ['BoundMul', 'BoundMatMul'],
        },
        device=args.device)
    logger.info('Bounded model: %s', model)

    opt = get_optimizer(model, args)
    loss_func = LyapunovLoss(valid_lower=dataset.lower_limit,
                             valid_upper=dataset.upper_limit,
                             rho=args.rho,
                             loss_rho_size=args.loss_rho_size,
                             loss_rho_sorted=args.loss_rho_sorted,
                             rho_penalty=args.rho_penalty,
                             rho_ratio=args.rho_ratio,
                             loss_rho_clip=args.loss_rho_clip,
                             border_size=args.border_size,
                             hole_size=args.hole_size,
                             normalize=args.loss_normalize,
                             kappa=args.kappa,
                             kappa_adv=args.kappa_adv,
                             margin=args.margin,
                             margin_adv=args.margin_adv,
                             margin_rho=args.margin_rho,
                             loss_x_next_weight=args.loss_x_next_weight,
                             loss_max=args.loss_max,
                             device=args.device)

    model_init = copy.deepcopy(model_ori.state_dict())
    if args.load or args.load_last:
        epoch, step = load_checkpoint(args, args.load, model_ori,
                                      dataset=dataset, loss_func=loss_func, opt=opt)
    else:
        epoch = step = 0
    if args.load_controller:
        load_checkpoint_controller_only(args, args.load_controller, model_ori)
        for param in model_ori.controller.parameters():
            param.requires_grad = False

    attacker = PGD(
        model=model_ori, loss_func=loss_func,
        alpha=args.pgd_alpha, restarts=args.pgd_restarts,
        steps=args.pgd_steps,
    )
    lr_ = args.lr
    if args.lr_scheduler == 'multistep':
        lr_milestones = []
        for i in range(args.steps // args.lr_decay_interval):
            lr_milestones.append((i + 1) * args.lr_decay_interval)
            lr_ *= args.lr_decay
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=lr_milestones, gamma=args.lr_decay)

    elif args.lr_scheduler == 'cosine':
        num_cycles = 0.5
        factor_func = lambda step: (
            (float(step) / float(max(1, args.lr_warmup_steps)))
            if step < args.lr_warmup_steps
            else max(0.0, 0.5 * (1.0 + math.cos(
                math.pi * float(num_cycles) * 2.0
                * float(step - args.lr_warmup_steps)
                / float(max(1, args.steps - args.lr_warmup_steps)))))
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, factor_func)
    else:
        raise NotImplementedError

    meter = MultiAverageMeter()

    for i in range(step):
        lr_scheduler.step()
    logger.info('Learning rate: %s', [p['lr'] for p in opt.param_groups])

    if args.plot:
        V_baseline = load_V_baseline(args=args)
        plot_V(model=model_ori, rho=args.rho, V_baseline=V_baseline,
               lower_limit=dataset.lower_limit, upper_limit=dataset.upper_limit,
               save_path=args.dir, args=args)
        exit(0)

    if args.eval:
        eval_roa(
            model_ori.lyapunov, dataset.lower_limit, dataset.upper_limit,
            dataset.hole_lower, dataset.hole_upper,
            rho=args.rho, scale_input=args.scale_input,
            num_ticks=args.eval_roa_ticks)
        exit(0)

    cex = torch.zeros(len(dataset), device=args.device, dtype=torch.bool)

    while epoch < args.epochs:
        epoch += 1
        if step >= args.steps:
            break
        inside_only = epoch % args.sample_all_interval > 0 and step >= args.split_start_steps
        train_data = get_data_loader(
            dataset, batch_size=args.batch_size,
            num_workers=args.num_data_workers, inside_only=inside_only)
        logger.info(f'Epoch {epoch}: '
                    f'{len(dataset)} examples, '
                    f'{len(train_data)} batches')
        num_batches = len(train_data)
        start_time = time.time()
        start_step = step
        for i, (x_L, x_U, weight, idx) in enumerate(train_data):
            step += 1
            x_L = x_L.to(args.device)
            x_U = x_U.to(args.device)
            weight = weight.to(args.device)
            idx = idx.to(args.device)

            ret = train_step(
                model=model, model_ori=model_ori, attacker=attacker,
                x_L=x_L, x_U=x_U, weight=weight, loss_func=loss_func,
                opt=opt, meter=meter, dataset=dataset, epoch=epoch, step=step)
            if args.sample_all_interval > 1 and not inside_only:
                dataset.update_inside(idx, ret['inside'].detach())

            unsafe = ~ret['verified']['safe']
            if not args.adv_only and unsafe.any():
                if (len(dataset) < args.max_num_domains
                    and (step <= args.split_end_steps
                            or ret['verified']['safe'].float().mean() > args.split_threshold)
                        and step >= args.split_start_steps
                        and epoch % args.split_interval == 0):
                    input_split(
                        dataset, unsafe=unsafe, x_L=x_L, x_U=x_U, weight=weight,
                        idx=idx, model=model, loss_func=loss_func, ret=ret,
                        step=step, args=args)
            meter.update('num_domains', len(dataset), step)

            lr_scheduler.step()

            if step % args.log_interval == 0 or i + 1 == num_batches:
                time_per_step = (time.time() - start_time) / (step - start_step)
                logger.info(f'Training step {step} ({time_per_step:.2f}s/step): '
                            f'{print_selected_meters(meter)}')
                if step % args.log_interval == 0:
                    write_tensorboard(writer, meter, step, inside_only=inside_only)
                    write_tensorboard(hf_writer, meter, step, inside_only=inside_only)
                    meter.reset()

            if (step + 1 == args.split_start_steps
                    or step % args.save_interval == 0
                    or step == args.split_end_steps):
                save_checkpoint(
                    model_ori, model, opt, dataset, loss_func,
                    epoch, step, args.dir, args.hf, hf_writer,
                    save_domains=args.save_domains)

        else:
            dataset.commit_split()

if __name__ == '__main__':
    main(args)

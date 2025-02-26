"""Utilities."""
import ast
import random
import os
import copy
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from auto_LiRPA import BoundedModule
from auto_LiRPA.utils import logger
from models import *  # pylint: disable=wildcard-import,unused-wildcard-import


def set_file_handler(logger, logdir):
    file_handler = logging.FileHandler(os.path.join(logdir, 'train.log'))
    file_handler.setFormatter(
        logging.Formatter('%(levelname)-8s %(asctime)-12s    %(message)s'))
    logger.addHandler(file_handler)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_checkpoint(args, load_path, model, dataset=None, loss_func=None, opt=None):
    if args.load_last:
        last_step = -1
        for filename in os.listdir(args.dir):
            if filename.endswith('.pt'):
                last_step = max(last_step, int(filename.split('.')[0]))
        if last_step != -1:
            load_path = os.path.join(args.dir, f'{last_step}.pt')
    checkpoint = torch.load(load_path)
    logger.info('Checkpoint loaded from %s', load_path)

    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    if opt is not None:
        opt.load_state_dict(checkpoint['optimizer'])

    if dataset is not None:
        if args.load_domains:
            path_domains = args.load_domains
        else:
            path_domains = '.'.join(load_path.split('.')[:-1]) + '_domains.pt'
            if not os.path.exists(path_domains):
                path_domains = None
        if path_domains:
            dataset.load_checkpoint(path_domains)
        else:
            logger.warning('Domains not loaded')
    if loss_func is not None:
        if 'rho' in checkpoint:
            loss_func.rho = checkpoint['rho']
    return checkpoint.get('epoch', 0), checkpoint['step']


def load_checkpoint_controller_only(args, load_path, model):
    checkpoint = torch.load(load_path)
    logger.info('Checkpoint loaded from %s', load_path)

    state_dict = checkpoint['state_dict']
    ret_load = model.load_state_dict(
        {k: v for k, v in state_dict.items()
         if k.startswith('controller.')},
        strict=False
    )
    logger.info('Partially loaded the state_dict: %s', ret_load)


def save_checkpoint(model, model_bound, opt, dataset, loss_func, epoch, step, dir,
                    hf, hf_writer=None, save_domains=True):
    sync_params(model, model_bound)
    save_path = os.path.join(dir, f'{step}.pt')
    torch.save({
        'state_dict': model.state_dict(),
        'optimizer': opt.state_dict(),
        'step': step,
        'epoch': epoch,
        'rho': loss_func.rho,
    }, save_path)
    logger.info('Checkpoint saved to %s', save_path)

    if hf:
        try:
            model.save_pretrained(dir)
            model.push_to_hub(hf, private=True)
            hf_writer.scheduler.trigger()
        except:
            pass

    if save_domains:
        path_domains = os.path.join(dir, 'domains.pt')
        dataset.save(path_domains)
        logger.info('Domains saved to %s', path_domains)


def get_optimizer(model, args):
    logger.info('Model parameters:')
    for param in model.named_parameters():
        logger.info('%s: shape: %s', param[0], param[1].shape)
    params = [p for p in model.parameters() if p.requires_grad]
    if args.opt == 'Adam':
        opt = torch.optim.Adam(
            params, lr=args.lr,
            betas=(args.adam_beta_1, args.adam_beta_2),
            eps=args.adam_eps, weight_decay=args.wd)
    elif args.opt == 'SGD':
        opt = torch.optim.SGD(params, lr=args.lr)
    else:
        raise NotImplementedError
    logger.info('Optimizer: %s', opt)
    return opt


def write_tensorboard(writer, meter, step, inside_only=False):
    if writer is None:
        return
    if inside_only:
        return
    for k in meter.sum_meter.keys():
        writer.add_scalar(k, meter.avg(k), step)


def print_selected_meters(meter):
    ret = ''
    ret += f'loss={meter.avg("main/loss"):.4f}'
    ret += f', unsafe={meter.avg("main/unsafe"):.2f}'
    ret += f', ver={meter.avg("main/ver_safe_in"):.4f}'
    ret += f', inside={meter.avg("main/inside"):.6f}'
    return ret


def sync_params(model_ori, model):
    """Update the state_dict of model_ori using the state_dict of model."""

    state_dict_loss = model.state_dict()
    state_dict = model_ori.state_dict()
    for name in state_dict_loss:
        v = state_dict_loss[name]
        assert name.endswith('.param')
        name_ori = model[name[:-6]].ori_name
        assert name_ori in state_dict
        state_dict[name_ori] = v
    model_ori.load_state_dict(state_dict)


@torch.no_grad()
def scale_grad_norm(parameters, norm_goal, norm_type=2.0) -> torch.Tensor:
    # Modified from https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
    # The major difference is that we not only clip the gradient norm
    # but also scale it when it is smaller than the threshold
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    norm_goal = float(norm_goal)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.0)
    norms = [torch.linalg.vector_norm(g, norm_type) for g in grads]
    total_norm = torch.linalg.vector_norm(
        torch.stack([norm for norm in norms]), norm_type
    )
    scale = norm_goal / (total_norm + 1e-6)
    for g in grads:
        g.mul_(scale)
    return total_norm

from collections import defaultdict
import os
import time
import torch
from torch.nn import functional as F
from auto_LiRPA import BoundedTensor, PerturbationLpNorm
from auto_LiRPA.bound_ops import *
from split_utils import bf_split


def get_bounds(model, bound_method=None, x=None, x_L=None, x_U=None, C=None,
               return_A=False, args=None):
    if bound_method is None:
        bound_method = args.bound_method
    if x_L is not None and x_U is not None:
        x = BoundedTensor(x_L, PerturbationLpNorm(x_L=x_L, x_U=x_U))
    if isinstance(x, torch.Tensor):
        x = (x,)
    if return_A:
        needed_A_dict = defaultdict(set)
        needed_A_dict[model.output_name[0]].add(model.input_name[0])
    else:
        needed_A_dict = None
    kwargs = {
        'C': C,
        'bound_lower': False,
        'return_A': return_A,
        'needed_A_dict': needed_A_dict
    }
    if bound_method == 'IBP':
        ret = model.compute_bounds(x=x, method='IBP', **kwargs)
    elif bound_method == 'CROWN-IBP':
        ret = model.compute_bounds(x=x, method='crown-ibp', **kwargs)
    elif bound_method == 'full-CROWN':
        ret = model.compute_bounds(x=x, method='CROWN', **kwargs)
    elif bound_method == 'alpha-CROWN':
        ret = model.compute_bounds(x=x, method='alpha-CROWN', **kwargs)
    elif bound_method == 'CROWN':
        def is_activation(node):
            return (isinstance(node, BoundRelu)
                    or isinstance(node, BoundLeakyRelu)
                    or isinstance(node, BoundSigmoid))

        ibp_nodes = []
        for node in model.nodes():
            if ((isinstance(node, BoundLinear) or isinstance(node, BoundAdd))
                    and len(node.output_name) > 0)  :
                relu_only_in_out = True
                for out_name in node.output_name:
                    if not is_activation(model[out_name]):
                        relu_only_in_out = False
                        break
                if relu_only_in_out:
                    ibp_nodes.append(node.name)
            elif (args.ibp_for_rx
                    and isinstance(node, BoundMatMul)
                    and len(node.output_name) == 2
                    and is_activation(model[node.output_name[0]])):
                ibp_nodes.append(node.name)

        for node in model.nodes():
            if isinstance(node, BoundLinear) and node.inputs[0].name == '/x_next':
                node.requires_input_bounds = [0]

        if args.more_crown_for_output_feedback:
            for node in model.nodes():
                if (node.perturbed and isinstance(node, BoundLinear)
                        and isinstance(node.inputs[0], BoundConcat)):
                    node.requires_input_bounds = [0]

        ret = model.compute_bounds(
            x=x, method='crown', ibp_nodes=ibp_nodes, **kwargs)

        if os.environ.get('AUTOLIRPA_DEBUG', 0):
            breakpoint()

    else:
        raise NotImplementedError(bound_method)

    if len(ret) == 2:
        ub = ret[1]
        uA = None
    else:
        ub, A = ret[1:]
        uA = A[model.output_name[0]][model.input_name[0]]['uA']
    if return_A:
        return ub, uA
    else:
        return ub


def get_verified_result(x_L, x_U, model=None, loss_func=None, bound_method=None,
                        args=None, meter=None):
    if bound_method is None:
        bound_method = args.bound_method

    x_M = (x_L + x_U) / 2
    ptb = PerturbationLpNorm(x_L=x_L, x_U=x_U)
    x = BoundedTensor(x_M, ptb)
    output_M = model(x) if loss_func.normalize else None

    ub = get_bounds(model, bound_method=bound_method, args=args)

    ret = loss_func(loss_term='verified', y=ub, y_M=output_M)
    ret.update({
        'ub': ub,
    })
    return ret

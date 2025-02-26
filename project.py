from utils import logger


def project_params(model_ori, verbose=False, use_abs=False):
    last_weight = None
    for param in model_ori.named_parameters():
        if 'lyapunov.' not in param[0]:
            continue
        if 'bias' in param[0]:
            if verbose:
                logger.info('Making %s non-positive', param[0])
            param[1].data = param[1].clamp(max=0)
        elif 'weight' in param[0]:
            last_weight = param
    if verbose:
        logger.info('Making %s non-negative', last_weight[0])
    # if use_abs:
    #     last_weight[1].data = last_weight[1].abs()
    # else:
    #     last_weight[1].data = last_weight[1].clamp(min=0)

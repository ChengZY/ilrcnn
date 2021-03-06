# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch


# TODO maybe push this to nn?
def smooth_l1_loss(input, target, beta=1. / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    # print('smooth_l1_loss.py | input size: {0}'.format(input.size()))
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()


def smooth_l1_loss_weight(input, target, beta=1. / 9, size_average=True, weight=None):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    # print('smooth_l1_loss.py | input size: {0}'.format(input.size()))
    # from ipdb import set_trace; set_trace()
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta) * weight[:, None]
    if size_average:
        return loss.mean()
    return loss.sum()
# -*- coding: utf-8 -*-

"""
Commonly used utils for the package
"""
import time
import random

import numpy as np
import torch
from torch import Tensor


def shuffle(t: Tensor) -> Tensor:
    """
    Args:
        t (Tensor): input tensor with t.dim() >= 1.

    Returns:
        Tensor - shallow copy of an input tensor randomly shuffled along the first (batch) axis
    """
    idx = torch.randperm(t.shape[0], device=t.device)
    return t[idx].view(t.size())


def interp1d(x: Tensor, y: Tensor, x_new: Tensor) -> Tensor:
    eps = torch.finfo(y.dtype).eps

    ind = torch.searchsorted(x.contiguous(), x_new.contiguous())

    ind = torch.clamp(ind - 1, 0, x.shape[0] - 2)

    slopes = (y[1:] - y[:-1]) / (eps + (x[1:] - x[:-1]))

    return y[ind] + slopes[ind] * (x_new - x[ind])


def cov(arr: Tensor, weights: Tensor = None, bias: bool = False):
    if weights is None:
        arr = arr - arr.mean(0, keepdim=True)
        factor = 1 / (arr.shape[0] - int(not bias))
        return factor * arr.T @ arr.conj()

    w_sum = weights.sum()

    if weights.dim() == 1:
        weights = weights[:, None]

    arr = arr - (arr * weights).sum(0, keepdims=True) / w_sum

    if bias:
        factor = 1 / w_sum
    else:
        factor = 1 / (w_sum - (weights ** 2).sum() / w_sum)

    return factor * (arr * weights).T @ arr.conj()


def set_random_seed(seed=None):
    if seed is None:
        seed = int((time.time() * 1e6) % 1e8)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

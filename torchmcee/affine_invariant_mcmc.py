# -*- coding: utf-8 -*-

"""
This module implements affine invariant algorithms described by Goodman & Weare (2010) (DOI: 10.2140/camcos.2010.5.65)
and hugely relied on emcee package (see https://emcee.readthedocs.io)
"""


from typing import Tuple

import torch
from torch import Tensor

from torchmcee.base import (
    State,
    SamplerStep,
    LOG_PROB_FUNC_TYPE,
)

from torchmcee.utils import shuffle


class AffineInvariantMove(SamplerStep):
    def __call__(self, state: State, log_prob_func: LOG_PROB_FUNC_TYPE, **kwargs) -> Tuple[State, Tensor]:
        return self._split_sample_step(state, log_prob_func)

    def _move(self, s, c) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def _split_sample_step(self, state: State, log_prob_func: LOG_PROB_FUNC_TYPE) -> Tuple[State, Tensor]:
        nwalkers, ndim = state.coords.shape
        device, dtype = state.device, state.dtype

        accepted = torch.zeros((nwalkers,), dtype=torch.bool, device=device)

        all_indices = torch.arange(nwalkers, device=device)

        split_num_indices = shuffle(all_indices % 2)

        for split_num in range(2):
            s1 = split_num_indices == split_num
            s, c = _split_walkers(split_num, split_num_indices, state.coords)
            q, factors = self._move(s, c)
            new_log_probs = log_prob_func(q)

            sampled_rands = torch.log(torch.rand(factors.shape[0], device=device, dtype=dtype))

            lnpdiff = factors + new_log_probs - state.log_prob[all_indices[s1]]

            accepted[s1] = lnpdiff > sampled_rands

            new_state = State(q, log_prob=new_log_probs)
            state = _update_state(state, new_state, accepted, s1)

        return state, accepted


class StretchMove(AffineInvariantMove):
    def __init__(self, a: float = 2.):
        self.a = a

    def _move(self, s, c):
        return stretch_move(s, c, self.a)


def _split_walkers(split_num: int, split_num_indices: Tensor, coords: Tensor):
    rest = (split_num + 1) % 2
    sets = [coords[split_num_indices == j] for j in range(2)]
    s = sets[split_num]
    c = sets[rest]
    return s, c


def _update_state(old_state, new_state, accepted, subset):
    m1 = subset & accepted
    m2 = accepted[subset]
    old_state.coords[m1] = new_state.coords[m2]
    old_state.log_prob[m1] = new_state.log_prob[m2]
    return old_state


def stretch_move(s: Tensor, c: Tensor, a: float) -> Tuple[Tensor, Tensor]:
    ns, nc, ndim = s.shape[0], c.shape[0], c.shape[1]

    zz = ((a - 1.0) * torch.rand(ns, device=s.device, dtype=s.dtype) + 1) ** 2.0 / a
    factors = (ndim - 1.0) * torch.log(zz)
    rint = torch.randint(nc, size=(ns,), device=s.device)

    return c[rint] - (c[rint] - s) * zz[:, None], factors

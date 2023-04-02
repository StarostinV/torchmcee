# -*- coding: utf-8 -*-

"""
This module implements affine invariant algorithms described by ...
"""

from typing import Tuple
from torch import Tensor

from torchmcee.base import (
    State,
    SamplerStep,
    LOG_PROB_FUNC_TYPE,
)


class ImportanceSampling(SamplerStep):
    def __call__(self, state: State, log_prob_func: LOG_PROB_FUNC_TYPE, **kwargs) -> Tuple[State, Tensor]:
        raise NotImplementedError

    def sample(self, state: State, log_prob_func: LOG_PROB_FUNC_TYPE, **kwargs):
        pass

    def update_proposals(self, state: State, log_prob_func: LOG_PROB_FUNC_TYPE, **kwargs):
        pass

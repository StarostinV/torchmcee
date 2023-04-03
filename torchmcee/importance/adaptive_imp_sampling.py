# -*- coding: utf-8 -*-

"""
This module implements basic functionality for adaptive importance sampling schemes
"""

from typing import Tuple

import torch
from torch import Tensor

from torchmcee.base import (
    State,
    SamplerStep,
    LOG_PROB_FUNC_TYPE,
)
from torchmcee.importance.kernels import Kernel, DeterministicKernelMixture
from torchmcee.mcmc.hamiltonian_mc import HMCStep


class ImportanceSampling(SamplerStep):
    def __init__(self, kernel: Kernel):
        self.kernel = kernel

    def __call__(self, state: State, log_prob_func: LOG_PROB_FUNC_TYPE, **kwargs) -> Tuple[State, Tensor]:
        self.update_proposals(state, log_prob_func, **kwargs)
        samples = self.sample(state.coords.shape[0])
        log_probs = log_prob_func(samples)
        finite_indices = torch.isfinite(log_probs)
        infinite_indices = ~finite_indices
        samples[infinite_indices] = state.coords[infinite_indices]
        log_probs[infinite_indices] = state.log_prob[infinite_indices]
        return State(samples, log_probs), finite_indices

    def sample(self, num_samples: int) -> Tensor:
        return self.kernel.sample(num_samples)

    def update_proposals(self, state: State, log_prob_func: LOG_PROB_FUNC_TYPE, **kwargs):
        pass


class LayeredAISStep(ImportanceSampling):
    def __init__(self, kernel: DeterministicKernelMixture, kernel_step: SamplerStep, stds: Tensor):
        super().__init__(kernel=kernel)
        self.kernel_step = kernel_step
        self.stds = stds

    def update_proposals(self, state: State, log_prob_func: LOG_PROB_FUNC_TYPE, **kwargs):
        state, accepted = self.kernel_step(state, log_prob_func, **kwargs)
        self.kernel.update_kernel(positions=state.coords, weights=state.log_prob.exp(), sizes=self.stds)


class HamiltonianAdaptiveImportanceSampling(LayeredAISStep):
    def __init__(self,
                 kernel: DeterministicKernelMixture,
                 stds: Tensor,
                 hmc_step: HMCStep = None,
                 num_steps_per_sample: int = 10,
                 step_size: float = 0.3
                 ):
        super().__init__(
            kernel=kernel,
            stds=stds,
            kernel_step=hmc_step or HMCStep(num_steps_per_sample=num_steps_per_sample, step_size=step_size),
        )

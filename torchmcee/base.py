from typing import Callable, Tuple

import torch
from torch import Tensor

from tqdm import trange

LOG_PROB_FUNC_TYPE = Callable[[Tensor], Tensor]


class State(object):
    __slots__ = ('coords', 'log_prob')

    def __init__(self, coords: Tensor, log_prob: Tensor):
        self.coords = coords
        self.log_prob = log_prob

    @property
    def device(self):
        return self.coords.device

    @property
    def dtype(self):
        return self.coords.dtype

    def __repr__(self):
        return "State({0}, log_prob={1})".format(
            self.coords, self.log_prob
        )


class SamplerStep(object):
    def __call__(self, state: State, log_prob_func: LOG_PROB_FUNC_TYPE, **kwargs) -> Tuple[State, Tensor]:
        raise NotImplementedError


class MCMCBackend(object):
    def __init__(self, num_walkers: int, device: torch.device = 'cpu', thin_by: int = 1):
        self.coords = []
        self.log_probs = []
        self.num_walkers = num_walkers
        self.accepted = torch.zeros(num_walkers).to(device)
        self.device = device
        self.thin_by = thin_by
        self._iteration = 0

    @property
    def accepted_fractions(self) -> Tensor:
        return (self.accepted / self.iteration if self.iteration else self.accepted).clone()

    @property
    def iteration(self) -> int:
        return self._iteration

    def save_state(self, state: State, accepted: Tensor):
        self.accepted += accepted.to(self.accepted)

        if self._iteration % self.thin_by == 0:
            self._save_state(state)

        self._iteration += 1

    def _save_state(self, state: State):
        self.coords.append(state.coords.to(self.device).clone())
        self.log_probs.append(state.log_prob.to(self.device).clone())

    def get_chain(self, flat: bool = False) -> Tensor:
        chain = torch.cat(self.coords, dim=0)
        if flat:
            chain = chain.flatten()
        return chain

    @property
    def chain_size(self):
        return sum(c.shape[0] for c in self.coords)

    def __repr__(self):
        return f'MCMCBackend(iterations={self.iteration}, num_walkers={self.num_walkers}, chain_size={self.chain_size},' \
               f'accepted_fractions={self.accepted_fractions})'


class MCMCSampler(object):
    def __init__(self, mcmc_step: SamplerStep, backend: MCMCBackend = None):
        self.mcmc_step = mcmc_step
        self.backend = backend

    def run(self,
            init_coords: Tensor,
            log_prob_func: LOG_PROB_FUNC_TYPE,
            num_steps: int,
            burn_in: int = 0,
            *,
            disable_tqdm: bool = False,
            progress_bar_cls=None,
            **kwargs
            ) -> None:

        state = State(init_coords, log_prob_func(init_coords))

        num_walkers, ndim = init_coords.shape

        backend = self.backend or MCMCBackend(num_walkers)
        self.backend = backend

        pbar = progress_bar_cls(num_steps) if progress_bar_cls else trange(num_steps, disable=disable_tqdm)

        for num_iteration in pbar:
            state, accepted = self.mcmc_step(state, log_prob_func, **kwargs)
            if num_iteration >= burn_in:
                backend.save_state(state, accepted)

    def get_chain(self, flat: bool = False):
        return self.backend.get_chain(flat=flat)

    @property
    def accepted_fractions(self) -> Tensor:
        return self.backend.accepted_fractions

    def __repr__(self):
        return f'MCMCSampler(backend={repr(self.backend)})'

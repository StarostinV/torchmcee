from typing import Tuple

import torch
from torch import Tensor

from torchmcee.base import (
    SamplerStep,
    LOG_PROB_FUNC_TYPE,
    State,
)


class HMCStep(SamplerStep):
    def __init__(self, num_steps_per_sample: int = 10, step_size: float = 0.3):
        self.num_steps_per_sample = num_steps_per_sample
        self.step_size = step_size

    def __call__(self, state: State, log_prob_func: LOG_PROB_FUNC_TYPE, **kwargs) -> Tuple[State, Tensor]:
        params, accepted = hmc_step(
            log_prob_func, state.coords,
            num_steps_per_sample=self.num_steps_per_sample,
            step_size=self.step_size,
        )
        state = State(params, log_prob_func(params))  # can accelerate by getting log_prob values from hmc step
        return state, accepted


def hmc_step(
        log_prob_func,
        params_init,
        num_steps_per_sample: int = 10,
        step_size: float = 0.1,
) -> Tuple[Tensor, Tensor]:
    params_init = torch.atleast_2d(params_init)

    params = params_init.clone()

    momentum = torch.randn_like(params)

    ham = hamiltonian(params, momentum, log_prob_func)

    params, momentum, finite_indices = leapfrog(
        params, momentum, log_prob_func,
        steps=num_steps_per_sample, step_size=step_size,
    )

    new_ham = torch.ones_like(ham) * float('inf')

    new_ham[finite_indices] = hamiltonian(
        params[finite_indices], momentum[finite_indices], log_prob_func
    )

    rho = torch.clamp_max(ham - new_ham, 0.)

    rejection_condition = rho < torch.log(torch.rand_like(rho))

    num_rejected = rejection_condition.sum().item()

    if num_rejected > 0:
        params[rejection_condition] = params_init[rejection_condition]

    accepted = ~rejection_condition

    return params, accepted


def hamiltonian(params, momentum, log_prob_func):
    return -log_prob_func(params) + 0.5 * torch.sum(momentum * momentum, dim=-1)


def leapfrog(params, momentum, log_prob_func, steps: int = 10, step_size: float = 0.1):
    params = params.clone()
    momentum = momentum.clone()

    grad, finite_indices = _get_batched_params_grad(params, log_prob_func)
    momentum[finite_indices] += 0.5 * step_size * grad

    for n in range(steps):
        if finite_indices.sum().item() == 0:
            break

        params = params.detach()
        params[finite_indices] += step_size * momentum[finite_indices]

        grad, finite_indices = _get_batched_params_grad(params, log_prob_func, finite_indices)

        assert finite_indices.sum().item() == grad.shape[
            0], f"shape mismatch: {finite_indices.sum().item()} != {grad.shape[0]}"

        momentum[finite_indices] += step_size * grad

    if finite_indices.sum().item():
        momentum[finite_indices] -= 0.5 * step_size * grad

    return params, momentum, finite_indices


def _get_batched_params_grad(p, func, finite_indices=None):
    if finite_indices is None:
        p = p.clone().requires_grad_()
        log_probs = func(p)
        finite_indices = torch.isfinite(log_probs)
        grad_indices = finite_indices
    else:
        log_probs = torch.zeros_like(p[..., -1])
        p = p[finite_indices]
        p = p.clone().requires_grad_()
        log_probs[finite_indices] = func(p)
        grad_indices = torch.isfinite(log_probs[finite_indices])
        finite_indices = torch.isfinite(log_probs) & finite_indices

    if finite_indices.sum().item():
        grad = torch.autograd.grad(log_probs[finite_indices].sum(), p)[0][grad_indices]
    else:
        grad = torch.empty(0, p.shape[-1], device=p.device, dtype=p.dtype)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return grad, finite_indices

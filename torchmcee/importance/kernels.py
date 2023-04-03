from typing import Union
import torch
from torch import Tensor

from math import log, pi


class Distribution(object):
    def log_prob(self, samples: Tensor) -> Tensor:
        """
        Calculate log probabilities of the samples
        Args:
            samples (Tensor): tensor of shape (num_samples, dim)

        Returns:
            Tensor of shape (num_samples, ) with log probabilities of the samples
        """
        raise NotImplementedError

    def sample(self, num_samples: int) -> Tensor:
        """
        Generate num_samples from the distribution.
        Args:
            num_samples (int): number of samples to generate

        Returns:
            Tensor - samples with shape (num_samples, dim)
        """
        raise NotImplementedError


class Kernel(Distribution):
    def update_kernel(self,
                      positions: Tensor,
                      weights: Tensor,
                      sizes: Union[float, Tensor],
                      ):
        """
        Updates kernel based on kernel centers, weights, and sizes.

        Args:
            positions (Tensor): (m, dim) tensor of kernel centers
            weights (Tensor): (m, ) tensor of kernel weights
            sizes (float or Tensor): Float number or (m, ) tensor of Gaussian kernel widths.
        """
        raise NotImplementedError


class BasicKernel(Kernel):
    _initialized = False

    def update_kernel(self,
                      positions: Tensor,
                      weights: Tensor,
                      sizes: Union[float, Tensor],
                      ):
        """
        Updates kernel based on kernel centers, weights, and sizes.

        Args:
            positions (Tensor): (m, dim) tensor of kernel centers
            weights (Tensor): (m, ) tensor of kernel weights
            sizes (float or Tensor): Float number or (m, ) tensor of kernel sizes
        """

        self._update_kernel(positions, weights, sizes)
        self._initialized = True

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def _check_initialized(self):
        if not self.is_initialized:
            raise RuntimeError("Kernel is not initialized.")

    def _update_kernel(self, positions: Tensor,
                       weights: Tensor,
                       sizes: Union[float, Tensor],
                       ):
        raise NotImplementedError


class DeterministicKernelMixture(BasicKernel):
    min_weights: float = 0
    max_batch: int = None

    def _kernel_log_prob(self, samples, positions, sizes, **kwargs):
        raise NotImplementedError

    def _update_kernel(self,
                       positions: Tensor,
                       weights: Tensor,
                       sizes: Union[float, Tensor],
                       ) -> None:
        """
        Updates kernel based on kernel centers, weights, and sizes.

        Args:
            positions (Tensor): (m, dim) tensor of kernel centers
            weights (Tensor): (m, ) tensor of kernel weights
            sizes (float or Tensor): Float number or (m, ) tensor of kernel widths.
        """

        assert positions.dim() == 2

        num_kernels, dim = positions.shape

        self.dim = dim

        if weights is None:
            weights = torch.ones_like(positions[..., 0])

        weights = weights / weights.sum()

        if isinstance(sizes, (float, int)):
            sizes = torch.ones_like(weights) * sizes
        else:
            assert sizes.shape == weights.shape

        indices = self._filter_small_weights(weights)

        weights, positions, sizes = weights[indices], positions[indices], sizes[indices]

        self.weights, self.positions, self.stds = weights, positions, sizes

    def _filter_small_weights(self, weights: Tensor):
        # remove zero weights
        return weights > self.min_weights

    def log_prob(self, samples: Tensor) -> Tensor:
        """
        Calculates log probabilities of samples for a weighted sum of kernels
        via the deterministic mixture (DM) scheme.

        Args:
            samples (Tensor): (n, dim) tensor of samples for which log_prob is evaluated

        Returns:
            Tensor: log probs of shape (n, ) for n samples.
        """

        self._check_initialized()

        num_samples, dim = samples.shape

        assert dim == self.dim

        log_prob = calc_dm_log_probs(
            samples, self.positions, self.stds, log_prob_func=self._kernel_log_prob,
            log_weights=torch.log(self.weights), max_batch=self.max_batch,
        )

        return log_prob


class GaussianMixtureKernel(DeterministicKernelMixture):
    def __init__(self,
                 positions: Tensor = None,
                 weights: Tensor = None,
                 sizes: Union[float, Tensor] = None,
                 min_weights: float = 0., max_batch: int = None):
        self.min_weights = min_weights
        self.max_batch = max_batch

        if positions is not None:
            self.update_kernel(positions, weights, sizes)

    def sample(self, num_samples: int) -> Tensor:
        """
        Args:
            num_samples (int): required number of samples

        Returns:
            Tensor: samples of shape (num_samples, dim)

        """
        self._check_initialized()

        weights = self.weights
        indices = torch.multinomial(weights, num_samples, replacement=True)
        samples = torch.randn(num_samples, self.dim, device=weights.device, dtype=weights.dtype)
        samples = samples * self.stds[indices] + self.positions[indices]
        return samples

    def _kernel_log_prob(self, samples, positions, sizes, **kwargs):
        return calc_log_gauss(samples, positions, sizes)


def calc_dm_log_probs(
        samples: Tensor,
        positions: Tensor,
        stds: Tensor,
        log_prob_func,
        log_weights: Tensor = None,
        max_batch: int = None,
        **kwargs,
):
    """

    Calculate log probabilities of the samples for the deterministic mixture of kernels
    characterized by positions, stds, log prob function, and weights.

    Args:
        samples:
        positions (Tensor): (m, dim) tensor of kernel centers
        stds (Tensor): (m, ) tensor of kernel widths.
        log_prob_func: function for calculating log probabilities of the kernel
        log_weights (Tensor): (m, ) tensor of kernel log weights
        max_batch (int): maximum batch for parallelized computation of log_prob (provided to avoid memory issues)
        **kwargs: additional arguments for the log_prob_func.

    Returns:
        Tensor with shape (m, ) of log probabilities of the samples.
    """

    max_batch = max_batch or samples.shape[0]

    log_prob = - torch.cat([
        log_prob_func(s, positions, stds, **kwargs)
        for s in torch.split(samples, max_batch, 0)
    ])

    if log_weights is not None:
        log_prob += log_weights[None]

    log_prob = torch.logsumexp(log_prob, dim=-1)

    return log_prob


def calc_log_gauss(samples, positions, stds):
    stds = 2 * stds ** 2
    dim = samples.shape[-1]
    log_z = 0.5 * dim * log(2 * pi)
    return torch.sum((samples[:, None] - positions[None]) ** 2 / stds[None], dim=-1) - log_z

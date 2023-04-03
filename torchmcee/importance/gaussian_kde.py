# -*- coding: utf-8 -*-

"""
Parallel Gaussian KDE implemented in PyTorch that relies on scipy implementation.
"""

from typing import Union

import numpy as np
import torch
from torch import Tensor

from torchmcee.utils import cov
from torchmcee.importance.kernels import DeterministicKernelMixture, calc_log_gauss


class GaussianKDE(DeterministicKernelMixture):
    def __init__(self,
                 kernel_centers: Tensor = None,
                 weights: Tensor = None,
                 bandwidth: Union[str, float] = 'scott',
                 batch_size: int = None,
                 ):
        self.batch_size = batch_size
        self._bandwidth = bandwidth

        if kernel_centers is not None:
            self.update_kernel(kernel_centers, weights, bandwidth)

    def _update_kernel(self, dataset: Tensor, weights: Tensor = None, kernel_size: float = None):
        kernel_size = kernel_size or self._bandwidth

        self.dataset = dataset
        (
            self.positions,
            self.weights,
            self.neff,
            self.bandwidth,
            self.covariance,
            self.inv_cov,
            self.whitening,
            self.log_norm
        ) = _init_gaussian_kde(dataset, weights, kernel_size)

    def _kernel_log_prob(self, samples, positions, sizes, **kwargs):
        return calc_log_gauss(samples, positions, sizes) - self.log_norm

    def log_prob(self, samples: Tensor) -> Tensor:
        self._check_initialized()

        return super().log_prob(samples @ self.whitening)

    def sample(self, num_samples: int) -> Tensor:
        self._check_initialized()

        return _resample_from_gaussian_kde(self.dataset, self.covariance, self.weights, num_samples)


def _resample_from_gaussian_kde(dataset: Tensor, covariance: Tensor, weights: Tensor, size: int):
    norm = torch.distributions.MultivariateNormal(
        torch.zeros(covariance.shape[0], device=covariance.device, dtype=covariance.dtype),
        covariance
    ).sample((size,))

    # the simplest workaround to the absence of torch.choice function (at least in some older versions)
    indices = np.random.choice(dataset.shape[0], size=size, p=weights.flatten().cpu().numpy())
    means = dataset[indices]

    return means + norm


def _init_gaussian_kde(dataset: Tensor, weights: Tensor = None, bandwidth: Union[str, float] = 'scott'):
    num_points, dim = dataset.shape

    if weights is None:
        weights = torch.ones(num_points, 1, device=dataset.device, dtype=dataset.dtype) / num_points
        neff = num_points
    else:
        if weights.dim() == 1:
            weights = weights.unsqueeze(1)

        weights = weights / weights.sum()
        neff = 1 / torch.sum(weights ** 2)

    if isinstance(bandwidth, str):
        bandwidth = _get_bandwidth_by_rule(bandwidth, neff, dim)

    covariance, inv_cov = _calculate_covariance(dataset, weights, bandwidth)
    whitening = torch.linalg.cholesky(inv_cov)
    scaled_params = dataset @ whitening

    log_norm = torch.log(whitening.diag().prod()).item()

    return (
        scaled_params,
        weights,
        neff,
        bandwidth,
        covariance,
        inv_cov,
        whitening,
        log_norm,
    )


def _calculate_covariance(params, weights, factor):
    data_covariance = cov(params, weights)
    data_inv_cov = torch.linalg.inv(data_covariance)
    covariance = data_covariance * factor ** 2
    inv_cov = data_inv_cov / factor ** 2
    return covariance, inv_cov


def _get_bandwidth_by_rule(bandwidth: str, neff: int, dim: int) -> float:
    if bandwidth == 'scott':
        bandwidth = neff ** (-1. / (dim + 4))
    elif bandwidth == 'silver':
        bandwidth = (neff * (dim + 2) / 4.) ** (-1. / (dim + 4))
    else:
        raise ValueError(f'Unknown bandwidth rule {bandwidth}')

    return bandwidth

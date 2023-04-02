from typing import Union, Tuple

import numpy as np
import torch
from torch import Tensor

from torchmcee.utils import interp1d, cov


@torch.no_grad()
def estimate_density_on_grid(
        params: Tensor,
        weights: Tensor = None,
        grid_sizes: Union[Tuple[int, ...], int] = 100,
        batch_size: int = 512,
        bandwidth: Union[str, float] = 'scott',
        cut: float = 3.,
        param_ranges: Tuple[Tuple[float, float], ...] = None,
        normalize: bool = True,
):
    num_points, dim = params.shape

    if isinstance(grid_sizes, int):
        grid_sizes = [grid_sizes] * dim

    (
        scaled_dataset,
        weights,
        neff,
        bandwidth,
        covariance,
        inv_cov,
        whitening,
        norm,
    ) = _init_gaussian_kde(params, weights, bandwidth)

    grid = _get_grid_for_kde(params, grid_sizes, cut=cut, covariance=covariance, param_ranges=param_ranges)

    density = _evaluate_gaussian_kde(
        scaled_dataset, grid, weights, whitening, norm, batch_size
    )

    density = density.view(*grid_sizes)
    grid = grid.view(*grid_sizes, dim)

    if normalize:
        density = density / density.sum()

    return density, grid


@torch.no_grad()
def gaussian_kde(dataset, grid, weights: Tensor = None, bandwidth: Union[str, float] = 'scott', batch_size: int = None):
    (
        scaled_dataset,
        weights,
        neff,
        bandwidth,
        covariance,
        inv_cov,
        whitening,
        norm,
    ) = _init_gaussian_kde(dataset, weights, bandwidth)

    res = _evaluate_gaussian_kde(
        scaled_dataset, grid, weights, whitening, norm, batch_size
    )

    return res


@torch.no_grad()
def resample_from_kde(
        dataset, size: int,
        weights: Tensor = None,
        bandwidth: Union[str, float] = 'scott',
):
    (
        scaled_dataset,
        weights,
        neff,
        bandwidth,
        covariance,
        inv_cov,
        whitening,
        norm,
    ) = _init_gaussian_kde(dataset, weights, bandwidth)

    return _resample_from_gaussian_kde(dataset, covariance, weights, size)


class GaussianKDE(object):
    def __init__(self, dataset, weights: Tensor = None, bandwidth: Union[str, float] = 'scott', batch_size: int = None):
        self.num_points, self.dim = dataset.shape
        self.device, self.dtype = dataset.device, dataset.dtype
        self.dataset = dataset
        self.batch_size = batch_size

        (
            self.scaled_dataset,
            self.weights,
            self.neff,
            self.bandwidth,
            self.covariance,
            self.inv_cov,
            self.whitening,
            self.norm
        ) = _init_gaussian_kde(dataset, weights, bandwidth)

    def evaluate(self, grid: Tensor, batch_size: int = None):
        batch_size = batch_size or self.batch_size
        res = _evaluate_gaussian_kde(
            self.scaled_dataset, grid, self.weights, self.whitening, self.norm, batch_size
        )
        return res

    __call__ = evaluate

    def resample(self, size: int):
        return _resample_from_gaussian_kde(self.dataset, self.covariance, self.weights, size)


@torch.no_grad()
def _resample_from_gaussian_kde(dataset: Tensor, covariance: Tensor, weights: Tensor, size: int):
    norm = torch.distributions.MultivariateNormal(
        torch.zeros(covariance.shape[0], device=covariance.device, dtype=covariance.dtype),
        covariance
    ).sample((size,))

    # the simplest workaround to the absence of torch.choice function (at least in some older versions)
    indices = np.random.choice(dataset.shape[0], size=size, p=weights.flatten().cpu().numpy())
    means = dataset[indices]

    return means + norm


@torch.no_grad()
def _evaluate_gaussian_kde(
        scaled_dataset: Tensor,
        grid: Tensor,
        weights: Tensor,
        whitening: Tensor,
        norm: float,
        batch_size: int = None
):
    scaled_grid = grid @ whitening

    if not batch_size or grid.shape[0] <= batch_size:
        res = _calc_kde_gauss(scaled_dataset, scaled_grid, weights)
    else:
        res = torch.cat([
            _calc_kde_gauss(scaled_dataset, y_batch, weights)
            for y_batch in scaled_grid.split(batch_size)
        ])

    return res * norm


@torch.no_grad()
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

    norm = (2 * np.pi) ** (- dim / 2) * whitening.diag().prod()

    return (
        scaled_params,
        weights,
        neff,
        bandwidth,
        covariance,
        inv_cov,
        whitening,
        norm,
    )


def get_percentile_contours(res, levels: tuple = None, num: int = 1000):
    levels = levels if levels is not None else np.arange(1, 10) / 10

    levels = torch.tensor(levels, device=res.device, dtype=res.dtype)

    ts = torch.linspace(res.max(), 0, num, device=res.device, dtype=res.dtype)

    integral = ((res >= ts[:, None, None]) * res).sum(dim=(1, 2))

    t_contours = torch.flip(interp1d(integral, ts, levels), (-1,))

    return t_contours


def _get_grid_for_kde(
        params: Tensor,
        grid_sizes: Tuple[int, ...],
        cut: float = 3.,
        covariance: Tensor = None,
        param_ranges: Tuple[Tuple[float, float], ...] = None
):
    dim = len(grid_sizes)

    if not param_ranges:

        assert covariance is not None

        if dim == 1:
            bw = covariance.sqrt().flatten()
        else:
            bw = torch.diag(covariance).sqrt().flatten()

        pads = (bw * cut).tolist()

        x_mins, x_maxs = params.min(dim=0).values, params.max(dim=0).values

        param_ranges = ((x_min - pad, x_max + pad) for x_min, x_max, pad in zip(x_mins, x_maxs, pads))
    else:
        assert len(param_ranges) == dim

    grid_axes = [
        torch.linspace(x_min, x_max, grid_size, device=params.device, dtype=params.dtype)
        for (x_min, x_max), grid_size in zip(param_ranges, grid_sizes)
    ]

    grid = torch.stack(torch.meshgrid(*grid_axes), dim=-1)

    grid = grid.flatten(0, 1)

    if len(grid.shape) == 1:
        grid = grid[:, None]

    return grid


def _calc_kde_gauss(scaled_params, scaled_grid, weights: Tensor):
    return (torch.exp(- ((scaled_params[:, None] - scaled_grid[None]) ** 2).sum(-1) / 2) * weights).sum(0)


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

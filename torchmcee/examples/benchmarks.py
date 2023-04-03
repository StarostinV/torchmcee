from torch import Tensor


def banana_log_prob(samples: Tensor, sigma: float = 1., b: float = 3.):
    num_samples, dim = samples.shape
    assert dim >= 2
    x1, x2, x_rest = samples[..., 0], samples[..., 1], samples[..., 2:]
    s2 = 2 * sigma ** 2
    x12 = x1 ** 2
    return -x12 / s2 - (x2 + b * (x12 - s2 / 2)) ** 2 / s2 - (x_rest ** 2).sum(-1) / s2

import torch

from torchmcee import MCMCBackend

try:
    import matplotlib.pyplot as plt
    from matplotlib import collections as mpt_c
except ImportError:
    raise ImportError("This example requires matplotlib for plotting the results.")


def plot_mcmc_chains(backend: MCMCBackend, max_num_chains: int = 5, cmap: str = 'plasma'):
    """
    Plots 2 first dims of first n chains (n <= max_num_chains).

    Args:
        backend (MCMCBackend): backend with sampled mcmc chains
        max_num_chains (int): max number of chains to plot. 5 by default
        cmap (str): colormap name

    """
    num_chains = min(max_num_chains, backend.num_walkers)
    chains = [
        torch.stack([c[i][..., :2] for c in backend.coords])
        for i in range(num_chains)
    ]

    cmap = plt.get_cmap(cmap, num_chains + 1)

    for i, chain in enumerate(chains):
        lines = torch.stack([chain[1:], chain[:-1]], 0).cpu().tolist()

        line_c = mpt_c.LineCollection(
            lines, linewidths=1, colors=cmap(i + 1),
        )
        plt.scatter(*chain.T, color=cmap(i + 1), s=2)
        plt.gca().add_collection(line_c)
    plt.show()

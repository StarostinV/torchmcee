# _torchmcee_

## Monte Carlo sampling methods implemented in PyTorch with parallelization.

Currently supported samplers:

- Hamiltonian Monte Carlo (inspired by __hamiltorch__ package and improved by providing parallel computation)

- Affine-Invariant Markov Chain Monte Carlo (inspired by numpy implementation in __emcee__ package)

- Hamiltonian Adaptive Importance Sampling

Additionally, torchmcee provides Gaussian KDE implemented in PyTorch that follows the __scipy__ implementation. 
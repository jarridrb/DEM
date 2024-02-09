import math
from typing import Dict

import torch
from torch.distributions import constraints


class Prior:
    def __init__(self, dim, scale, device="cpu"):
        self.dim = dim
        self.scale = scale
        self.dist = torch.distributions.MultivariateNormal(
            torch.zeros(dim, device=device),
            torch.eye(dim, device=device) * (scale**2),
        )

    def log_prob(self, x):
        return self.dist.log_prob(x)

    def sample(self, n_samples):
        return self.dist.sample((n_samples,))


class MeanFreePrior(torch.distributions.Distribution):
    arg_constraints: Dict[str, constraints.Constraint] = {}

    def __init__(self, n_particles, spatial_dim, scale, device="cpu"):
        super().__init__()
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self.dim = n_particles * spatial_dim
        self.scale = scale
        self.device = device

    def log_prob(self, x):
        x = x.reshape(-1, self.n_particles, self.spatial_dim)
        N, D = x.shape[-2:]

        # r is invariant to a basis change in the relevant hyperplane.
        r2 = torch.sum(x**2, dim=(-1, -2)) / self.scale**2

        # The relevant hyperplane is (N-1) * D dimensional.
        degrees_of_freedom = (N - 1) * D

        # Normalizing constant and logpx are computed:
        log_normalizing_constant = (
            -0.5 * degrees_of_freedom * math.log(2 * torch.pi * self.scale**2)
        )
        log_px = -0.5 * r2 + log_normalizing_constant
        return log_px

    def sample(self, n_samples):
        samples = torch.randn(n_samples, self.dim, device=self.device) * self.scale
        samples = samples.reshape(-1, self.n_particles, self.spatial_dim)
        samples = samples - samples.mean(-2, keepdims=True)
        return samples.reshape(-1, self.n_particles * self.spatial_dim)

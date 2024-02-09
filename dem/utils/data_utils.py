import torch


def remove_mean(samples, n_particles, n_dimensions):
    """Makes a configuration of many particle system mean-free.

    Parameters
    ----------
    samples : torch.Tensor
        Positions of n_particles in n_dimensions.

    Returns
    -------
    samples : torch.Tensor
        Mean-free positions of n_particles in n_dimensions.
    """
    shape = samples.shape
    if isinstance(samples, torch.Tensor):
        samples = samples.view(-1, n_particles, n_dimensions)
        samples = samples - torch.mean(samples, dim=1, keepdim=True)
        samples = samples.view(*shape)
    else:
        samples = samples.reshape(-1, n_particles, n_dimensions)
        samples = samples - samples.mean(axis=1, keepdims=True)
        samples = samples.reshape(*shape)
    return samples


def interatomic_dist(samples):
    n_particles = samples.shape[-2]
    # Compute the pairwise differences and distances
    distances = samples[:, None, :, :] - samples[:, :, None, :]
    distances = distances[:, torch.triu(torch.ones((n_particles, n_particles)), diagonal=1) == 1]

    dist = torch.linalg.norm(distances, dim=-1)

    return dist

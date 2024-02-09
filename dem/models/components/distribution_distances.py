import math
from typing import Union

import numpy as np
import torch

from .mmd import linear_mmd2, mix_rbf_mmd2, poly_mmd2
from .optimal_transport import wasserstein


def compute_distances(pred, true):
    """Computes distances between vectors."""
    mse = torch.nn.functional.mse_loss(pred, true).item()
    me = math.sqrt(mse)
    mae = torch.mean(torch.abs(pred - true)).item()
    return mse, me, mae


def compute_distribution_distances(
    pred: torch.Tensor, true: Union[torch.Tensor, list], energy_function
):
    """computes distances between distributions.
    pred: [batch, times, dims] tensor
    true: [batch, times, dims] tensor or list[batch[i], dims] of length times

    This handles jagged times as a list of tensors.
    """
    NAMES = [
        "1-Wasserstein",
        "2-Wasserstein",
        "Linear_MMD",
        "Poly_MMD",
        "RBF_MMD",
        "Mean_MSE",
        "Mean_L2",
        "Mean_L1",
        "Median_MSE",
        "Median_L2",
        "Median_L1",
        "Eq-EMD2",
    ]
    is_jagged = isinstance(true, list)
    pred_is_jagged = isinstance(pred, list)
    dists = []
    to_return = []
    names = []
    filtered_names = [name for name in NAMES if not is_jagged or not name.endswith("MMD")]
    ts = len(pred) if pred_is_jagged else pred.shape[1]
    for t in np.arange(ts):
        if pred_is_jagged:
            a = pred[t]
        else:
            a = pred[:, t, :]
        if is_jagged:
            b = true[t]
        else:
            b = true[:, t, :]
        w1 = wasserstein(a, b, power=1)
        w2 = wasserstein(a, b, power=2)

        if energy_function.is_molecule:
            eq_emd2 = eot(
                a.reshape(-1, energy_function.n_particles, energy_function.n_spatial_dim).cpu(),
                b.reshape(-1, energy_function.n_particles, energy_function.n_spatial_dim).cpu(),
            )

        if not pred_is_jagged and not is_jagged:
            mmd_linear = linear_mmd2(a, b).item()
            mmd_poly = poly_mmd2(a, b, d=2, alpha=1.0, c=2.0).item()
            mmd_rbf = mix_rbf_mmd2(a, b, sigma_list=[0.01, 0.1, 1, 10, 100]).item()
        mean_dists = compute_distances(torch.mean(a, dim=0), torch.mean(b, dim=0))
        median_dists = compute_distances(torch.median(a, dim=0)[0], torch.median(b, dim=0)[0])
        if pred_is_jagged or is_jagged:
            dists.append((w1, w2, *mean_dists, *median_dists))
        else:
            if energy_function.is_molecule:
                dists.append(
                    (w1, w2, mmd_linear, mmd_poly, mmd_rbf, *mean_dists, *median_dists, eq_emd2)
                )
            else:
                dists.append((w1, w2, mmd_linear, mmd_poly, mmd_rbf, *mean_dists, *median_dists))
        # For multipoint datasets add timepoint specific distances
        if ts > 1:
            names.extend([f"t{t+1}/{name}" for name in filtered_names])
            to_return.extend(dists[-1])

    to_return.extend(np.array(dists).mean(axis=0))
    names.extend(filtered_names)
    return names, to_return


import ot as pot
from scipy.optimize import linear_sum_assignment


def find_rigid_alignment(A, B):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    2-D or 3-D registration with known correspondences.
    Registration occurs in the zero centered coordinate system, and then
    must be transported back.
        Args:
        -    A: Torch tensor of shape (N,D) -- Point Cloud to Align (source)
        -    B: Torch tensor of shape (N,D) -- Reference Point Cloud (target)
        Returns:
        -    R: optimal rotation
        -    t: optimal translation
    Test on rotation + translation and on rotation + translation + reflection
        >>> A = torch.tensor([[1., 1.], [2., 2.], [1.5, 3.]], dtype=torch.float)
        >>> R0 = torch.tensor([[np.cos(60), -np.sin(60)], [np.sin(60), np.cos(60)]], dtype=torch.float)
        >>> B = (R0.mm(A.T)).T
        >>> t0 = torch.tensor([3., 3.])
        >>> B += t0
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
        >>> B *= torch.tensor([-1., 1.])
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
    """
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.mm(B_c)
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V.mm(U.T)
    # Translation vector
    t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
    t = t.T
    return R, t.squeeze()


def ot(x0, x1):
    dists = torch.cdist(x0, x1)
    _, col_ind = linear_sum_assignment(dists)
    x1 = x1[col_ind]
    return x1


def eot(x0, x1):
    M = []
    for i in range(len(x0)):
        reordered = []
        for j in range(len(x1)):
            x1_reordered = ot(x0[i], x1[j])
            reordered.append(x1_reordered)
        reordered = torch.stack(reordered)
        R, t = torch.vmap(find_rigid_alignment)(x0[i][None].repeat(len(x1), 1, 1), reordered)
        superimposed = torch.matmul(reordered, R)
        M.append(torch.cdist(x0[i].reshape(1, -1), superimposed.reshape(len(x1), -1)))
    M = torch.stack(M).squeeze()
    return pot.emd2(M=M, a=torch.ones(len(x0)) / len(x0), b=torch.ones(len(x1)) / len(x1))

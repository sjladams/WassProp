"""
Unscented transform (UT) baseline: the standard symmetric sigma-point method of
Julier & Uhlmann (SPIE 1997; Proc. IEEE 92(3), 2004), in its scaled form
(Julier 2002; Wan & van der Merwe 2000).

This is a *moment-propagation* method with Gaussian closure: each step draws
2n+1 sigma points from the current Gaussian, pushes them through the dynamics,
and refits a Gaussian to the weighted empirical moments of the images. It is
therefore not an approximation of the pushforward law in W2 -- it only tracks
its first two moments -- and exists here purely as the citable baseline to
compare the formal propagation of `propagation.py` against.

Two modelling choices worth flagging:

* Non-additive noise is handled by the *augmented* UT (Julier & Uhlmann 2004):
  the state is stacked with the noise, so n = num_state_dims + num_noise_dims
  and the step costs 2(d+q)+1 evaluations of the dynamics.
* The original 1997 tuning kappa = 3 - n makes the centre weight
  w_0 = kappa / (n + kappa) negative whenever n > 3, so the sigma set is a
  signed measure rather than a distribution, and the refitted covariance is no
  longer guaranteed positive semi-definite. `kappa=None` reproduces that
  original tuning (and warns); `kappa=0.` gives the non-negative-weight
  variant.
"""

import warnings
from typing import Optional, Tuple, Union

import torch

import discretize_distributions.distributions as dd_dists

from wass_prop.dynamics import StochasticDynamics, AdditiveNoiseDynamics
from wass_prop.propagation import DistPath

__all__ = [
    'unscented_weights',
    'unscented_points',
    'unscented_moments',
    'single_step_unscented',
    'multi_step_unscented',
]

Dist = Union[dd_dists.MultivariateNormal, dd_dists.MixtureMultivariateNormal, dd_dists.CategoricalFloat]

TOL = 1e-8


def unscented_weights(
        num_dims: int,
        alpha: float = 1.,
        beta: float = 2.,
        kappa: Optional[float] = None,
        warn_negative: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Weights of the scaled symmetric UT on `num_dims` dimensions.

    alpha: spread of the sigma points around the mean (alpha=1 recovers the unscaled UT).
    beta:  incorporates prior knowledge on the distribution (beta=2 is optimal for Gaussians).
    kappa: secondary scaling; None selects the original 3 - num_dims of Julier & Uhlmann (1997),
        which yields a negative centre weight for num_dims > 3.

    Returns (w_mean, w_cov, lambda_), with both weight vectors of shape (2 * num_dims + 1,).
    """
    if kappa is None:
        kappa = 3. - num_dims

    lambda_ = alpha ** 2 * (num_dims + kappa) - num_dims
    if num_dims + lambda_ == 0.:
        raise ValueError(f"degenerate UT scaling: num_dims + lambda = 0 for {alpha=}, {kappa=}")

    w_mean = torch.full((2 * num_dims + 1,), 0.5 / (num_dims + lambda_))
    w_mean[0] = lambda_ / (num_dims + lambda_)
    w_cov = w_mean.clone()
    w_cov[0] = w_mean[0] + (1. - alpha ** 2 + beta)

    if warn_negative and w_mean[0] < 0.:
        warnings.warn(
            f"UT centre weight is negative (w_0 = {w_mean[0]:.4f}) for {num_dims=}, {alpha=}, {kappa=}: "
            f"the sigma set is a signed measure and the refitted covariance may lose positive "
            f"semi-definiteness. Pass kappa=0. for non-negative weights.",
            RuntimeWarning,
        )

    return w_mean, w_cov, lambda_


def _symmetric_sqrt(mat: torch.Tensor, scale: float) -> torch.Tensor:
    """Symmetric square root of `scale * mat`, for a symmetric positive semi-definite `mat`."""
    eigvals, eigvecs = torch.linalg.eigh(0.5 * (mat + mat.swapdims(-1, -2)))
    eigvals = scale * eigvals
    if (eigvals < -TOL).any():
        raise ValueError(
            f"cannot take the square root of an indefinite matrix (min eigenvalue "
            f"{eigvals.min().item():.3e}); with a negative UT weight the refitted covariance is "
            f"not guaranteed positive semi-definite."
        )
    return torch.einsum('...ij,...j,...kj->...ik', eigvecs, eigvals.clamp_min(0.).sqrt(), eigvecs)


def unscented_points(
        mean: torch.Tensor,
        covariance_matrix: torch.Tensor,
        alpha: float = 1.,
        beta: float = 2.,
        kappa: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Symmetric sigma points of a (batch of) Gaussian(s).

    mean: (..., n), covariance_matrix: (..., n, n).
    Returns (locs, w_mean, w_cov) with locs of shape (..., 2n+1, n).
    """
    num_dims = mean.size(-1)
    w_mean, w_cov, lambda_ = unscented_weights(num_dims, alpha=alpha, beta=beta, kappa=kappa)

    offsets = _symmetric_sqrt(covariance_matrix, scale=num_dims + lambda_)  # (..., n, n)
    offsets = offsets.swapdims(-1, -2)  # rows are the +/- offsets
    locs = torch.cat((
        torch.zeros_like(offsets[..., :1, :]),
        offsets,
        -offsets,
    ), dim=-2) + mean.unsqueeze(-2)

    return locs, w_mean.to(locs), w_cov.to(locs)


def unscented_moments(
        locs: torch.Tensor,
        w_mean: torch.Tensor,
        w_cov: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Weighted mean (..., m) and covariance (..., m, m) of sigma points `locs` of shape (..., 2n+1, m)."""
    mean = torch.einsum('k,...ki->...i', w_mean, locs)
    dev = locs - mean.unsqueeze(-2)
    cov = torch.einsum('k,...ki,...kj->...ij', w_cov, dev, dev)
    return mean, 0.5 * (cov + cov.swapdims(-1, -2))


def _mean_and_covariance(dist: Dist) -> Tuple[torch.Tensor, torch.Tensor]:
    """First two moments of `dist`; mixtures are collapsed by the law of total covariance."""
    if isinstance(dist, dd_dists.MultivariateNormal):
        return dist.loc, dist.covariance_matrix
    elif isinstance(dist, dd_dists.MixtureMultivariateNormal):
        probs = dist.mixture_distribution.probs
        locs = dist.component_distribution.loc
        covs = dist.component_distribution.covariance_matrix
        mean = torch.einsum('k,...ki->...i', probs, locs)
        dev = locs - mean.unsqueeze(-2)
        cov = torch.einsum('k,...kij->...ij', probs, covs) \
            + torch.einsum('k,...ki,...kj->...ij', probs, dev, dev)
        return mean, cov
    elif isinstance(dist, dd_dists.CategoricalFloat):
        mean = torch.einsum('k,ki->i', dist.probs, dist.locs)
        dev = dist.locs - mean
        return mean, torch.einsum('k,ki,kj->ij', dist.probs, dev, dev)
    else:
        raise TypeError(f"cannot extract moments from {type(dist)}")


def _block_diag(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Batched block-diagonal stacking of (..., m, m) and (..., n, n) into (..., m+n, m+n)."""
    batch_shape = torch.broadcast_shapes(left.shape[:-2], right.shape[:-2])
    left = left.expand(batch_shape + left.shape[-2:])
    right = right.expand(batch_shape + right.shape[-2:])
    upper = torch.cat((left, left.new_zeros(batch_shape + (left.size(-2), right.size(-1)))), dim=-1)
    lower = torch.cat((right.new_zeros(batch_shape + (right.size(-2), left.size(-1))), right), dim=-1)
    return torch.cat((upper, lower), dim=-2)


def _push_forward(f: torch.nn.Module, locs: torch.Tensor) -> torch.Tensor:
    """Evaluate `f` on sigma points of shape (..., 2n+1, n), preserving the leading shape."""
    flat = locs.reshape(-1, locs.size(-1))
    return f(flat).reshape(*locs.shape[:-1], -1)


@torch.no_grad()
def single_step_unscented(
        dynamics: StochasticDynamics,
        q: Dist,
        noise: Dist,
        use_additive_noise: bool = True,
        alpha: float = 1.,
        beta: float = 2.,
        kappa: Optional[float] = None,
) -> dd_dists.MultivariateNormal:
    """
    One UT prediction step: sigma points of `q` (augmented with `noise` unless the noise enters
    additively) are pushed through `dynamics` and a Gaussian is refitted to their weighted moments.
    """
    state_mean, state_cov = _mean_and_covariance(q)
    noise_mean, noise_cov = _mean_and_covariance(noise)

    if isinstance(dynamics, AdditiveNoiseDynamics) and use_additive_noise:
        state_locs, w_mean, w_cov = unscented_points(state_mean, state_cov, alpha=alpha, beta=beta, kappa=kappa)
        mean, cov = unscented_moments(_push_forward(dynamics.state_dynamics, state_locs), w_mean, w_cov)

        noise_locs, w_mean_noise, w_cov_noise = unscented_points(
            noise_mean, noise_cov, alpha=alpha, beta=beta, kappa=kappa)
        noise_mean, noise_cov = unscented_moments(
            _push_forward(dynamics.noise_dynamics, noise_locs), w_mean_noise, w_cov_noise)

        mean, cov = mean + noise_mean, cov + noise_cov
    else:
        augmented_mean = torch.cat((state_mean, noise_mean), dim=-1)
        augmented_cov = _block_diag(state_cov, noise_cov)
        locs, w_mean, w_cov = unscented_points(augmented_mean, augmented_cov, alpha=alpha, beta=beta, kappa=kappa)
        mean, cov = unscented_moments(_push_forward(dynamics, locs), w_mean, w_cov)

    return dd_dists.MultivariateNormal(loc=mean, covariance_matrix=cov)


def multi_step_unscented(
        dynamics: StochasticDynamics,
        q: Dist,
        noise: Dist,
        num_time_steps: int,
        use_additive_noise: bool = True,
        alpha: float = 1.,
        beta: float = 2.,
        kappa: Optional[float] = None,
        print_progress: bool = True,
) -> DistPath:
    """Iterates `single_step_unscented`, threading the refitted Gaussian through a `DistPath`."""
    path = DistPath()
    path.append(-1, q)

    with warnings.catch_warnings():
        warnings.simplefilter('once', RuntimeWarning)  # the negative-weight warning is per-step
        for k in range(num_time_steps):
            path.append(k, single_step_unscented(
                dynamics=dynamics,
                q=path.at(k - 1),
                noise=noise,
                use_additive_noise=use_additive_noise,
                alpha=alpha,
                beta=beta,
                kappa=kappa,
            ))
            if print_progress:
                print(f"unscented step {k + 1}/{num_time_steps}")

    return path

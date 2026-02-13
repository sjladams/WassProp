# WassProp — Formal Uncertainty Propagation in Wasserstein Distance

WassProp is a PyTorch toolkit for **formal uncertainty propagation** through (stochastic) **dynamical systems**. It represents the uncertain state of a stochastic system by means of **Wasserstein ambiguity balls**  B<sub>θ</sub>(P) = { Q : W₂(Q, P) ≤ θ }, where $P$ denotes a reference distribution and $\theta$ an upper bound on the 2-Wasserstein distance deviation from that reference. 

The algorithmic backbone and theoretical guarantees were first proposed in  [Figueiredo *et al.* (2025)](https://arxiv.org/pdf/2506.08689) and extended to the additive-noise setting in [Adams *et al.* (2025)](https://arxiv.org/pdf/2505.11219). 

The framework builds on the [**discretize_distributions**](https://github.com/sjladams/discretize_distributions) package for quantization (i.e., discretization) of continuous distributions with Wasserstein guarantees, and employs a modified version of [**bound_propagation**](https://github.com/Zinoex/bound_propagation)  
to compute local Lipschitz constants.

For a hands-on introduction to the theoretical principles, see [`demo_cdc_workshop.ipynb`](experiments/demo_cdc_workshop.ipynb) notebook.

## Installation
You can install the package directly from this repository:

```bash
pip install git+https://github.com/sjladams/WassProp
```
*(A PyPI release will follow soon.)*

## Quick start
## Quick start

```python
    import torch
    import discretize_distributions.distributions as dd
    from wass_prop import AmbiguityBall, multi_step, dynamics as dyn

    # 1) Define dynamics: x_{k+1} = A x_k + w_k
    A = torch.tensor([[0.8, 0.0],
                    [0.0, 0.95]])  # contractive example
    state_dyn = dyn.LinearDynamics(weight=A)
    f = dyn.AdditiveNoiseDynamics(state_dynamics=state_dyn)  # additive noise model

    # 2) Define ambiguity balls for initial state and noise
    q0_center = dd.MultivariateNormal(
        loc=torch.tensor([0.0, 0.0]),
        covariance_matrix=torch.diag(torch.tensor([1e-3, 1e-3]))
    )
    noise_center = dd.MultivariateNormal(
        loc=torch.tensor([0.0, 0.0]),
        covariance_matrix=torch.diag(torch.tensor([1e-4, 1e-4]))
    )

    q0    = AmbiguityBall(center=q0_center,    radius=0.01)
    noise = AmbiguityBall(center=noise_center, radius=0.01)

    # 3) Propagate K steps with N quantization points
    path = multi_step(
        dynamics=f,
        q=q0,
        noise=noise,
        num_time_steps=20,           # horizon K
        num_locs=100,                # quantization size for centers
        use_lagrangian_duality=True, # tighter radius updates (default True)
    )

    # 4) Access ambiguity ball at step k
    k = 10
    ball_k   = path.at(k)      # AmbiguityBall at time k
    center_k = ball_k.center   # mixture/discrete center distribution
    radius_k = ball_k.radius   # W2 radius (float / tensor)
    print(f"W2 radius at step {k}: {radius_k}")
```


## Citation

If you use **WassProp** in academic work, please cite the relevant paper:

- [Figueiredo *et al.* (2025)](https://arxiv.org/pdf/2506.08689) — *Formal Uncertainty Propagation for Stochastic Dynamical Systems with Additive Noise*  
- [Adams *et al.* (2025)](https://arxiv.org/pdf/2505.11219) — *Efficient Uncertainty Propagation with Guarantees in Wasserstein Distance*

If you specifically use the **WassProp** package, please also cite the software itself:

```bibtex
@misc{Adams2025,
  author       = {Steven J. L. Adams and Eduardo Figueiredo},
  title        = {WassProp: Formal Uncertainty Propagation in Wasserstein Distance},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/sjladams/WassProp}},
  note         = {Python package}
}
```

## Authors
- **Eduardo Figueiredo** — PhD Candidate, Delft University of Technology (TU Delft)  
- **Steven J. L. Adams** — PhD Candidate, Delft University of Technology (TU Delft)

## Funding and Support

- Delft University of Technology
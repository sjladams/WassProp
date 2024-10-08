import ot
import torch
import matplotlib.pyplot as plt
from bound_propagation import HyperRectangle

import dynamics
import regions

import DistSignatures as ds
import uq_via_dro

from optimization_utils import optimize_with_adam


if __name__ == '__main__':
    torch.manual_seed(0) # for reproducibility

    ### 1) Define dynamics
    num_dims = 1
    r = 4
    f = dynamics.ChaoticDynamics(r)

    # Plot dynamics
    nun_samples = 1000
    x = torch.linspace(start=-5, end=5, steps=500).view(-1, 1)
    y = f(x)

    fig_dynamics = plt.figure()
    plt.plot(x, y)
    plt.title("Dynamics f(x)")
    plt.show()

    # Global Lipschitz constant of f
    print(f"Global Lipschitz constant of f: {f.global_lipschitz()}")

    ### 2) System assumptions

    ##### 2.1) Initial distribution
    p0 = ds.MultivariateNormal(loc=1*torch.ones(num_dims), covariance_matrix=1*torch.diag(1*torch.ones(num_dims)))

    ##### 2.2) Noise structure
    mean_noise = torch.zeros(num_dims)
    variance_noise = torch.ones(num_dims) * 0.3**2
    noise_distribution = ds.MultivariateNormal(mean_noise, torch.diag(variance_noise))

    # noise_samples = torch.normal(mean=mean_noise, std=std_dev_noise, size=initial_distribution_samples.shape)

    ##### 2.3) Monte Carlo simulation of the system
    p0_samples = p0.sample(torch.Size((nun_samples,)))
    p1_samples = f(p0_samples) + noise_distribution.sample(torch.Size((nun_samples,)))

    fig_histograms = plt.figure()
    plt.hist(p0_samples.squeeze(), alpha=0.5, label='t = 0', bins=100, density=True)
    plt.hist(p1_samples.squeeze(), alpha=0.5, label='t = 1', bins=100, density=True)
    plt.legend()
    plt.title("Histograms of the initial and propagated states")
    plt.show()

    ### 3) Create signature approximation for $\mathbb{P}_0$
    sign_p0 = ds.discretization_generator(dist=p0, nr_signature_points=5, compute_w2=True)

    # fig_initial_signature = plt.figure()
    # plt.bar(sign_p0.locs.squeeze(), sign_p0.probs, width=0.1)
    # plt.hist(p0_samples, alpha=0.5, label='t = 0', bins=100, density=True)
    # plt.show()

    ##### 3.2) Compute $\mathbb{W}_{2}(\mathbb{P}_0, \Delta \# \hat{\mathbb{P}}_0)$
    print(f"2-W distance between the true P_0 and our signature approximation {sign_p0.w2:.4f}")

    ##### 3.3) Compute $\mathbb{W}_{2}(\mathbb{P}_1, \hat{\mathbb{P}}_1)$
    q1 = ds.MixtureMultivariateNormal(
        mixture_distribution=torch.distributions.Categorical(probs=sign_p0.probs),
        component_distribution=ds.MultivariateNormal(loc=f(sign_p0.locs) + noise_distribution.loc,
                                                     covariance_matrix=noise_distribution.covariance_matrix))
    q1_samples = q1.sample(torch.Size((nun_samples,)))

    wasserstein_squared_propagation = ot.solve_sample(p1_samples.view(-1, num_dims), q1_samples.view(-1, num_dims)).value
    print(f"wasserstein distance after propagation: {wasserstein_squared_propagation.sqrt():.4f}")

    # fig_t1 = plt.figure()
    # plt.hist(p1_samples, alpha=0.5, label='p', bins=100, density=True)
    # plt.hist(q1_samples, alpha=0.5, label='q', bins=100, density=True)
    # plt.legend()
    # plt.title("Histograms q1 vs p1")
    # plt.show()


    ### 4) Propagating with global Lipschitz
    print(f"Bound using Global Lipschitz constant: {f.global_lipschitz() * sign_p0.w2:.4f}")

    ### 5) Propagating our methods
    ## 5.1) Create Discretization
    # this gives the true Voronoi partitioning for axis-aligned grids of signatures, else it is an hyper-rectangular
    # over-approximation of each Voronoi partition
    regions = regions.generate_voronoi_partition(sign_p0)

    #print(sign_p0.locs.squeeze())
    #print(regions.lower.squeeze())
    #print(regions.upper.squeeze())

    #print(f.interval_approximation(regions).squeeze())


    # Method 1: Finite Linear Problem

    ## Term 1: bound W_2(f#P, f#D#P)
    term1 = uq_via_dro.BoundW2_f_push_P_vs_f_push_SignatureP(sign_p0, f, regions, budget=sign_p0.w2)
    objective_term1 = term1.get_objective()

    lambd = torch.tensor(0.1, requires_grad=True)

    optimized_lambda, losses = optimize_with_adam(param=lambd, lr=0.001, num_iterations=3000, objective=objective_term1)

    bound_term1 = objective_term1(optimized_lambda)
    print(f"Bound (I): {bound_term1:.4f}")

    ## Term 2: bound W_2(f#D#P, f#D#Q)
    term2 = uq_via_dro.BoundW2_f_push_SignatureP_vs_f_push_SignatureQ(sign_p0, f, budget=2*sign_p0.w2)
    bound_term2  = term2.solve_lin_problem()

    print(f"Bound (II): {bound_term2:.4f}")



    term3 = uq_via_dro.BoundW2_f_push_SignatureP_vs_f_push_SignatureQ_via_Trivial(sign_p0, f, regions, budget=sign_p0.w2)
    objective_term3 = term3.get_objective()

    lambd = torch.tensor(6.0, requires_grad=True)

    optimized_lambda, losses = optimize_with_adam(param=lambd, lr=0.001, num_iterations=20000, objective=objective_term3)

    bound_term3 = objective_term3(optimized_lambda)
    print(f"Bound (II) - OTHER: {bound_term3:.4f}")
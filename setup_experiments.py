import ot
import torch
import matplotlib.pyplot as plt

from regions import HyperRectangularVoronoiPartition


import DistSignatures as ds
import uq_via_dro

from optimization_utils import optimize_with_adam



def multi_step_uq(dynamics, params_dynamics, params_noise_dist, params_initial_dist, params_signature,
                  params_simulation):
    # Initialize System Dynamics
    f = dynamics(**params_dynamics)
    print(f"Global Lipschitz constant of f: {f.global_lipschitz()}")
    noise_distribution = ds.MultivariateNormal(**params_noise_dist)

    if f.num_dims == 1:
        # Plot dynamics
        x = torch.linspace(start=-5, end=5, steps=500).view(-1, 1)
        y = f(x)

        fig_dynamics = plt.figure()
        plt.plot(x, y)
        plt.title("Dynamics f(x)")
        plt.show()

    # Initialize state distributions
    q0 = ds.MultivariateNormal(**params_initial_dist)
    q0_samples = q0.sample(torch.Size((params_simulation['num_samples'],)))
    p0_samples = q0_samples.clone()

    # store wasserstein error bounds
    w2_empirical = torch.zeros(params_simulation['K']+1)
    w2_global_lipschitz = torch.zeros(params_simulation['K']+1)
    w2_finite_method = torch.zeros(params_simulation['K']+1)

    # Propagate the system
    for k in range(params_simulation['K']):
        ### Generate the signature of the approximate state distribution at time k:
        sign_q0 = ds.discretization_generator(dist=q0, compute_w2=True, **params_signature)
        print(f"W_2(q_{k}, Delta_C#q_{k}) = {sign_q0.w2:.4f}")

        ### Propagate the (approximate) state distribution over the dynamics
        q1 = ds.MixtureMultivariateNormal(
            mixture_distribution=torch.distributions.Categorical(
                probs=sign_q0.probs),
            component_distribution=ds.MultivariateNormal(
                loc=f(sign_q0.locs) + noise_distribution.loc,
                covariance_matrix=noise_distribution.covariance_matrix
            ))
        q1_samples = q1.sample(torch.Size((params_simulation['num_samples'],)))
        p1_samples = f(p0_samples) + noise_distribution.sample(torch.Size((params_simulation['num_samples'],)))

        ### Compute W_2(p_{k+1}, q_{k+1}) = W_2(f#p_k, f#\Delta_C#q_k)
        ## Empirical Approximate
        w2_empirical[k+1] = ot.solve_sample(p1_samples.view(-1, f.num_dims),
                                            q1_samples.view(-1, f.num_dims)
                                            ).value.sqrt()

        ## Global Lipschitz
        w2_global_lipschitz[k+1] = f.global_lipschitz() * (sign_q0.w2 + w2_global_lipschitz[k])

        ## Finite Linear Method
        # Term 1: bound W_2(f#p_k, f#\Delta_C#p_k)
        fn_bound_on_w2_fP_fdiscP = uq_via_dro.get_fn_bound_on_w2_fP_fdiscP(sign_q0, f,
                                                                           budget=sign_q0.w2 + w2_finite_method[k])

        lambd = torch.tensor(0.01, requires_grad=True)
        optimized_lambda, losses = optimize_with_adam(param=lambd, lr=0.001, num_iterations=300,
                                                      objective=fn_bound_on_w2_fP_fdiscP)

        w2_finite_method_term1 = fn_bound_on_w2_fP_fdiscP(optimized_lambda)

        ## Term 2: bound W_2(f#\Delta#p_k, f#\Delta#q_k)
        if k == 0:
            w2_finite_method_term2 = 0.
        else:
            term2 = uq_via_dro.BoundW2_fdiscP_vs_fdiscQ(sign_q0, f, budget=2*(sign_q0.w2 + w2_finite_method[k]))
            w2_finite_method_term2 = term2.solve_lin_problem()

        print(f"Bounds on W_2(W_2(p_{k+1}, q_{k+1})) via:\n"
              f"\t Global Lipschits: {w2_global_lipschitz[k+1]:.4f}\n"
              f"\t Empirical: {w2_empirical[k+1]:.4f}\n"
              f"\t Finite Linear Method: {w2_finite_method_term1 + w2_finite_method_term2:.4f}\n"
              f"\t\t Term 1: {w2_finite_method_term1:.4f}\n"
              f"\t\t Term 2: {w2_finite_method_term2:.4f}")

        if f.num_dims == 1:
            fig_propagation = plt.figure()
            plt.hist(p0_samples.squeeze(), alpha=0.5, label=f'k={k}', bins=100, density=True)
            plt.hist(p1_samples.squeeze(), alpha=0.5, label=f'k={k+1}', bins=100, density=True)
            plt.legend()
            plt.title(f"Histograms of p_{k} and p_{k+1}")
            plt.show()

            fig_signature = plt.figure()
            plt.bar(sign_q0.locs.squeeze(), sign_q0.probs, width=0.1)
            plt.hist(q0_samples, alpha=0.5, label=f'k={k}', bins=100, density=True)
            plt.title(f"Signature of q_{k} and Histogram of q_{k}")
            plt.show()

            fig_approximation_error = plt.figure()
            plt.hist(p1_samples, alpha=0.5, label=f'p_{k}', bins=100, density=True)
            plt.hist(q1_samples, alpha=0.5, label=f'q_{k}', bins=100, density=True)
            plt.legend()
            plt.title("Histograms q1 vs p1")
            plt.show()


    # Overwrite for next iteration
    q0 = q1
    q0_samples = q1_samples
    p0_samples = p1_samples


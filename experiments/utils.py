from typing import Union, Tuple, List, Optional
from argparse import Namespace
import torch
import discretize_distributions.distributions as dd_dists

from duq_via_wasserstein import multi_step, get_dynamics, AmbiguitySet

from handlers import save_csv


def get_initial_dist(
        loc_initial_dist: torch.Tensor, 
        variance_initial_dist: torch.Tensor,
):
    return construct_diag_gaussian_dist(loc_initial_dist, variance_initial_dist)


def get_noise_dist(loc_noise_dist: torch.Tensor, variance_noise_dist: torch.Tensor):
    if all(isinstance(i, list) for i in loc_noise_dist) and all(isinstance(i, list) for i in variance_noise_dist):
        return dd_dists.MixtureMultivariateNormal(
            mixture_distribution=torch.distributions.Categorical(probs=torch.ones(len(loc_noise_dist))),
            component_distribution=dd_dists.MultivariateNormal(
                loc=torch.as_tensor(loc_noise_dist),
                covariance_matrix=torch.diag_embed(torch.as_tensor(variance_noise_dist))))
    else:
        return construct_diag_gaussian_dist(loc_noise_dist, variance_noise_dist)


def construct_diag_gaussian_dist(loc_dist: Union[list, torch.Tensor], variance_dist: Union[list, torch.Tensor]):
    loc_dist = torch.as_tensor(loc_dist)
    covariance_dist = torch.diag(torch.as_tensor(variance_dist))
    return dd_dists.MultivariateNormal(loc=loc_dist, covariance_matrix=covariance_dist)


def hyper_params_analysis(
    args: Namespace, 
    name_dynamics: str, 
    num_time_steps: int = 20, 
    w2_p__q = 0.01, 
    w2_noise_dist = 0.01, 
    num_locs_options: List[int] = [10], # [10, 100, 1000]
):

    dynamics = get_dynamics(**vars(args))
    print(f"global lipschitz: {dynamics.global_lipschitz}")
    initial_dist = get_initial_dist(args.loc_initial_dist, args.variance_initial_dist)
    noise_dist = get_noise_dist(args.loc_noise_dist, args.variance_noise_dist)
    q = AmbiguitySet(initial_dist, w2_p__q)
    noise = AmbiguitySet(noise_dist, w2_noise_dist)

    store = list()
    for num_locs in num_locs_options:
        path_lagr = multi_step(
                dynamics=dynamics, 
                q=q, 
                noise=noise,
                num_time_steps=num_time_steps,
                use_lagrangian_duality=True,
                num_locs=num_locs,
            )
        
        path_glob = multi_step(
                dynamics=dynamics, 
                q=q, 
                noise=noise,
                num_time_steps=num_time_steps,
                use_lagrangian_duality=False,
                num_locs=num_locs,
            )
        store.append(dict(
            num_locs=num_locs,
            w2_p1__q1_global_lipschitz=float(path_lagr.at(num_time_steps-1).w2),
            w2_p1__q1_lagrangian_duality=float(path_glob.at(num_time_steps-1).w2)
        ))

    if args.save:
        file_name = f"{name_dynamics}_locs={num_locs_options}_steps={num_time_steps}_w2_p__q={w2_p__q}_w2_noise={w2_noise_dist}"
        save_csv(store, f"{args.results_folder}{file_name}")
    else:
        print(store)


def boundary_cond_analysis(
    args: Namespace, 
    name_dynamics: str, 
    num_time_steps: int = 20, 
    num_locs: int = 100, 
    w2_p__q_options: List[float] = [0.001],   # [0.001, 0.01, 0.1]
    w2_noise_dist_options: List[float] = [0.001],   # [0.001, 0.01, 0.1]
):

    dynamics = get_dynamics(**vars(args))
    print(f"global lipschitz: {dynamics.global_lipschitz}")
    initial_dist = get_initial_dist(args.loc_initial_dist, args.variance_initial_dist)
    noise_dist = get_noise_dist(args.loc_noise_dist, args.variance_noise_dist)

    store = list()
    for w2_p__q in w2_p__q_options:
        for w2_noise_dist in w2_noise_dist_options:
            path_lagr = multi_step(
                dynamics=dynamics, 
                q=AmbiguitySet(initial_dist, w2_p__q), 
                noise=AmbiguitySet(noise_dist, w2_noise_dist),
                num_time_steps=num_time_steps,
                use_lagrangian_duality=True,
                num_locs=args.num_locs,
            )

            path_glob = multi_step(
                dynamics=dynamics, 
                q=AmbiguitySet(initial_dist, w2_p__q), 
                noise=AmbiguitySet(noise_dist, w2_noise_dist),
                num_time_steps=num_time_steps,
                use_lagrangian_duality=False,
                num_locs=args.num_locs,
            )

            store.append(dict(
                w2_p__q=w2_p__q,
                w2_noise_dist=w2_noise_dist,
                w2_p1__q1_global_lipschitz=float(path_glob.at(num_time_steps-1).w2),
                w2_p1__q1_lagrangian_duality=float(path_lagr.at(num_time_steps-1).w2)
            ))

    if args.save:
        file_name = f"{name_dynamics}_locs={num_locs}_steps={num_time_steps}_w2_p__q={w2_p__q_options}_w2_noise={w2_noise_dist_options}"
        save_csv(store, f"{args.results_folder}{file_name}")
    else:
        print(store)
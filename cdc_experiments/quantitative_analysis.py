import os

from experiments import multi_step
from dynamics import get_dynamics

from utils import parse_arguments, save_csv
from utils_distributions import get_initial_dist, get_noise_dist


FOLDER = f"{os.getcwd()}{os.sep}results"


def hyper_params_analysis(args, name: str):
    hyper_params_options = dict(
        num_locs=[10, 100, 1000],
        size_after_compr=[1, 5, 10]
    )

    num_time_steps = 20
    w2_p__q = 0.01
    w2_noise_dist = 0.01

    tag = f"{name}_locs={hyper_params_options['num_locs']}_compr={hyper_params_options['size_after_compr']}_steps={num_time_steps}_w2_p__q={w2_p__q}_w2_noise={w2_noise_dist}"

    dynamics = get_dynamics(**vars(args))
    print(f"global lipschitz: {dynamics.global_lipschitz}")
    initial_dist = get_initial_dist(args.loc_initial_dist, args.variance_initial_dist)
    noise_dist = get_noise_dist(args.loc_noise_dist, args.variance_noise_dist)

    store = list()
    for num_locs in hyper_params_options['num_locs']:
        for size_after_compr in hyper_params_options['size_after_compr']:
            _, w2_p1__q1_store, _, _ = multi_step(
                w2_p__q= w2_p__q,
                w2_noise_dist= w2_noise_dist,
                dynamics=dynamics,
                noise_dist=noise_dist,
                q=initial_dist,
                num_time_steps=num_time_steps,
                run_lagrangian_duality=True,
                run_empirical=False,
                propagate_via_gmm=True,
                num_samples=args.num_samples,
                num_locs=num_locs,
                size_after_compr=size_after_compr
            )
            store.append(dict(
                num_locs=num_locs,
                size_after_compr=size_after_compr,
                w2_p1__q1_global_lipschitz=w2_p1__q1_store[num_time_steps-1]['w2_p1__q1_global_lipschitz'].item(),
                w2_p1__q1_lagrangian_duality=w2_p1__q1_store[num_time_steps-1]['w2_p1__q1_lagrangian_duality'].item()))

    save_csv(store, f"{FOLDER}{os.sep}{tag}")


def boundary_cond_analysis(args, name: str):
    hyper_params_options = dict(
        w2_p__q=[0.001, 0.01, 0.1],
        w2_noise_dist=[0.001, 0.01, 0.1]
    )
    num_time_steps = 20
    num_locs = 100
    size_after_compr = 5

    tag = f"{name}_locs={num_locs}_compr={size_after_compr}_steps={num_time_steps}_w2_p__q={hyper_params_options['w2_p__q']}_w2_noise={hyper_params_options['w2_noise_dist']}"

    dynamics = get_dynamics(**vars(args))
    print(f"global lipschitz: {dynamics.global_lipschitz}")
    initial_dist = get_initial_dist(args.loc_initial_dist, args.variance_initial_dist)
    noise_dist = get_noise_dist(args.loc_noise_dist, args.variance_noise_dist)

    store = list()
    for w2_p__q in hyper_params_options['w2_p__q']:
        for w2_noise_dist in hyper_params_options['w2_noise_dist']:
            _, w2_p1__q1_store, _, _ = multi_step(
                w2_p__q=w2_p__q,
                w2_noise_dist=w2_noise_dist,
                dynamics=dynamics,
                noise_dist=noise_dist,
                q=initial_dist,
                num_time_steps=num_time_steps,
                run_lagrangian_duality=True,
                run_empirical=False,
                propagate_via_gmm=True,
                num_samples=args.num_samples,
                num_locs=num_locs,
                size_after_compr=size_after_compr
            )
            store.append(dict(
                w2_p__q=w2_p__q,
                w2_noise_dist=w2_noise_dist,
                w2_p1__q1_global_lipschitz=w2_p1__q1_store[num_time_steps-1]['w2_p1__q1_global_lipschitz'].item(),
                w2_p1__q1_lagrangian_duality=w2_p1__q1_store[num_time_steps-1]['w2_p1__q1_lagrangian_duality'].item()))

    save_csv(store, f"{FOLDER}{os.sep}{tag}")
from typing import Union, Tuple, List, Optional
from argparse import Namespace

from duq_via_wasserstein import multi_step, AmbiguityBall

from dynamics import get_stoch_dynamics
from handlers import save_csv
import utils

def hyper_params_analysis(
    args: Namespace, 
    name_dynamics: str, 
    num_time_steps: int = 20, 
    w2_p__q = 0.01, 
    w2_noise_dist = 0.01, 
    num_locs_options: List[int] = [10], # [10, 100, 1000]
):

    dynamics = get_stoch_dynamics(name=args.dynamics_type, **vars(args.dynamics))
    initial_dist = utils.get_initial_dist(loc=args.initial_dist.loc, variance=args.initial_dist.variance)
    noise_dist = utils.get_noise_dist(loc=args.noise_dist.loc, variance=args.noise_dist.variance)
    q = AmbiguityBall(initial_dist, w2_p__q)
    noise = AmbiguityBall(noise_dist, w2_noise_dist)

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

    dynamics = get_stoch_dynamics(name=args.dynamics_type, **vars(args.dynamics))
    initial_dist = utils.get_initial_dist(loc=args.initial_dist.loc, variance=args.initial_dist.variance)
    noise_dist = utils.get_noise_dist(loc=args.noise_dist.loc, variance=args.noise_dist.variance)

    store = list()
    for w2_p__q in w2_p__q_options:
        for w2_noise_dist in w2_noise_dist_options:
            path_lagr = multi_step(
                dynamics=dynamics, 
                q=AmbiguityBall(initial_dist, w2_p__q), 
                noise=AmbiguityBall(noise_dist, w2_noise_dist),
                num_time_steps=num_time_steps,
                use_lagrangian_duality=True,
                num_locs=args.num_locs,
            )

            path_glob = multi_step(
                dynamics=dynamics, 
                q=AmbiguityBall(initial_dist, w2_p__q), 
                noise=AmbiguityBall(noise_dist, w2_noise_dist),
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
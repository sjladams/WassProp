import json
import argparse
from typing import Optional

def load_json(filename: str):
    with open(f"configs/{filename}.json", "r") as read_file:
        data = json.load(read_file)
    return data


def param_handler(param_name: str, dataset_name: str, num_dims: int, setting_tag: int = None):
    params = load_json(param_name)[dataset_name]
    if "dimensions" in params:
        return params["dimensions"][str(num_dims)]["options"][str(setting_tag)]
    else:
        return params["options"][str(setting_tag)]


def parse_arguments(
        dynamics_type: str = 'ChaoticDynamics',
        num_dims: int = 1,
        dynamics_setting: int = 0,
        num_locs: int = 10,
        num_samples: int = 1000,
        lr: float = 0.01,
        num_iterations: int = 1000,
        plot: bool = False,
        optimize_locs: bool = False,
        additive_gaussian_noise: bool = True,
        num_locs_after_compr: Optional[int] = None,
):
    parser = argparse.ArgumentParser(description='Setup experiments for dynamics.')
    parser.add_argument('--dynamics_type',
                        type=str,
                        choices=['GaussianDynamics1d', 'ChaoticDynamics', 'LinearDynamics'],
                        default=dynamics_type,
                        help='Type of dynamics to use.')
    parser.add_argument('--num_dims',
                        type=int,
                        default=num_dims,
                        help='Number of dimensions of the dynamics')
    parser.add_argument('--dynamics_setting',
                        type=int,
                        default=dynamics_setting,
                        help='Parameters for the dynamics as a dictionary string.')
    parser.add_argument('--num_locs',
                        type=int,
                        default=num_locs,
                        help='Size of discretization grid.')
    parser.add_argument('--num_locs_after_compr',
                        type=int,
                        default=num_locs if num_locs_after_compr is None else num_locs_after_compr,
                        help='Size of discretization grid after compression operation')
    parser.add_argument('--additive_gaussian_noise',
                        type=bool,
                        default=additive_gaussian_noise,
                        help='Assume dynamics has additive Gaussian noise, i.e. g(x, eps) = f(x) + eps, eps Gaussian')
    parser.add_argument('--optimize_locs',
                        type=bool,
                        default=optimize_locs,
                        help='Allow to optimize the signature locations w.r.t. W2(p,q) using gradient descent')
    parser.add_argument('--num_samples',
                        type=int,
                        default=num_samples,
                        help='Number of samples for empirical distribution estimate.')
    parser.add_argument('--plot',
                        type=bool,
                        default=plot,
                        help='Plot the dynamics and distributions.')
    parser.add_argument('--lr',
                        type=float,
                        default=lr,
                        help='Learning rate gradient descent.')
    parser.add_argument('--num_iterations',
                        type=int,
                        default=num_iterations,
                        help='Number of iterations.')

    return parser.parse_args()


def load_params(args):
    dynamics_params = param_handler(
        param_name="dynamics",
        dataset_name=args.dynamics_type,
        num_dims=args.num_dims,
        setting_tag=args.dynamics_setting
    )

    return {"dynamics_type": args.dynamics_type,  # \todo just transform args to dict and merge with dynamics_params, current implementation requires to include every argument in the argument parser explicitly in here
            "num_samples": args.num_samples,
            "num_locs": args.num_locs,
            "num_locs_after_compr": args.num_locs_after_compr,
            "lr": args.lr,
            "num_iterations": args.num_iterations,
            "additive_gaussian_noise": args.additive_gaussian_noise,
            "optimize_locs": args.optimize_locs,
            "plot": args.plot,
            **dynamics_params}
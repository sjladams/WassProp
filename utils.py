import json
import argparse
from typing import Optional
import os

dir = os.path.dirname(os.path.abspath(__file__))

def load_json(filename: str):
    file_path = os.path.join(dir, "configs", f"{filename}.json")
    with open(file_path, "r") as read_file:
        data = json.load(read_file)
    return data


def param_handler(param_name: str, dataset_name: str, setting_tag: int = None):
    params = load_json(param_name)[dataset_name]
    return argparse.Namespace(**params["options"][str(setting_tag)])


def parse_arguments(
        dynamics_type: str = 'ChaoticDynamics',
        dynamics_setting: int = 0,
        num_locs: int = 10,
        num_samples: int = 1000,
        lr: float = 0.01,
        num_iterations: int = 1000,
        plot: bool = False,
        num_locs_after_compr: Optional[int] = None,
):
    parser = argparse.ArgumentParser(description='Setup experiments for dynamics.')
    parser.add_argument('--dynamics_type',
                        type=str,
                        choices=['GaussianDynamics1d', 'ChaoticDynamics', 'LinearDynamics'],
                        default=dynamics_type,
                        help='Type of dynamics to use.')
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

    args = parser.parse_args()

    dynamics_params = param_handler(
        param_name="dynamics",
        dataset_name=args.dynamics_type,
        setting_tag=args.dynamics_setting
    )

    args.__dict__.update(vars(dynamics_params))
    return args

import json
import argparse

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
        plot: bool = False
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

    return {"dynamics_type": args.dynamics_type,
            "num_samples": args.num_samples,
            "num_signature_points": args.num_locs,
            "lr": args.lr,
            "num_iterations": args.num_iterations,
            **dynamics_params}
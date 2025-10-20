import json
from argparse import Namespace, ArgumentParser
import os
import csv


folder = os.path.dirname(os.path.abspath(__file__))


def load_json(filename: str):
    file_path = os.path.join(folder, "configs", f"{filename}.json")
    with open(file_path, "r") as read_file:
        data = json.load(read_file)
    return data

def param_handler(args):
    params = load_json("dynamics")[args.dynamics_type][str(args.dynamics_setting)]
    params = Namespace(**{k: Namespace(**v) for k, v in params.items()})
    args.__dict__.update(vars(params))

    assert "initial_dist" in args and "noise_dist"in args, "Please specify initial and noise distribution in config file."
    if not "dynamics" in args:
        args.__dict__.update(dict(dynamics=Namespace()))
    
    return args

def parse_arguments(
        dynamics_type: str = 'ChaoticDynamics',
        dynamics_setting: int = 0,
        num_locs: int = 10,
        num_samples: int = 1000,
        save: bool = False,
):
    parser = ArgumentParser(description='Setup experiments for dynamics.')
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
    parser.add_argument('--num_samples',
                        type=int,
                        default=num_samples,
                        help='Number of samples for empirical distribution estimate.')
    parser.add_argument('--save',
                        type=bool,
                        default=save,
                        help='Whether to save the plots or show them.')
    
    args = parser.parse_args()

    args = param_handler(args)

    args.results_folder = f"{os.path.dirname(os.path.abspath(__file__))}{os.sep}results{os.sep}"
    args.data_folder = f"{os.path.dirname(os.path.abspath(__file__))}{os.sep}data{os.sep}"
    return args


def save_csv(store: dict, path: str):
    with open(f"{path}.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=store[0].keys(),delimiter=';')
        writer.writeheader()
        writer.writerows(store)
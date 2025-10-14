import torch

from duq_via_wasserstein import single_step, Path, AmbiguitySet
from duq_via_wasserstein.dynamics import FourModesOpenLoopDynamics, AdditiveGaussianDynamics

from handlers import parse_arguments
import utils


if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        dynamics_type="FourModesOpenLoopDynamics",
        dynamics_setting = 0,
        num_locs = 100,
        num_samples = 1000,
        save = False
    )

    initial_dist = utils.get_initial_dist(args.loc_initial_dist, args.variance_initial_dist)
    noise_dist = utils.get_noise_dist(args.loc_noise_dist, args.variance_noise_dist)

    # Define open loop control strategy
    open_loop_control = [1, 4, 2, 4, 4, 1, 3, 3, 3, 2]

    # Define initial radius
    noise = AmbiguitySet(noise_dist, 0.)
    
    path_lagr, path_glob = Path(), Path()
    path_lagr.append(-1, AmbiguitySet(initial_dist, 0.1))
    path_glob.append(-1, AmbiguitySet(initial_dist, 0.1))

    for k, control in enumerate(open_loop_control):
        dynamics = AdditiveGaussianDynamics(FourModesOpenLoopDynamics(control=control, **vars(args))) # Build dynamics according to control mode

        print(f'---- TIME STEP {k} ----')
        path_lagr.append(k, single_step(
            dynamics=dynamics,
            q=path_lagr.at(k-1),
            noise=noise,
            num_locs=args.num_locs,
            use_lagrangian_duality=True
        ))

        path_glob.append(k, single_step(
            dynamics=dynamics,
            q=path_glob.at(k-1),
            noise=noise,
            num_locs=args.num_locs,
            use_lagrangian_duality=False
        ))

        print(
            f"Bounds on W_2(p_{k+1}, q_{k+1}) via:\n"
            f"\t Global Lipschitz: {path_glob.at(k).w2:.4f}\n"
            f"\t Lagrangian Duality: {path_lagr.at(k).w2:.4f}\n"
        )

from experiments import single_step
from dynamics import FourModesOpenLoopDynamics, AdditiveGaussianDynamics
from utils import parse_arguments
import torch

from utils_distributions import get_initial_dist, get_noise_dist


if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        dynamics_type="FourModesOpenLoopDynamics",
        dynamics_setting = 0,
        num_locs = 100,
        size_after_compr=100,
        num_samples = 1000,
        plot = False
    )

    # Build center distributions
    initial_dist = get_initial_dist(args.loc_initial_dist, args.variance_initial_dist)
    noise_dist = get_noise_dist(args.loc_noise_dist, args.variance_noise_dist)

    # Define open loop control strategy
    open_loop_control = [1, 4, 2, 4, 4, 1, 3, 3, 3, 2]

    # Define initial radius
    w2_p1__q1_store = {-1: dict(w2_p1__q1_global_lipschitz=0.1, w2_p1__q1_lagrangian_duality=0.1)}
    w2_q__sign_q_store = dict()
    samples_store = dict()

    q = initial_dist

    for k, control in enumerate(open_loop_control):

        dynamics = AdditiveGaussianDynamics(FourModesOpenLoopDynamics(control=control, **vars(args))) # Build dynamics according to control mode

        print(f'---- TIME STEP {k} ----')
        out = single_step(
            dynamics=dynamics,
            noise_dist=noise_dist,
            q=q,
            num_samples=args.num_samples,
            num_locs=args.num_locs,
            propagate_via_gmm=True,
            size_after_compr=args.size_after_compr,
            w2_p__q_global_lipschitz=w2_p1__q1_store[k-1]['w2_p1__q1_global_lipschitz'],
            w2_p__q_lagrangian_duality=w2_p1__q1_store[k-1]['w2_p1__q1_lagrangian_duality'],
            p_samples=samples_store[k - 1]['p1_samples'] if k - 1 in samples_store else None,
        )

        w2_p1__q1_store[k] = {key: value for key, value in out.items() if 'w2_p1__q1' in key}
        w2_q__sign_q_store[k] = out['w2_q__sign_q']
        samples_store[k] = {key: value for key, value in out.items() if 'samples' in key}

        # Get mixture as the center of the ball
        q = out['q1']

        print(
            f"Bounds on W_2(p_{k+1}, q_{k+1}) via:\n"
            f"\t Global Lipschitz: {out['w2_p1__q1_global_lipschitz']:.4f}\n"
            f"\t Lagrangian Duality: {out['w2_p1__q1_lagrangian_duality']:.4f}\n"
        )

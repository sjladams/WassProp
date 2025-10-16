import torch
import bound_propagation as bp

from duq_via_wasserstein import single_step, Path, AmbiguitySet
import duq_via_wasserstein.dynamics as dyn

from handlers import parse_arguments
import utils


class FourModesOpenLoopDynamics(dyn.Dynamics):
    def __init__(self, control: int = 1):
        linear_part = dyn.LinearDynamics(weight=torch.eye(2))

        if control == 1:
            trig_part = torch.nn.Sequential(
                dyn.Linear(
                    torch.tensor([
                        [0.0, 1.0],
                        [1.0, 0.0]
                    ]),
                    torch.tensor([0.0, torch.pi / 2])
                    ),
                bp.Sin(),
                dyn.Linear(
                    torch.tensor([
                        [0.2, 0.0],
                        [0.0, 0.4]
                    ]),
                    torch.tensor([0.5, 0.0])
                ),
            )
        elif control == 2:
            trig_part = torch.nn.Sequential(
                dyn.Linear(
                    torch.tensor([
                        [0.0, 1.0],
                        [1.0, 0.0]
                    ]),
                    torch.tensor([0.0, torch.pi / 2])
                    ),
                bp.Sin(),
                dyn.Linear(
                    torch.tensor([
                        [0.2, 0.0],
                        [0.0, 0.4]
                    ]),
                    torch.tensor([-0.5, 0.0])
                ),
            )
        elif control==3:
            trig_part = torch.nn.Sequential(
                dyn.Linear(
                    torch.tensor([
                        [0.0, 1.0],
                        [1.0, 0.0]
                    ]),
                    torch.tensor([torch.pi / 2, 0.0])
                    ),
                bp.Sin(),
                dyn.Linear(
                    torch.tensor([
                        [0.4, 0.0],
                        [0.0, 0.2]
                    ]),
                    torch.tensor([0.0, 0.5])
                ),
            )
        elif control==4:
            trig_part = torch.nn.Sequential(
                dyn.Linear(
                    torch.tensor([
                        [0.0, 1.0],
                        [1.0, 0.0]
                    ]),
                    torch.tensor([torch.pi / 2, 0.0])
                    ),
                bp.Sin(),
                dyn.Linear(
                    torch.tensor([
                        [0.4, 0.0],
                        [0.0, 0.2]
                    ]),
                    torch.tensor([0.0, -0.5])
                ),
            )
        else:
            raise Exception

        super().__init__(
            num_dims=2, 
            modules=[bp.Parallel(linear_part, trig_part), bp.VectorAdd()],
        )

    @property
    def global_lipschitz(self):
        return 1.4


if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        dynamics_type="FourModesOpenLoopDynamics",
        dynamics_setting = 0,
        num_locs = 100,
        num_samples = 1000,
        save = False
    )

    initial_dist = utils.get_initial_dist(loc=args.initial_dist.loc, variance=args.initial_dist.variance)
    noise_dist = utils.get_noise_dist(loc=args.noise_dist.loc, variance=args.noise_dist.variance)
    
    # Define open loop control strategy
    open_loop_control = [1, 4, 2, 4, 4, 1, 3, 3, 3, 2]

    # Define initial radius
    noise = AmbiguitySet(noise_dist, 0.)
    
    path_lagr, path_glob = Path(), Path()
    path_lagr.append(-1, AmbiguitySet(initial_dist, 0.1))
    path_glob.append(-1, AmbiguitySet(initial_dist, 0.1))

    for k, control in enumerate(open_loop_control):
        dynamics = dyn.AdditiveNoiseDynamics(FourModesOpenLoopDynamics(control=control))

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

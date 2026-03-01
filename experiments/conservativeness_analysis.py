import ot
import torch
import pprint

from discretize_distributions import CategoricalFloat, MultivariateNormal

from experiments.dimension_analysis import sample_weight, get_normalized_variance
from wass_prop import single_step, AmbiguityBall

from dynamics import get_stoch_dynamics
from handlers import parse_arguments
import utils
from wass_prop.dynamics import AdditiveNoiseDynamics, NNLayerDynamics
from wass_prop.propagation import single_step_empirical


def run_single_step_no_ambiguity(dynamics_type, setting, num_locs):
    args = parse_arguments(
        dynamics_type=dynamics_type,
        dynamics_setting=setting,
        num_locs=num_locs,
    )

    dynamics = get_stoch_dynamics(name=args.dynamics_type, **vars(args.dynamics))
    initial_dist = utils.get_initial_dist(loc=args.initial_dist.loc, variance=args.initial_dist.variance)
    noise_dist = CategoricalFloat(
        locs=torch.zeros(1, len(args.initial_dist.loc)),
        probs=torch.tensor([1.0])
    )

    initial_ball = AmbiguityBall(initial_dist, 0.)
    noise_ball = AmbiguityBall(noise_dist, 0.) # we don't consider the noise in this experiment (Dirac at zero)

    results = dict()
    for method in ['global_lipschitz', 'lagrangian_duality']:
        propagated_ball = single_step(
            dynamics=dynamics,
            q=initial_ball,
            noise=noise_ball,
            num_locs=args.num_locs,
            use_lagrangian_duality=True if method == 'lagrangian_duality' else False,
        )
        results[method] = float(propagated_ball.w2)

    num_samples_empirical = 50000
    p_samples = initial_ball.sample(num_samples_empirical) # sample from zero radius set is equivalent to sampling from center
    samples_empirical = single_step_empirical(
        dynamics=dynamics,
        p_emp=p_samples,
        noise=noise_ball,
        num_samples=num_samples_empirical,
    )

    empirical_w2 = ot.solve_sample(X_a=samples_empirical, X_b=propagated_ball.center.locs, b=propagated_ball.center.probs, metric="sqeuclidean").value.pow(1 / 2).item()
    results['empirical'] = empirical_w2

    return results

def run_nn_layer_from_dimensionality_analysis(dimension, num_locs):
    weight = sample_weight(dimension)  # random NN layer
    distribution = MultivariateNormal(loc=torch.zeros(dimension),
                                      covariance_matrix=get_normalized_variance(dimension) * torch.eye(dimension))

    dynamics = AdditiveNoiseDynamics(state_dynamics=NNLayerDynamics(weight=weight, bias=None))

    noise_dist = CategoricalFloat(
        locs=torch.zeros(1, len(distribution.loc)),
        probs=torch.tensor([1.0])
    )

    initial_ball = AmbiguityBall(distribution, 0.)
    noise_ball = AmbiguityBall(noise_dist, 0.) # we don't consider the noise in this experiment (Dirac at zero)

    results = dict()
    for method in ['global_lipschitz', 'lagrangian_duality']:
        propagated_ball = single_step(
            dynamics=dynamics,
            q=initial_ball,
            noise=noise_ball,
            num_locs=num_locs,
            use_lagrangian_duality=True if method == 'lagrangian_duality' else False,
        )
        results[method] = float(propagated_ball.w2)

    num_samples_empirical = 50000
    p_samples = initial_ball.sample(num_samples_empirical) # sample from zero radius set is equivalent to sampling from center
    samples_empirical = single_step_empirical(
        dynamics=dynamics,
        p_emp=p_samples,
        noise=noise_ball,
        num_samples=num_samples_empirical,
    )

    empirical_w2 = ot.solve_sample(X_a=samples_empirical, X_b=propagated_ball.center.locs, b=propagated_ball.center.probs, metric="sqeuclidean").value.pow(1 / 2).item()
    results['empirical'] = empirical_w2

    return results

def conservativeness_analysis():
    run_inputs = { # [dynamics_type, dynamics_setting, num_locs]
        'Sigmoid (1D)' : ('SigmoidDynamics', 0, 10),
        'Bounded Linear (2D)' : ('BoundedLinearDynamics', 0, 100),
        'Quadruple-Tank (4D)' : ('LinearDynamics', 0, 1000),
        'NN Layer (10D)' : ('DiagonalSigmoidDynamics', 2, 10000),
        'Mountain Car (2D)' : ('MountainCarDynamics', 0, 100),
        'Dubins car (3D)' : ('DubinsCarDynamics', 0, 1000)
    }

    results = {}
    for dynamics_name, (dynamics_type, setting, num_locs) in run_inputs.items():
        results[dynamics_name] = {}
        results[dynamics_name][num_locs] = run_single_step_no_ambiguity(dynamics_type, setting, num_locs)

    pprint.pprint(results)

    # Also report the NN Layer from dimension analysis
    results = run_nn_layer_from_dimensionality_analysis(dimension=10, num_locs=10000)
    pprint.pprint(results)


if __name__ == '__main__':
    torch.manual_seed(0)

    conservativeness_analysis()
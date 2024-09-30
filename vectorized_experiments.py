import math
import matplotlib.pyplot as plt
import ot
import torch
import numpy as np
from scipy.optimize import linprog


class _Dynamics:
    def interval_approximation(self, regions: torch.Tensor, centers: torch.Tensor):
        """

        :param regions: shape =
        :return:
        """
        # \todo use bound_propagation to compute the interval or affine approximates of dynamics
        # Note that currently, we imply a monotone function within each region. This is not always the case.
        return (self(regions) - self(centers).view(-1, 1)).abs().max(dim=-1).values

    def F(self, centers: torch.Tensor):
        return (self(centers).view(1, -1) - self(centers).view(-1, 1)).abs()

    def __call__(self, *args, **kwargs):
        pass


class GaussianDynamics(_Dynamics):
    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x: torch.Tensor):
        coefficient = 1 / (self.sigma * math.sqrt(2 * math.pi))
        exponent = (-0.5 * ((x - self.mu) / self.sigma).pow(2)).exp()
        return coefficient * exponent

    @staticmethod
    def global_lipschitz():
        return math.exp(-1 / 2) / 2 * math.pi



if __name__ == "__main__":
    # \TODO use snn2mgps package to create signatures, and initiate gmms objects.
    torch.manual_seed(0) # for reproducibility

    # Experiment 1: 1D Gaussian dynamics

    # x_{t+1} = f(x_t) + \epsilon_t, \; \epsilon_t \sim \mathcal{N}(0, \sigma^{2}_\epsilon)
    #
    # In this first experiment, our dynamics is given by $f(x) = \phi(x)$, where $\phi(x)$ is the function describing
    # the Gaussian density with mean $0$ and variance $1$. This dynamics takes states far from zero and bring back to
    # zero, while states close to zero are taken to positions close to $0.4$. Thus, for any initial distribution, as
    # time goes on, the state distribution tends to get more concentrated close to zero.
    #
    # We consider an initial distribution $\mathbb{P}_{0} = \mathcal{N}(1, \sigma^{2}_0)$, which is centered in the
    # region where the dynamics $f$ has the highest variation.


    ### 1) Define dynamics
    n_dims = 1
    f = GaussianDynamics(mu=0, sigma=1)

    # Plot dynamics
    n_samples = 1000
    x = torch.linspace(start=-5, end=5, steps=500)
    y = f(x)

    fig_dynamics = plt.figure()
    plt.plot(x, y)
    plt.title("Dynamics f(x)")
    plt.show()

    # Global Lipschitz constant of f
    print(f"Global Lipschitz constant of f: {f.global_lipschitz()}")

    ### 2) System assumptions

    ##### 2.1) Initial distribution
    # We consider $\mathbb{P}_{0} = \mathcal{N}(1, \sigma^{2}_0)$

    initial_distribution = torch.distributions.Normal(loc=1, scale=1)
    initial_distribution_samples = initial_distribution.sample(sample_shape=torch.Size((n_samples,)))

    ##### 2.2) Noise structure
    # We consider, for every $t$, the noise structure to be given by
    # $\epsilon_t \sim \mathcal{N}(0, \sigma^{2}_\epsilon)$, with $\sigma^{2}_\epsilon = 0.3^2$.

    mean_noise = 0
    std_dev_noise = 0.3
    noise_samples = torch.normal(mean=mean_noise, std=std_dev_noise, size=initial_distribution_samples.shape)

    ##### 2.3) Monte Carlo simulation of the system
    # We sample the system and build the histograms for $x_0$ and $x_1$. Those are approximations of the true
    # distributions $\mathbb{P}_0$ and $\mathbb{P}_1$.
    propagated_states = f(initial_distribution_samples) + noise_samples

    fig_histograms = plt.figure()
    plt.hist(initial_distribution_samples, alpha=0.5, label='t = 0', bins=100, density=True)
    plt.hist(propagated_states, alpha=0.5, label='t = 1', bins=100, density=True)
    plt.legend()
    plt.title("Histograms of the initial and propagated states")
    plt.show()

    ### 3) Create signature approximation for $\mathbb{P}_0$
    # We create a naive signature approximation for $\mathbb{P}_0 \sim \mathcal{N}(\mu_0, \sigma^{2}_{0})$ by taking N-1
    # equally spaced points between $[\mu_0 - 3\sigma_0, \mu_0 + 3\sigma_0]$. Then we create $N$ regions forming a
    # partition of $\mathbb{R}$ in which each signature is in its center location (except for the unbounded regions,
    # where the signatures are in arbitrary pre-chosen locations).

    n_signatures = 10
    inner_edges = torch.linspace(start=initial_distribution.mean - 3 * initial_distribution.stddev,
                                 end=initial_distribution.mean + 3 * initial_distribution.stddev,
                                 steps=n_signatures-1)
    edges = torch.cat((-torch.tensor(torch.inf).view(1), inner_edges, torch.tensor(torch.inf).view(1)))

    initial_signature_probs = initial_distribution.cdf(inner_edges).diff(prepend=torch.zeros(1), append=torch.ones(1))
    signatures = torch.cat(((initial_distribution.mean - 3 * initial_distribution.stddev - 1).view(1),
                            inner_edges[:-1] + inner_edges.diff() * 0.5,
                           (initial_distribution.mean + 3 * initial_distribution.stddev + 1).view(1)))
    print(f"Signatures: {signatures}")

    fig_initial_signature = plt.figure()
    plt.bar(signatures, initial_signature_probs, width=0.1)
    plt.show()

    ##### 3.2) Compute $\mathbb{W}_{2}(\mathbb{P}_0, \Delta \# \hat{\mathbb{P}}_0)$
    # We compute an approximation of this quantity by sampling from $\mathbb{P}_0$ and using the POT package.
    wasserstein_squared_zero = ot.solve_sample(X_a=initial_distribution_samples.view(-1, n_dims),
                                               X_b=signatures.view(-1, n_dims), b=initial_signature_probs).value
    print(f"2-W distance between the true P_0 and our signature approximation {wasserstein_squared_zero.sqrt():.4f}")

    ##### 3.3) Compute $\mathbb{W}_{2}(\mathbb{P}_1, \hat{\mathbb{P}}_1)$
    # Recall that although $\mathbb{P}_1$ is unknown, we can generate samples via MC simulation. Additionally, in our
    # setting, $\hat{\mathbb{P}}_1$ is a GMM with $N$ components centered at $f(x^{\text{signature}}_{i})$ with
    # variance $\sigma^{2}_{\epsilon}$. We sample from this GMM and compute the Wasserstein distance between those
    # two samples to obtain an approximation of the actual Wasserstein distance between the true $\mathbb{P}_1$
    # and our GMM.

    def sample_gmm(centers: torch.Tensor, probabilities: torch.Tensor, st_dev: float, n_samples: int=1):
        #Choose a component based on the provided probabilities
        chosen_components = torch.multinomial(probabilities, n_samples, replacement=True)

        #Sample from the chosen Gaussian component
        return torch.normal(mean=centers[chosen_components], std=st_dev)

    centers = f(signatures)
    samples_gmm = sample_gmm(centers, initial_signature_probs, std_dev_noise, n_samples=1000)

    wasserstein_squared_propagation = ot.solve_sample(samples_gmm.view(-1, n_dims), propagated_states.view(-1, n_dims)).value
    print(f"wasserstein distance after propagation: {wasserstein_squared_propagation.sqrt():.4f}")

    # **Remark**: As previously mentioned, the dynamics tends to concentrate the mass around zero. Therefore,
    # although $\mathbb{W}_{2}(\mathbb{P}_0, \Delta \# \hat{\mathbb{P}}_0)$ is reasonably big, we see that
    # $\mathbb{W}_{2}(\mathbb{P}_1, \hat{\mathbb{P}}_1)$ is actually very small.

    # It is clear that propagating the uncertainty with the global Lipschitz is considerably bad in this case.

    ### 4) Propagating with global Lipschitz
    print(f"Bound using Global Lipschitz constant: {f.global_lipschitz() * wasserstein_squared_zero.sqrt():.4f}")

    ### 5) Propagating with our method
    ##### 5.1) Create regions
    regions = torch.vstack((edges[:-1], edges[1:])).swapaxes(0, 1)

    ##### 5.2) Compute interval approximations for $f$
    # Given two regions $\mathcal{R}_1, \mathcal{R}_2$, we return a $\beta$ such that $|f(x_1) - f(x_2)|^2 \leq \beta$
    # for any $x_1 \in \mathcal{R}_1$, $x_2 \in \mathcal{R}_2$. Note that this works for $\mathcal{R}_\text{unbounded}$
    # as $f$ is a bounded function.

    beta = f.interval_approximation(regions, signatures).pow(2)

    ##### 5.3) Compute projection matrix
    # This element $(i, j)$ of this matrix contains the quantity $|\text{proj}_{\mathcal{R}_i}(c_j) - c_j|^2$.
    # TODO This function can be vectorized. We address this later as we only compute this matrix once.

    projection_matrix = torch.zeros(n_signatures, n_signatures)

    for i in range(len(regions)):
        for j in range(len(regions)):
            if signatures[j] <= regions[i][0]:
                project = regions[i][0] - signatures[j]
            elif signatures[j] >= regions[i][1]:
                project = signatures[j] - regions[i][1]
            else:
                project = 0.

            projection_matrix[i][j] = project ** 2

    ##### 5.5) Project matrix on subspace
    # Our matrix $\Omega$ needs to satisfy $\sum_{j} \omega_{ij} \hat{p}_j = 1$. Given a matrix $\Omega^{'}$ that may
    # disrespect this restriction, we enforce it by making the following adjustment (denote by $\hat{p}$ the vector
    # with all $\hat{p}_j$):

    # $$ \Omega^{'} \leftarrow \Omega^{'} + \frac{\mathbb{1} - \Omega^{'} v}{v^T v} v^T $$

    def constraint_subspace(omega, signature_probs):
        return torch.matmul(omega, signature_probs) - torch.ones(signature_probs.size(0))

    ##### 5.6) Compute bound using method
    def compute_bound(lambd, signature_probs, beta, projection_matrix, budget):
        # \TODO optimize memory w.r.t. projection_matrix
        value_matrix = beta - lambd * projection_matrix[:, None]

        # Take the max over the computed value_matrix
        max_values = value_matrix.max(0).values

        # Compute the outer_sum using vectorized operations
        outer_sum = torch.sum(signature_probs * max_values)
        outer_sum += lambd * budget

        return torch.sqrt(outer_sum)

    ##### 5.7) Project $\Omega$ to $M_{+}$

    # To ensure that we only have positive values in $\Omega$, what we currently do is: in each step of the gradient
    # descent, we check whether this is true. If not, we replace the negative value for a random value
    # between $0$ and $0.1$.


    ##### 5.8) Gradient descent
    def gradient_descent(
        lambd,
        signature_probas,
        beta,
        projection_matrix,
        budget,
        lr=0.001,  # Learning rate
        num_iterations=100,  # Number of iterations
        epsilon=1e-8  # Tolerance for early stopping
    ):

        torch.autograd.set_detect_anomaly(True)

        # Initialize the Adam optimizer
        optimizer = torch.optim.Adam([lambd], lr = lr)

        # Store losses for tracking the optimization progress
        loss_history = []

        for iteration in range(num_iterations):
            optimizer.zero_grad()  # Reset gradients to zero before backpropagation

            result = compute_bound(lambd, signature_probas, beta, projection_matrix, budget)
            result.backward()

            # Perform an optimization step (gradient descent step)
            optimizer.step()

            # Optionally track the loss (objective function value)
            loss_history.append(result.item())

            # Optional: Print progress
            if iteration % 500 == 0:
                print(f"Iteration {iteration}/{num_iterations}, Bound: {result.item()}")

            # Check for convergence (early stopping)
            if len(loss_history) > 1 and abs(loss_history[-1] - loss_history[-2]) < epsilon:
                print("Converged after {} iterations.".format(iteration))
                break

        return lambd, loss_history

    print('--------------BOUND (I)--------------')
    lambd = torch.tensor(0.1, requires_grad=True)

    optimized_lambda, losses = gradient_descent(
        lambd=lambd,
        signature_probas=initial_signature_probs,
        beta=beta,
        projection_matrix=projection_matrix,
        budget=wasserstein_squared_zero,
        lr=0.001,
        num_iterations=300) # 3000

    print(f"Optimized lambda: {optimized_lambda}")
    bound_continuous_discrete = compute_bound(optimized_lambda, initial_signature_probs, beta, projection_matrix, wasserstein_squared_zero)
    print(f"Bound (I): {bound_continuous_discrete.item():.4f}")

    print('--------------BOUND (II)--------------')

    # Assume other variables are given and fixed
    n = signatures.shape[-1]
    F = f.F(signatures).pow(2)
    pi_q = initial_signature_probs
    C = (signatures.view(1, -1) - signatures.view(-1, 1)).abs()
    w = 2*wasserstein_squared_zero

    # Reshape F, C, and Pi for linprog (they need to be 1D vectors)
    F_flat = F.flatten().numpy()
    C_flat = C.flatten().numpy()

    # Objective function is to maximize F * Pi, which is the same as minimizing -(F * Pi)
    c = -F_flat  # Minimizing -F is the same as maximizing F

    # Constraints:
    # Simplex constraint (Pi.sum() == 1): equality constraint
    A_eq = np.ones((1, n * n))  # Sum of all elements in Pi should be 1
    b_eq = [1]

    # Marginal equality constraint: Pi.sum(dim=0) == pi_q
    A_marg = np.zeros((n, n * n))
    for i in range(n):
        A_marg[i, i::n] = 1  # Select rows corresponding to each column sum
    b_marg = pi_q.numpy()

    # Wasserstein constraint: (C * Pi).sum() <= w
    A_ineq = np.array([C_flat])  # One inequality constraint for the Wasserstein bound
    b_ineq = [w]

    # Combine constraints
    A_eq_combined = np.vstack([A_eq, A_marg])  # Combine the equality constraints
    b_eq_combined = np.hstack([b_eq, b_marg])

    # Bounds for each element of Pi: 0 <= Pi <= infinity (non-negative)
    bounds = [(0, None)] * (n * n)

    # Solve the linear program
    result = linprog(c, A_ub=A_ineq, b_ub=b_ineq, A_eq=A_eq_combined, b_eq=b_eq_combined, bounds=bounds, method='highs')

    # Check if the solution was successful
    if result.success:
        Pi_optimized = result.x.reshape(n, n)
        # print("Optimized Pi:")
        # print(Pi_optimized)
        bound_discrete_discrete = (F * Pi_optimized).sum()
        print(f"Bound (I): {bound_discrete_discrete.item():.4f}")
    else:
        print("Optimization failed:", result.message)

    print('--------------FINAL RESULTS--------------')
    final_bound = bound_continuous_discrete.item() + bound_discrete_discrete.item()
    print(f"Final bound: {final_bound:.4f}")


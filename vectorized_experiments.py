import math
import matplotlib.pyplot as plt
import ot
import torch


class _Dynamics:
    def interval_approximation(self, region_1: torch.Tensor, region_2: torch.Tensor):
        # TODO</span>: Quick fix. We need to make sure only to compare f(region_1) to f(region_2)

        zero_tensor = torch.tensor([0])

        points_1 = region_1.clone()
        points_2 = region_2.clone()

        if (region_1[0] <= 0) & (region_1[1] >= 0):
            points_1 = torch.cat((region_1, zero_tensor))
        if (region_2[0] <= 0) & (region_2[1] >= 0):
            points_2 = torch.cat((region_2, zero_tensor))

        # Evaluate the dynamics at all boundary points
        values_1 = self(points_1)
        values_2 = self(points_2)

        # Compute the maximum absolute difference between all pairs
        max_value = (torch.max(torch.abs(values_1.unsqueeze(0) - values_2.unsqueeze(1))))
        return max_value.item()

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

    n_signatures = 20
    inner_edges = torch.linspace(start=initial_distribution.mean - 3 * initial_distribution.stddev,
                           end=initial_distribution.mean + 3 * initial_distribution.stddev,
                           steps=n_signatures-1)
    #inner_edges = torch.Tensor([])
    edges = torch.cat((-torch.tensor(torch.inf).view(1), inner_edges, torch.tensor(torch.inf).view(1)))
    print(f"Edges: {edges}")

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

    beta = torch.zeros(n_signatures, n_signatures)

    # \TODO parrallelize this loop..
    for i in range(n_signatures):
        for j in range(n_signatures):
            interval_approx = f.interval_approximation(regions[i], regions[j])
            beta[i][j] = interval_approx ** 2

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

    ##### 5.4) Compute constant term

    def compute_constant_term(beta):

        return beta.diagonal()


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
        constant_term = compute_constant_term(beta)
        value_matrix = constant_term - lambd * projection_matrix[:, None]

        # Take the max over the computed value_matrix
        max_values = value_matrix.max(0).values

        # Compute the outer_sum using vectorized operations
        outer_sum = torch.sum(signature_probs * max_values)
        outer_sum += lambd * budget

        return torch.sqrt(outer_sum)


    ##### 5.7) Project $\Omega$ to $M_{+}$

    # To ensure that we only have positive values in $\Omega$, what we currently do is: in each step of the gradient descent, we check whether this is true. If not, we replace the negative value for a random value between $0$ and $0.1$.

    # TODO: WE ARE NOT USING THIS IN THE GRADIENT DESCENT. OUR METHOD IS CURRENTLY ALLOWING FOR NEGATIVE TERMS. WE NEED TO FIX THIS.

    # TODO: Steven gave the idea of optimizing for the squared values.

    def compute_lower_probas(signature_probs, signatures, regions, wass_budget):

        distances_right = regions[:, 1] - signatures
        distances_left = signatures - regions[:, 0]
        distances_signature_to_extremity = torch.minimum(distances_left, distances_right)

        probs_to_be_moved = wass_budget / (distances_signature_to_extremity ** 2)

        inf_bounds = torch.maximum(torch.zeros_like(signature_probs), signature_probs - probs_to_be_moved)

        return inf_bounds


    def compute_upper_probas(signature_probs, signatures, regions, wass_budget):

        sup_bounds = []

        new_states = signatures.clone()
        new_probas = signature_probs.clone()

        k = 0
        for signature, region, original_proba in zip(new_states, regions, new_probas):

            sup_distance = 0
            sup_bound = 0

            sorted_indices = torch.argsort(abs(new_states - signature))  # Sort by difference to the signature
            sorted_states = new_states[sorted_indices]
            sorted_probas = new_probas[sorted_indices]

            for state, proba in zip(sorted_states, sorted_probas):

                if state == signature:
                    sup_bound += original_proba
                else:
                    qt = proba * min(abs(state - region[0]), abs(state - region[1])) ** 2
                    if sup_distance + qt < wass_budget:
                        sup_distance += qt
                        sup_bound += proba
                    else:
                        delta = (wass_budget - sup_distance) / min(abs(state - region[0]), abs(state - region[1])) ** 2
                        sup_bound += delta
                        break

            sup_bounds.append(sup_bound.item())

        return torch.Tensor(sup_bounds)

    ##### 5.8) Gradient descent

    #TODO: CODE TEST
    lower = compute_lower_probas(initial_signature_probs, signatures, regions, wasserstein_squared_zero)
    upper = compute_upper_probas(initial_signature_probs, signatures, regions, wasserstein_squared_zero)


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



    def compute_discrete_to_discrete_upper_bound(signatures, signature_probas):

        f_signatures = f(signatures)
        f_distance_signatures = f_signatures.unsqueeze(1) - f_signatures.unsqueeze(0)

        max_values, _ = torch.max(f_distance_signatures ** 2, dim=1)
        bound = torch.sum(signature_probas * max_values)

        return torch.sqrt(bound)



    print(f"Signature probas: {initial_signature_probs}")
    print(f"Lower probas: {lower}")
    print(f"Upper probas: {upper}")


    print('--------------BOUND (I)--------------')
    lambd = torch.tensor(0.1, requires_grad=True)

    optimized_lambda, losses = gradient_descent(
        lambd=lambd,
        signature_probas=initial_signature_probs,
        beta=beta,
        projection_matrix=projection_matrix,
        budget=wasserstein_squared_zero,
        lr=0.001,
        num_iterations=3000)

    print(f"Optimized lambda: {optimized_lambda}")
    bound_continuous_discrete = compute_bound(optimized_lambda, initial_signature_probs, beta, projection_matrix, wasserstein_squared_zero)
    print(f"Bound (I): {bound_continuous_discrete.item():.4f}")


    print('--------------BOUND (II)--------------')
    bound_discrete_discrete = compute_discrete_to_discrete_upper_bound(signatures, initial_signature_probs)
    print(f"Bound (II): {bound_discrete_discrete:.4f}")


    print('--------------FINAL RESULTS--------------')
    final_bound = bound_continuous_discrete.item() + bound_discrete_discrete.item()
    print(f"Final bound: {final_bound:.4f}")


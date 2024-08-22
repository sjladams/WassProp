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
    torch.manual_seed(10) # for reproducibility

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

    #TODO: CHECK IF IT SHOULD BE THE TRANSPOSE
    #projection_matrix = torch.transpose(projection_matrix, 0, 1)

    ##### 5.4) Compute constant term
    # This term is equivalent to $\sum_{m=1}^N\Tilde{\beta}_{km}(\omega_{km}, \omega_{lm}}))$ in Corollary 9

    def compute_constant_term(omega, signature_probas, beta):
        # Expand omega to create pairwise combinations
        omega_k = omega[:, None]
        omega_l = omega[None, :]

        # Compute the value for each (k, l) pair using broadcasting
        # beta[:, None] expands beta to match the shape (num_regions, 1) for broadcasting
        value = torch.sum(omega_k * signature_probas * beta[:, None] * (2 - omega_l), dim=-1)

        return value


    ##### 5.5) Project matrix on subspace
    # Our matrix $\Omega$ needs to satisfy $\sum_{j} \omega_{ij} \hat{p}_j = 1$. Given a matrix $\Omega^{'}$ that may
    # disrespect this restriction, we enforce it by making the following adjustment (denote by $\hat{p}$ the vector
    # with all $\hat{p}_j$):

    # $$ \Omega^{'} \leftarrow \Omega^{'} + \frac{\mathbb{1} - \Omega^{'} v}{v^T v} v^T $$

    def projection_on_subspace(omega, proba_vector):

        if omega.dim() == 1:
            omega = omega.unsqueeze(0)

        if omega.dtype != proba_vector.dtype:
            # Convert both tensors to float32 if their types don't match
            omega = omega.to(torch.float32)
            proba_vector = proba_vector.to(torch.float32)

        ones = torch.ones(omega.size(0), dtype=torch.float32)
        factor = torch.dot(proba_vector, proba_vector)
        omega_dot_proba = torch.matmul(omega, proba_vector)
        c = (ones - omega_dot_proba) / factor

        projection = omega + (c.unsqueeze(1) * proba_vector.unsqueeze(0))

        return projection

    def constraint_subspace(omega, signature_probs):
        return torch.matmul(omega, signature_probs) - torch.ones(signature_probs.size(0))

    ##### 5.6) Compute bound using method
    def compute_bound(lambd, omega, signature_probs, beta, projection_matrix, budget):
        # \TODO optimize memory w.r.t. projection_matrix
        constant_term = compute_constant_term(omega, signature_probs, beta)
        value_matrix = constant_term[:, :, None, None] - lambd * (
                projection_matrix[:, None, :, None] + projection_matrix[None, :, None, :])

        # Take the max over the computed value_matrix
        max_values = value_matrix.max(0).values.max(0).values

        #print("MAX VALUES")
        #print(max_values)

        # Compute the outer_sum using vectorized operations
        outer_sum = torch.sum(signature_probs[:, None] * signature_probs[None, :] * max_values)
        outer_sum += lambd * 2 * budget

        #Avoid negative values (as our GD does not take (39) into account)
        outer_sum = torch.clamp(outer_sum, min = 0)

        return torch.sqrt(outer_sum)

    ##### 5.7) Project $\Omega$ to $M_{+}$

    # To ensure that we only have positive values in $\Omega$, what we currently do is: in each step of the gradient descent, we check whether this is true. If not, we replace the negative value for a random value between $0$ and $0.1$.

    # TODO: WE ARE NOT USING THIS IN THE GRADIENT DESCENT. OUR METHOD IS CURRENTLY ALLOWING FOR NEGATIVE TERMS. WE NEED TO FIX THIS.

    # TODO: Steven gave the idea of optimizing for the squared values.

    ##### 5.8) Gradient descent

    def gradient_descent(
        lambd,
        alpha,
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
        optimizer = torch.optim.Adam([lambd, alpha], lr = lr)

        # Store losses for tracking the optimization progress
        loss_history = []

        for iteration in range(num_iterations):
            optimizer.zero_grad()  # Reset gradients to zero before backpropagation

            # Perform the forward and backward passes to compute the gradients
            omega = torch.exp(alpha)

            constraint = constraint_subspace(omega, signature_probas)
            penalty = torch.sum(constraint ** 2)

            bound = compute_bound(lambd, omega, signature_probas, beta, projection_matrix, budget)

            result = bound + penalty
            result.backward()

            # Modify gradients before the optimizer step (only for omega)
            #with torch.no_grad():
            #    for param in [omega]:
            #        if param.grad is not None:
            #            # Project the gradient onto the tangent space defined by Omega v = 1
            #            grad_projection = param.grad - (param.grad @ signature_probas) * signature_probas / signature_probas.dot(signature_probas)
            #            param.grad = grad_projection

            # Perform an optimization step (gradient descent step)
            optimizer.step()

            #with torch.no_grad():
            #    omega = projection_on_subspace(omega, signature_probas)

            # Optionally track the loss (objective function value)
            loss_history.append(result.item())

            # Optional: Print progress
            if iteration % 20 == 0:
                print(f"Iteration {iteration}/{num_iterations}, Bound: {result.item()}")

            # Check for convergence (early stopping)
            if len(loss_history) > 1 and abs(loss_history[-1] - loss_history[-2]) < epsilon:
                print("Converged after {} iterations.".format(iteration))
                break

        #Project omega on subspace (39)
        #with torch.no_grad():
        #    omega = torch.exp(alpha)
        #    omega = projection_on_subspace(omega, signature_probas)

        return lambd, omega, loss_history


    lambd = torch.tensor(0.1, requires_grad=True)

    alpha = 0.1 * torch.randn(n_signatures, n_signatures)
    alpha.requires_grad_()

    #omega = torch.ones(n_signatures, n_signatures, requires_grad=True)
    #omega = torch.rand(n_signatures, n_signatures)
    #omega = torch.diag(1.0 / initial_signature_probs)
    #omega = projection_on_subspace(omega, initial_signature_probs)
    #omega.requires_grad_()
    #print(torch.matmul(omega, initial_signature_probs))

    # Perform gradient descent using Adam
    optimized_lambda, optimized_omega, losses = gradient_descent(
        lambd=lambd,
        alpha=alpha,
        signature_probas=initial_signature_probs,
        beta=beta,
        projection_matrix=projection_matrix,
        budget=wasserstein_squared_zero,
        lr=0.01,
        num_iterations=1000
    )


    print('--------------RESULTS--------------')
    print(f"Optimized lambda: {optimized_lambda}")

    print(f"Optimized omega: {optimized_omega}")
    print(torch.matmul(optimized_omega, initial_signature_probs))

    optimized_omega = projection_on_subspace(optimized_omega, initial_signature_probs)
    print(torch.matmul(optimized_omega, initial_signature_probs))

    bound = compute_bound(optimized_lambda, optimized_omega, initial_signature_probs, beta, projection_matrix, wasserstein_squared_zero)
    print(f"Final bound: {bound.item()}")



    # f_old = result.item()
    # print(f_old)
    #
    # def objective(f_old, lambd, alpha_proj, signature_probs, beta, projection_matrix, budget):
    #
    #     f_old = f_old
    #
    #     omega_proj = torch.exp(alpha_proj)
    #     f_new = compute_bound(lambd, omega_proj, signature_probs, beta, projection_matrix, budget)
    #
    #     return torch.norm(f_new - f_old) ** 2
    #
    #
    # def constraint(alpha_proj, signature_probs):
    #     omega_proj = torch.exp(alpha_proj)
    #     return torch.matmul(omega_proj, signature_probs) - torch.ones(signature_probs.size(0))
    #
    #
    # import torch.optim as optim
    # # Initialize Omega_new as a tensor with requires_grad=True for optimization
    # #omega_proj = optimized_omega.detach().clone() # Flatten for optimization
    # #omega_proj.requires_grad_()
    # alpha_proj = 0.1 * torch.randn(n_signatures, n_signatures)
    # alpha_proj.requires_grad_()
    #
    # # Define optimizer
    # optimizer = optim.Adam([alpha_proj], lr=0.01)  # You can use other optimizers if preferred
    #
    # # Training loop
    # for epoch in range(10000):  # Number of epochs
    #     optimizer.zero_grad()
    #
    #     # Calculate objective
    #     loss = objective(f_old, lambd, alpha_proj, initial_signature_probs, beta, projection_matrix, wasserstein_squared_zero)
    #
    #     # Calculate constraints
    #     con = constraint(alpha_proj, initial_signature_probs)
    #
    #     # Add constraint penalty (if any)
    #     penalty = torch.sum(con ** 2)  # Penalty for violation of equality constraint
    #
    #     # Total loss with penalty
    #     total_loss = loss + penalty
    #
    #     # Backward pass and optimization
    #     total_loss.backward()
    #     optimizer.step()
    #
    #     # Print loss every 100 iterations
    #     if epoch % 100 == 0:
    #         print(f'Epoch [{epoch + 1}/1000], Loss: {total_loss.item()}')
    #
    # # Reshape Omega_new to the original shape
    # #omega_proj = omega_proj.detach().view(optimized_omega.shape)
    # print("New omega:")
    # print(torch.exp(alpha_proj))
    #
    # omega_final = torch.exp(alpha_proj)
    # print(torch.matmul(omega_final, initial_signature_probs))
    #
    # result = compute_bound(optimized_lambda, omega_final, initial_signature_probs, beta, projection_matrix,
    #                        wasserstein_squared_zero)
    # print(f"Final bound: {result.item()}")



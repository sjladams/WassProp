import ot
import torch
import matplotlib.pyplot as plt

import dynamics
import bounds
import gradient_descent as desc

torch.manual_seed(0) # for reproducibility

### 1) Define dynamics
n_dims = 1
f = dynamics.GaussianDynamics(mu=0, sigma=1)

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
initial_distribution = torch.distributions.Normal(loc=1, scale=1)
initial_distribution_samples = initial_distribution.sample(sample_shape=torch.Size((n_samples,)))

##### 2.2) Noise structure

mean_noise = 0
std_dev_noise = 0.3
noise_samples = torch.normal(mean=mean_noise, std=std_dev_noise, size=initial_distribution_samples.shape)

##### 2.3) Monte Carlo simulation of the system
propagated_states = f(initial_distribution_samples) + noise_samples

fig_histograms = plt.figure()
plt.hist(initial_distribution_samples, alpha=0.5, label='t = 0', bins=100, density=True)
plt.hist(propagated_states, alpha=0.5, label='t = 1', bins=100, density=True)
plt.legend()
plt.title("Histograms of the initial and propagated states")
plt.show()

### 3) Create signature approximation for $\mathbb{P}_0$
n_signatures = 10
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
wasserstein_squared_zero = ot.solve_sample(X_a=initial_distribution_samples.view(-1, n_dims),
                                               X_b=signatures.view(-1, n_dims), b=initial_signature_probs).value
print(f"2-W distance between the true P_0 and our signature approximation {wasserstein_squared_zero.sqrt():.4f}")

##### 3.3) Compute $\mathbb{W}_{2}(\mathbb{P}_1, \hat{\mathbb{P}}_1)$


def sample_gmm(centers: torch.Tensor, probabilities: torch.Tensor, st_dev: float, n_samples: int=1):
    #Choose a component based on the provided probabilities
    chosen_components = torch.multinomial(probabilities, n_samples, replacement=True)

    #Sample from the chosen Gaussian component
    return torch.normal(mean=centers[chosen_components], std=st_dev)

centers = f(signatures)
samples_gmm = sample_gmm(centers, initial_signature_probs, std_dev_noise, n_samples=1000)

wasserstein_squared_propagation = ot.solve_sample(samples_gmm.view(-1, n_dims), propagated_states.view(-1, n_dims)).value
print(f"wasserstein distance after propagation: {wasserstein_squared_propagation.sqrt():.4f}")


### 4) Propagating with global Lipschitz
print(f"Bound using Global Lipschitz constant: {f.global_lipschitz() * wasserstein_squared_zero.sqrt():.4f}")

### 5) Propagating with our method
regions = torch.vstack((edges[:-1], edges[1:])).swapaxes(0, 1)



beta = torch.zeros(n_signatures)

for i in range(n_signatures):
    interval_approx = f.interval_approximation(regions[i])
    beta[i] = interval_approx ** 2

##### 5.3) Compute projection matrix

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





print(f"Signature probas: {initial_signature_probs}")


print('--------------BOUND (I)--------------')
lambd = torch.tensor(0.01, requires_grad=True)

optimized_lambda, losses = desc.gradient_descent(
    lambd=lambd,
    signature_probas=initial_signature_probs,
    beta=beta,
    projection_matrix=projection_matrix,
    budget=wasserstein_squared_zero,
    lr=0.001,
    num_iterations=3000)

print(f"Optimized lambda: {optimized_lambda}")
bound_continuous_discrete = bounds.compute_bound(optimized_lambda, initial_signature_probs, beta, projection_matrix, wasserstein_squared_zero)
print(f"Bound (I): {bound_continuous_discrete.item():.4f}")
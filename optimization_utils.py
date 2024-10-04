import torch
from typing import Callable

def optimize_with_adam(objective: Callable,param: torch.Tensor, lr=0.001, num_iterations=100, tolerance=1e-8, **kwargs):
    """

    :param objective:
    :param param:
    :param lr:
    :param num_iterations:
    :param tolerance:
    :return:
    """
    torch.autograd.set_detect_anomaly(True)

    # Initialize the Adam optimizer
    optimizer = torch.optim.Adam([param], lr=lr)

    # Store losses for tracking the optimization progress
    loss_history = []

    for iteration in range(num_iterations):
        optimizer.zero_grad()  # Reset gradients to zero before backpropagation

        result = objective(param, **kwargs)
        result.backward()

        # Perform an optimization step (gradient descent step)
        optimizer.step()

        # Optionally track the loss (objective function value)
        loss_history.append(result.item())

        # Optional: Print progress
        if iteration % 500 == 0:
            print(f"Iteration {iteration}/{num_iterations}, Bound: {result.item()}")

        # Check for convergence (early stopping)
        if len(loss_history) > 1 and abs(loss_history[-1] - loss_history[-2]) < tolerance:
            print("Converged after {} iterations.".format(iteration))
            break

    return param, loss_history
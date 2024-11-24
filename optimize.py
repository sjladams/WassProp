import torch
from typing import Callable

def minimize_with_adam(objective: Callable,
                       param: torch.Tensor,
                       lr=0.01, num_iterations=100, tolerance=1e-8,
                       print_progress: bool = True,
                       non_negative_constraint: bool = False, min_value: torch.Tensor = None,
                       **kwargs):
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

    if print_progress:
        print("\n Starting optimization...")

    for iteration in range(num_iterations):
        optimizer.zero_grad()  # Reset gradients to zero before backpropagation

        loss = objective(param, **kwargs)
        loss.backward(retain_graph=True)

        # Perform an optimization step (gradient descent step)
        optimizer.step()

        if non_negative_constraint:
            # Projection step to ensure param >= 0
            with torch.no_grad():
                if min_value is None:
                    param.clamp_(min=0)
                else:
                    param.clamp_(min=min_value.item())

        # Optionally track the loss (objective function value)
        loss_history.append(loss.item())

        # Optional: Print progress
        if (iteration % 500 == 0 or iteration == num_iterations - 1) and print_progress:
            print(f"Iteration {iteration}/{num_iterations}, Bound: {loss.item()}")

        # Check for convergence (early stopping)
        if len(loss_history) > 1 and abs(loss_history[-1] - loss_history[-2]) < tolerance and print_progress:
            print("Converged after {} iterations.".format(iteration))
            break

    if print_progress and param.size() == torch.Size():
        print(f"Optimal param: {param:.4f} \n")

    return param, loss_history
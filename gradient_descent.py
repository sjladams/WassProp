import torch
import bounds

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
    optimizer = torch.optim.Adam([lambd], lr=lr)

    # Store losses for tracking the optimization progress
    loss_history = []

    for iteration in range(num_iterations):
        optimizer.zero_grad()  # Reset gradients to zero before backpropagation

        result = bounds.compute_bound(lambd, signature_probas, beta, projection_matrix, budget)
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
import numpy as np
from scipy.optimize import minimize

# Define the objective function
def objective_function(omega, source_density, target_density, source_positions, target_positions):
    # Compute the objective function based on Equation 3
    f = np.sum(omega * source_density) - np.sum(target_density * np.linalg.norm(source_positions - target_positions, axis=1))
    return f

# Define the gradient of the objective function
def gradient(omega, source_density, target_density, source_positions, target_positions):
    # Compute the gradient based on Equation 4
    grad = source_density - target_density
    return grad

# Define the optimization function
def optimize_ot(source_density, target_density, source_positions, target_positions):
    # Initial guess for weights omega
    initial_guess = np.ones(len(source_density))

    # Define the objective function to minimize
    obj_func = lambda omega: objective_function(omega, source_density, target_density, source_positions, target_positions)

    # Define the gradient function
    grad_func = lambda omega: gradient(omega, source_density, target_density, source_positions, target_positions)

    # Minimize the objective function using L-BFGS-B method
    result = minimize(obj_func, initial_guess, jac=grad_func, method='L-BFGS-B')

    # Extract the optimal weights
    optimal_weights = result.x
    return optimal_weights

# Example usage
if __name__ == "__main__":
    # Example data (dummy data)
    source_density = np.array([0.2, 0.3, 0.5])
    target_density = np.array([0.4, 0.4, 0.2])

    source_positions = np.array([[1, 2], [3, 4], [5, 6]])
    target_positions = np.array([[2, 3], [4, 5], [6, 7]])

    # Optimize the optimal transport problem
    optimal_weights = optimize_ot(source_density, target_density, source_positions, target_positions)

    print("Optimal weights:", optimal_weights)

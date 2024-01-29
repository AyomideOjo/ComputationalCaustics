import numpy as np
import matplotlib.pyplot as plt
import ot

# Generate or obtain source and target Voronoi diagrams
# For demonstration purposes, let's create random points
num_points = 100
source_points = np.random.rand(num_points, 2)  # Source Voronoi diagram points
target_points = np.random.rand(num_points, 2)  # Target Voronoi diagram points

# Compute cost matrix (Euclidean distance in this case)
cost_matrix = ot.dist(source_points, target_points)

# Define source and target distributions (uniform for demonstration)
source_distribution = np.ones(num_points) / num_points
target_distribution = np.ones(num_points) / num_points

# Solve the optimal transport problem using POT library
optimal_transport_plan = ot.emd(source_distribution, target_distribution, cost_matrix)

print(optimal_transport_plan)

# Visualize the optimal transport plan
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(source_points[:, 0], source_points[:, 1], c='b', label='Source')
plt.scatter(target_points[:, 0], target_points[:, 1], c='r', label='Target')
plt.title('Source and Target Voronoi Diagrams')
plt.legend()

plt.subplot(1, 2, 2)
plt.imshow(optimal_transport_plan, cmap='Blues')
plt.title('Optimal Transport Plan')
plt.colorbar(label='Transport amount')
plt.tight_layout()
plt.show()

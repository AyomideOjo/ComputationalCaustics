import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_modified_mesh(num_points_x=10, num_points_y=10):
    # Create a grid of points
    x = np.linspace(0, 1, num_points_x)
    y = np.linspace(0, 1, num_points_y)
    X, Y = np.meshgrid(x, y)

    # Flatten the grid to create vertices
    vertices = np.column_stack([X.flatten(), Y.flatten(), np.zeros(num_points_x * num_points_y)])

    # Generate triangular faces
    faces = []
    for i in range(num_points_y - 1):
        for j in range(num_points_x - 1):
            k1 = i * num_points_x + j
            k2 = k1 + num_points_x
            faces.append([k1, k2, k1 + 1])
            faces.append([k1 + 1, k2, k2 + 1])

    # Convert faces to numpy array
    faces = np.array(faces)

    # Plot the mesh with the modified point
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot triangular mesh
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Triangular Mesh')

    # Show the plot
    plt.show()

# Example usage:
plot_modified_mesh(num_points_x = 50, num_points_y = 50)

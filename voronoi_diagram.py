#https://stackoverflow.com/questions/48334853/using-pycharm-i-want-to-show-plot-extra-figure-windows
# TODO: separate sections of project into different files for better code separation
import numpy as np

from scipy.optimize import minimize, optimize
from scipy.spatial import Delaunay, cKDTree
from PIL import Image
import matplotlib.pyplot as plt

import os

def calculate_integration_energy(vertices, target_normals):
    """Computes the integration energy based on the vertex normals of the target surface
    and the desired normals derived from the optimal transport map (OTM)."""
    integration_energy = np.sum(np.linalg.norm(vertices[:, 1:4] - target_normals, axis=1) ** 2)
    return integration_energy

def calculate_direction_energy(vertices, source_positions, incoming_ray_directions):
    """Computes the direction energy to ensure that the vertices of the target surface
    do not deviate too much from the incoming ray direction."""
    direction_energy = np.sum(np.linalg.norm(vertices[:, 0] - np.dot(incoming_ray_directions, (source_positions - vertices[:, 0]).T).T) ** 2)
    return direction_energy

def calculate_flux_preservation_energy(target_triangles, source_triangles):
    """Computes the flux preservation energy to ensure that the flux over each triangle
    of the target surface remains constant, as computed by the OTM."""
    # TODO: find area of triangles and use that as radient flux equation
    flux_preservation_energy = np.sum(np.linalg.norm(target_triangles - source_triangles) ** 2)
    return flux_preservation_energy

def calculate_regularization_energy(vertices):
    """Computes the regularization energy to maintain well-shaped triangles on the target surface."""
    regularization_energy = np.sum(np.linalg.norm(vertices) ** 2)
    return regularization_energy

def calculate_barrier_energy(vertices, receiver_normal, receiver_position, threshold_distance):
    """Computes the barrier energy to prevent the target surface from falling beyond a certain
    distance from the receiver plane."""
    barrier_energy = np.sum(np.linalg.norm(np.maximum(0, -np.log((1 - vertices[:, 1]) + threshold_distance)) ** 2))
    return barrier_energy

def minimize_energy(vertices, source_positions, incoming_ray_directions, target_triangles, source_triangles,
                    target_normals, receiver_normal, receiver_position, threshold_distance, weights):
    """Minimizes the compound energy function using the provided weighting vector."""
    # TODO: Change gradient descent algorithm from L-BFGS-B to LM
    def energy_function(vertices):
        # Calculate individual energy terms
        integration_energy = calculate_integration_energy(vertices, target_normals)
        direction_energy = calculate_direction_energy(vertices, source_positions, incoming_ray_directions)
        flux_preservation_energy = calculate_flux_preservation_energy(target_triangles, source_triangles)
        regularization_energy = calculate_regularization_energy(vertices)
        barrier_energy = calculate_barrier_energy(vertices, receiver_normal, receiver_position, threshold_distance)

        # Compute the compound energy
        compound_energy = np.dot(weights, [integration_energy,
            direction_energy, flux_preservation_energy,
            regularization_energy, barrier_energy])
        return compound_energy

    # Perform minimization
    result = optimize.minimize(energy_function, vertices, method='L-BFGS-B')
    # result = least_squares(objective_function, X_initial_guess, method='lm')
    return result.x

def optimal_transport_map_interpolation(source_mesh_vertices, optimal_transport_map):
    # TODO: use this function to call minimize_energy function above
    pass

def normalization(matrix, type_of_normalization="Frobenius_norm"):
    """
    Perform Matrix Normalization in accordance with the value of the parameter 'type_of_normalization'.
    """
    if type_of_normalization == "L1_norm":
        return np.linalg.norm(matrix, ord=1)
    if type_of_normalization == "L2_norm":
        return np.linalg.norm(matrix, ord='fro')
    if type_of_normalization == "Frobenius_norm":
        return np.linalg.norm(matrix)
    else:
        raise ValueError("Wrong value for "
                         "'type_of_normalization' variable")

def fresnel_mapping(source_mesh_vertices, normalized_difference):
    # TODO: calculate normals using this function
    pass

def normal_integration(source_mesh_vertices, normalal_thelda):
    # TODO: calculate height field using this function
    pass

def target_optimization(source_mesh_vertices, optimal_transport_map):
    """
    Performs Target Optimization of mesh as defined in the Paper
    """
    target_mesh_vertices = optimal_transport_map_interpolation(source_mesh_vertices, optimal_transport_map)
    difference = np.sum(target_mesh_vertices - source_mesh_vertices)
    tolerance = 1e-6
    while difference > tolerance:
        difference = np.sum(target_mesh_vertices - source_mesh_vertices)
        normalized_difference = normalization(difference)
        normalal_thelda = fresnel_mapping(source_mesh_vertices, normalized_difference)
        integrated_normal = normal_integration(source_mesh_vertices, normalal_thelda)
        source_mesh_vertices = integrated_normal
    return source_mesh_vertices

def plot_gaussian_distribution(middle, scaler = 100):
    """
    Get 2D Gaussian distribution as the source image
    """
    x_range = (-middle, middle)
    y_range = (-middle, middle)
    num_points = 1000
    scale = num_points * scaler

    mean = np.array([0, 0])
    covariance = np.array([[scale, 0], [0, scale]])

    # Generate x and y coordinates
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    # Compute the height values of the Gaussian distribution at each point
    heights = np.exp(-0.5 * np.einsum('ijk,kl,ijl->ij', pos - mean, np.linalg.inv(covariance), pos - mean)) \
              / (2 * np.pi * np.sqrt(np.linalg.det(covariance)))

    # Compute the minimum and maximum heights
    min_height = np.min(heights)
    max_height = np.max(heights)

    # Scale the heights to be between 0 and 255
    scaled_heights = ((heights - min_height) / (max_height - min_height)) * 255

    # Display the image of the Gaussian distribution with scaled heights
    plt.imshow(scaled_heights, cmap='viridis', origin='lower', extent=(x_range[0], x_range[1], y_range[0], y_range[1]))
    plt.colorbar(label='Height')
    plt.title('Gaussian Distribution (Scaled)')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(False)
    plt.show()

    return scaled_heights

def compute_voronoi_diagram(Adata, number_of_dots):
    """
    Compute Voronoi diagram based on input data.

    Parameters:
    - Adata: numpy array, 2D array representing the input data.
    - number_of_dots: int, number of points to use in generating the Voronoi diagram.

    Returns:
    - p: numpy array, coordinates of the points used in the Voronoi diagram.
    - density_total: numpy array, total density calculated during the iterations.
    """
    n, m = Adata.shape
    x, y = np.meshgrid(np.arange(1, m + 1), np.arange(1, n + 1))
    gp = np.column_stack((x.flatten(), y.flatten()))

    A = 1.001 - Adata.flatten() / 255.0
    N = number_of_dots

    seedp = gp[A > 0.5, :]
    ix = np.random.permutation(seedp.shape[0])[:N]
    p = seedp[ix, :]

    dt = Delaunay(p)
    kdtree = cKDTree(p)
    ID = kdtree.query(gp)[1]

    vr = plt.imshow(ID.reshape(n, m), alpha=0.2)
    plt.plot(p[:, 0], p[:, 1], '.', markersize=8, color=[0, 0, 0])
    plt.axis('equal')
    plt.axis('tight')

    density_total = np.array([])
    maxiter = 30
    for it in range(maxiter):
        print("Step: {}".format(it + 1))
        cw = np.zeros((N, 2))
        density = np.array([])
        for i in range(N):
            mask = ID == i
            a = A[mask]
            b = gp[mask, :]
            cw[i, :] = np.mean(a[:, np.newaxis] * b, axis=0) / np.mean(a, axis=0)
            density = np.append(density, np.sum(a))

        p = cw
        dt = Delaunay(p)
        kdtree = cKDTree(p)
        ID = kdtree.query(gp)[1]

        vr = plt.imshow(ID.reshape(n, m), alpha=0.2)
        #plt.plot(p[:, 0], p[:, 1], '.', markersize=2, color=[0, 0, 0])
        #plt.draw(); plt.pause(0.01)
        density_total = density
    #plt.show()
    return p, density_total

def target_voronoi_diagram(imdata, number_of_dots):
    Adata = imdata[:, :, 2]  # red channel (arbitrarily)
    #plt.subplot(1, 2, 1); plt.imshow(imdata); plt.subplot(1, 2, 2)
    p, density_total = compute_voronoi_diagram(Adata, number_of_dots)

    np.save('targetVoronoiDiagram.npy', p)
    np.savetxt('targetVoronoiDensity.txt', density_total)
    return p

def source_voronoi_diagram(Adata, number_of_dots):
    #plt.subplot(1, 2, 1); plt.imshow(Adata); plt.subplot(1, 2, 2)
    p, density_total = compute_voronoi_diagram(Adata, number_of_dots)

    np.save('sourceVoronoiDiagram.npy', p)
    np.savetxt('sourceVoronoiDensity.txt', density_total)
    return p

# Define the objective function
def objective_function(omega, source_density, target_density, source_positions, target_positions):
    """Compute the objective function based on Equation 3"""
    f = np.sum(omega * source_density) - np.sum(target_density * np.linalg.norm(source_positions - target_positions, axis=1))
    return f

# Define the gradient of the objective function
def gradient(omega, source_density, target_density, source_positions, target_positions):
    """Compute the gradient based on Equation 4"""
    grad = source_density - target_density
    return grad

# Define the optimization function
def optimize_ot(source_density, target_density, source_positions, target_positions):
    """
    Optimize optimal transport with given densities and positions.

    Parameters:
    - source_density: numpy array, density of source points.
    - target_density: numpy array, density of target points.
    - source_positions: numpy array, positions of source points.
    - target_positions: numpy array, positions of target points.

    Returns:
    - optimal_weights: numpy array, optimal weights after optimization.
    """
    initial_guess = np.ones(len(source_density)) # Initial guess for weights omega
    obj_func = lambda omega: objective_function(omega, source_density, target_density, source_positions, target_positions) # Define the objective function to minimize
    grad_func = lambda omega: gradient(omega, source_density, target_density, source_positions, target_positions) # Define the gradient function

    # Minimize the objective function using L-BFGS-B method
    result = minimize(obj_func, initial_guess, jac = grad_func, method = 'L-BFGS-B')
    optimal_weights = result.x # Extract the optimal weights

    return optimal_weights

def plot_modified_mesh(num_points_x = 10, num_points_y = 10):
    """
    Plot a modified triangular mesh based on the given number of points along x and y axes.

    Parameters:
    - num_points_x: int, number of points along the x-axis.
    - num_points_y: int, number of points along the y-axis.

    Returns:
    - vertices: numpy array, vertices of the triangular mesh.
    - faces: numpy array, triangular faces of the mesh.
    """
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
    ax.set_xlabel('X-axis'); ax.set_ylabel('Y-axis'); ax.set_zlabel('Z-axis')
    ax.set_title('Triangular Mesh')

    plt.show() # Show the plot
    return vertices, faces

def main(image_path):
    """
    Main function to run the entire project
    """
    # Parameter Settings
    number_of_dots = 9000  # number of voronoi centroids
    mesh_resolution = np.array([50, 50])  # the number of triangles in our grid mesh

    file_A, file_B = 'targetVoronoiDiagram.npy', 'sourceVoronoiDiagram.npy'
    file_C, file_D = 'targetVoronoiDensity.txt', 'sourceVoronoiDensity.txt'

    # Functional Code
    if not (os.path.exists(file_A) and os.path.exists(file_B) and
            os.path.exists(file_C) and os.path.exists(file_D)):

        imdata = Image.open(image_path)
        imdata = np.array(imdata)
        imdata = 255 - imdata  # inverting the images

        coordinates = target_voronoi_diagram(imdata, number_of_dots)
        (x, y, z) = imdata.shape
        gaussian = plot_gaussian_distribution((x + y) / 2)
        gaussian = 255 - gaussian
        gaussian = source_voronoi_diagram(gaussian, number_of_dots)

    source_positions = np.load(file_A)
    source_positions *= [1, -1]
    target_positions = np.load(file_B)

    source_density, target_density = np.loadtxt(file_C), np.loadtxt(file_D)
    optimal_weights = optimize_ot(source_density, target_density, source_positions, target_positions)

    vertices_original, faces_original = \
        plot_modified_mesh(num_points_x=mesh_resolution[0],
                           num_points_y=mesh_resolution[1])

#image_path = r"C:\Users\Ayomide Enoch Ojo\Downloads\cat.png
image_path = r"C:\Users\Ayomide Enoch Ojo\PycharmProjects\Caustics\mlk_square.jpg"
main(image_path)
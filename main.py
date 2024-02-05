import numpy as np

def calculate_integration_energy(vertices, target_normals):
    """
    Computes the integration energy based on the vertex normals of the target surface
    and the desired normals derived from the optimal transport map (OTM).
    """
    integration_energy = np.sum(np.linalg.norm(vertices[:, 1:4] - target_normals, axis=1) ** 2)
    return integration_energy

def calculate_direction_energy(vertices, source_positions, incoming_ray_directions):
    """
    Computes the direction energy to ensure that the vertices of the target surface
    do not deviate too much from the incoming ray direction.
    """
    direction_energy = np.sum(np.linalg.norm(vertices[:, 0] - np.dot(incoming_ray_directions, (source_positions - vertices[:, 0]).T).T) ** 2)
    return direction_energy

def calculate_flux_preservation_energy(target_triangles, source_triangles):
    """
    Computes the flux preservation energy to ensure that the flux over each triangle
    of the target surface remains constant, as computed by the OTM.
    """
    flux_preservation_energy = np.sum(np.linalg.norm(target_triangles - source_triangles) ** 2)
    return flux_preservation_energy

def calculate_regularization_energy(vertices):
    """
    Computes the regularization energy to maintain well-shaped triangles on the target surface.
    """
    regularization_energy = np.sum(np.linalg.norm(vertices) ** 2)
    return regularization_energy

def calculate_barrier_energy(vertices, receiver_normal, receiver_position, threshold_distance):
    """
    Computes the barrier energy to prevent the target surface from falling beyond a certain
    distance from the receiver plane.
    """
    barrier_energy = np.sum(np.linalg.norm(np.maximum(0, -np.log((1 - vertices[:, 1]) + threshold_distance)) ** 2))
    return barrier_energy

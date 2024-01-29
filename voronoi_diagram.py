import numpy as np
from scipy.spatial import Delaunay, cKDTree
import matplotlib.pyplot as plt
from PIL import Image
import ot

def gaussian_filter(size, sigma):
    kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - (size-1)/2)**2 + (y - (size-1)/2)**2) / (2*sigma**2)), (size, size))
    kernel = kernel / np.sum(kernel)
    min_value, max_value = 0, 255
    kernel = kernel * (max_value - min_value) + min_value
    return np.round(kernel)

def voronoi_diagram(imdata, number_of_dots):
    # Load image
    Adata = imdata[:,  :, 2] # red channel (arbitrarily)
    plt.subplot(1, 2, 1)
    plt.imshow(imdata)
    plt.subplot(1, 2, 2)

    n, m = Adata.shape # n rows, c cols, i.e., y size then x size
    # meshgrid takes x,y size
    x, y = np.meshgrid(np.arange(1, m + 1), np.arange(1, n + 1))
    gp = np.column_stack((x.flatten(), y.flatten()))
    # pixel locations (grid points)
    # swap white and black, i.e., high weight for black points and make sure we
    # avoid division by zero by staying 1e-3 above 0

    # Swap white and black, avoid division by zero
    A = 1.001 - Adata.flatten() / 255.0
    N = number_of_dots

    # Seed points near dark (or bright) parts of the image
    seedp = gp[A > 0.5, :]
    ix = np.random.permutation(seedp.shape[0])[:N]
    p = seedp[ix, :]

    # Find ID for each pixel using nearest neighbor search
    dt = Delaunay(p)
    kdtree = cKDTree(p)
    ID = kdtree.query(gp)[1]

    # Plot Voronoi diagram
    vr = plt.imshow(ID.reshape(n, m), alpha=0.2)
    plt.plot(p[:, 0], p[:, 1], '.', markersize=8, color=[0, 0, 0])
    plt.axis('equal')
    plt.axis('tight')

    maxiter = 30
    for it in range(maxiter):
        print("Step: {}".format(it + 1))
        cw = np.zeros((N, 2))
        for i in range(N):
            # Compute image data weighted COM for each region
            mask = ID == i
            a = A[mask]
            b = gp[mask, :]
            cw[i, :] = np.mean(a[:, np.newaxis] * b, axis=0) / np.mean(a, axis=0)

        p = cw  # Move the points to the weighted average location
        dt = Delaunay(p)
        kdtree = cKDTree(p)
        ID = kdtree.query(gp)[1]

        # Update plot
        #vr.set_array(ID.reshape(n, m), alpha=0.2)
        vr = plt.imshow(ID.reshape(n, m), alpha=0.2)
        plt.plot(p[:, 0], p[:, 1], '.', markersize=2, color=[0, 0, 0])
        plt.draw()
        plt.pause(0.01)

    plt.show()
    return p

    #filename = 'output.txt'
    #np.savetxt(filename, imdata, fmt='%s', delimiter='\t')

def optimal_transport(coordinates, gaussian, num_points):
    cost_matrix = ot.dist(coordinates, gaussian)

    source_distribution = np.ones(num_points) / num_points
    target_distribution = np.ones(num_points) / num_points

    optimal_transport_plan = ot.emd(source_distribution, target_distribution, cost_matrix)
    return optimal_transport_plan

def main(image_path):
    number_of_dots = 9000

    imdata = Image.open(image_path)
    imdata = np.array(imdata)

    coordinates = voronoi_diagram(imdata, number_of_dots)
    gaussian = gaussian_filter(imdata.shape[0], 1.6)
    gaussian = voronoi_diagram(gaussian, number_of_dots)
    weights = optimal_transport(coordinates, gaussian, number_of_dots)


#image_path = r"C:\Users\Ayomide Enoch Ojo\Downloads\cat.png"
image_path = r"C:\Users\Ayomide Enoch Ojo\PycharmProjects\Lloyd'sVornoi\mlk_square.jpg"
main(image_path)

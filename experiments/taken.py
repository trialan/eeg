import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
from eeg.laplacian import create_triangular_dmesh, get_electrode_coordinates, plot_mesh

def generate_synthetic_timeseries(timeseries, base_delay, num_synthetics):
    """Generate synthetic timeseries with integer multiple delays of base_delay."""
    n, t = timeseries.shape
    synthetic_data = []
    for i in range(n):
        for k in range(1, num_synthetics + 1):
            delay = base_delay * k
            synthetic_data.append(np.roll(timeseries[i], delay))
    return np.array(synthetic_data)


def position_synthetic_electrodes(coords, num_synthetics):
    """Position synthetic electrodes on the convex hull."""
    hull = ConvexHull(coords)
    hull_points = coords[hull.vertices]
    synthetic_coords = []
    
    for i in range(len(hull_points) - 1):
        for k in range(1, num_synthetics + 1):
            new_position = hull_points[i] + (hull_points[i + 1] - hull_points[i]) * (k / (num_synthetics + 1))
            synthetic_coords.append(new_position)

    # Ensure to close the loop by connecting the last hull point to the first
    for k in range(1, num_synthetics + 1):
        new_position = hull_points[-1] + (hull_points[0] - hull_points[-1]) * (k / (num_synthetics + 1))
        synthetic_coords.append(new_position)

    return np.array(synthetic_coords)


def plot_electrodes(original_coords, synthetic_coords):
    """Plot original and synthetic electrodes."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(original_coords[:, 0], original_coords[:, 1], original_coords[:, 2], c='b', label='Original')
    ax.scatter(synthetic_coords[:, 0], synthetic_coords[:, 1], synthetic_coords[:, 2], c='r', label='Synthetic')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    original_coords = get_electrode_coordinates()
    timeseries = np.random.rand(len(original_coords), 1000)  # Simulated timeseries data
    base_delay = 1  # Base delay in samples
    num_synthetics = 5  # Number of synthetic timeseries per original electrode
    synthetic_timeseries = generate_synthetic_timeseries(timeseries, base_delay, num_synthetics)  # Generate synthetic timeseries
    synthetic_coords = position_synthetic_electrodes(original_coords, num_synthetics)  # Position synthetic electrodes
    all_coords = np.vstack([original_coords, synthetic_coords])
    print("Shape of all_coords:", all_coords.shape)
    plot_electrodes(original_coords, synthetic_coords)
    hull = ConvexHull(all_coords)
    print("Hull points: ", hull.points)
    mesh = create_triangular_dmesh(all_coords)
    plot_mesh(mesh)

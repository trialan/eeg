import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from scipy.spatial import ConvexHull


def generate_sphere_points(radius, num_points):
    phi = np.random.uniform(0, np.pi, num_points)
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    points = np.vstack((x, y, z)).T
    return points


def generate_sphere_mesh(radius, num_points=100):
    points = generate_sphere_points(radius, num_points)
    hull = ConvexHull(points)
    faces = hull.simplices
    return points, faces


def plot_sphere(vertices, faces):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each face
    for face in faces:
        poly = vertices[face]
        tri = art3d.Poly3DCollection([poly], edgecolor='k', alpha=0.3)
        ax.add_collection3d(tri)

    # Set limits and labels
    max_range = np.array([vertices[:,0].max()-vertices[:,0].min(), 
                          vertices[:,1].max()-vertices[:,1].min(), 
                          vertices[:,2].max()-vertices[:,2].min()]).max() / 2.0

    mid_x = (vertices[:,0].max()+vertices[:,0].min()) * 0.5
    mid_y = (vertices[:,1].max()+vertices[:,1].min()) * 0.5
    mid_z = (vertices[:,2].max()+vertices[:,2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


if __name__ == '__main__':
    radius = 1.0
    vertices, faces = generate_sphere_mesh(radius)
    plot_sphere(vertices, faces)



import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import mne
from mayavi import mlab
from scipy.spatial import Delaunay, ConvexHull
from eeg.laplacian import plot_mesh, plot_basis_functions, create_triangular_dmesh
from eeg import physionet_runs
from eeg.data import get_raw_data
import spharapy.trimesh as trimesh
from scipy.interpolate import Rbf
from mpl_toolkits.mplot3d import Axes3D

def compute_lead_field_matrix():
    fwd = compute_forward_solution()
    leadfield = fwd['sol']['data']
    print("\n#### Lead Field Matrix Computed ####\n")
    return leadfield

def compute_forward_solution():
    raw = get_raw_data(1, physionet_runs)
    subject = "sample"
    subjects_dir = mne.datasets.sample.data_path() / 'subjects'
    conductivity = (0.3, 0.006, 0.3)
    model = mne.make_bem_model(subject=subject, #is this OK??
                               ico=None,
                               conductivity=conductivity,
                               subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)
    src = mne.setup_source_space(subject,
                                 spacing='oct4',
                                 add_dist=False,
                                 subjects_dir=subjects_dir)

    trans = 'fsaverage'  # ?
    fwd = mne.make_forward_solution(raw.info,
                                    trans=trans,
                                    src=src,
                                    bem=bem,
                                    meg=False,
                                    eeg=True,
                                    mindist=5.0,
                                    n_jobs=1)

    print("\n#### Forward Solution Computed ####\n")
    return model, fwd

def generate_and_convert_bem_surfaces(subject, subjects_dir):
    # Set conductivity parameters for scalp, skull, and brain
    conductivity = (0.3, 0.006, 0.3)
    # Create BEM model surfaces
    bem_model = mne.make_bem_model(subject=subject,
                                      ico=None,
                                      conductivity=conductivity,
                                      subjects_dir=subjects_dir)
    # Convert BEM surfaces to SpharaPy mesh format
    sphara_meshes = []
    for surface in bem_model:
        vertlist = surface['rr']  # Vertex coordinates
        trilist = surface['tris']   # Triangle faces
        
        # Convert to SpharaPy mesh format
        sphara_mesh = trimesh.TriMesh(trilist, vertlist)
        sphara_meshes.append(sphara_mesh)
    return sphara_meshes    


def plot_vertices(vertices):
    fig = plt.figure()
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


def return_source_locations(fwd):
    src = fwd['src']
    vertices = [s['rr'][s['vertno']] for s in src]
    vertices = np.vstack(vertices)
    return vertices


def reconstruct_curved_surface_and_delaunay(coords):
    """
    Reconstruct a curved surface from a set of 3D coordinates and create a Delaunay mesh.

    Parameters:
    coords (array-like): An array of shape (n, 3) representing the x, y, z coordinates of the points.

    Returns:
    Delaunay object: The Delaunay triangulation of the surface.
    """

    # Convert coordinates to a NumPy array if it isn't already
    coords = np.array(coords)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    
    # Create a grid for interpolation
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    
    # Interpolate the surface using RBF
    rbf = Rbf(x, y, z, function='multiquadric', smooth=0.1)
    zi = rbf(xi, yi)
    
    # Flatten the grid for Delaunay triangulation
    points2D = np.vstack([xi.flatten(), yi.flatten()]).T
    points3D = np.vstack([xi.flatten(), yi.flatten(), zi.flatten()]).T
    
    # Perform Delaunay triangulation on the interpolated surface
    tri = Delaunay(points2D)
    
    # Plot the original points and the interpolated surface with the Delaunay mesh
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the interpolated surface
    ax.plot_surface(xi, yi, zi, cmap='viridis', alpha=0.7)

    # Plot the original points
    ax.scatter(x, y, z, color='r', s=20)

    # Plot the Delaunay edges
    for simplex in tri.simplices:
        simplex = np.append(simplex, simplex[0])  # Cycle back to the first vertex
        ax.plot(points3D[simplex, 0], points3D[simplex, 1], points3D[simplex, 2], 'r-', lw=0.5)
    
    # Label axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

    return tri


def plot_bem_and_sources(subject, subjects_dir, src):
        mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir, src=src, orientation='coronal')


if __name__ == '__main__':
    model, fwd = compute_forward_solution()
    src=fwd['src']
    #subjects_dir = mne.datasets.sample.data_path() / 'subjects'
    #sphara_meshes = generate_and_convert_bem_surfaces('sample', subjects_dir)
    vertices = return_source_locations(fwd)
    plot_vertices(vertices)
    tri = reconstruct_curved_surface_and_delaunay(vertices)


    #fwd_fixed = mne.convert_forward_solution(
         #       fwd, surf_ori=True, force_fixed=True, use_cps=True
          #      )
    #leadfield = fwd_fixed["sol"]["data"]
    #print("Leadfield matrix shape: ", leadfield.shape)
    #leadfield = compute_lead_field_matrix()
    #print("SHAPE:",leadfield.shape)
    #subjects_dir = mne.datasets.sample.data_path() / 'subjects'
    #for i,sphara_mesh in enumerate(sphara_meshes):
    #    print(f"\n#### SpharaPy Mesh {i+1} ####")
    #    plot_mesh(sphara_meshes[i])
    #    plot_basis_functions(sphara_meshes[i])


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import spharapy.trimesh as tm
from scipy.spatial import Delaunay, ConvexHull
from scipy.interpolate import griddata
from scipy.spatial.transform import Rotation as R
import spharapy.trimesh as trimesh
import spharapy.spharabasis as sb
import spharapy.datasets as sd
import random
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne import pick_types
from mne.channels import make_standard_montage


"""
    Based on the spharapy package. Read the tutorial below.
    https://spharapy.readthedocs.io/en/latest/auto_examples/plot_02_sphara_basis_eeg.html
"""


def compute_scalp_eigenvectors_and_values():
    xyz_coords = get_electrode_coordinates()
    mesh = create_triangular_dmesh(xyz_coords)
    eigenvectors, eigenvals = compute_mesh_eigenvectors_and_values(mesh)
    return eigenvectors, eigenvals


def compute_mesh_eigenvectors_and_values(mesh):
    sphara_basis_unit = sb.SpharaBasis(mesh, 'inv_euclidean')
    eigenvectors, eigenvals = sphara_basis_unit.basis()
    return eigenvectors, eigenvals


def get_256D_eigenvectors():
    mesh_in = sd.load_eeg_256_channel_study()
    vertlist = np.array(mesh_in['vertlist'])
    trilist = np.array(mesh_in['trilist'])
    mesh_eeg = tm.TriMesh(trilist, vertlist)
    sphara_basis = sb.SpharaBasis(mesh_eeg, 'fem')
    eigenvectors, _ = sphara_basis.basis()
    return eigenvectors


def create_triangular_dmesh(xyz_coords):
    """  Create a mesh using the Delaunay triangulation and/or ConvexHull: keep Delaunay approach commented out for now """
    mesh = Delaunay(xyz_coords)
    hull = ConvexHull(xyz_coords)
    print("Created mesh with hull points:", hull.points.shape)
    print("Created mesh with hull simplices:", hull.simplices.shape)
    mesh = trimesh.TriMesh(hull.simplices, mesh.points)
    mesh = remove_bottom_of_the_mesh(mesh)
    return mesh


class ED(BaseEstimator, TransformerMixin):
    """ This is like sklearn's PCA class, but for Eigen-decomposition (ED). """
    def __init__(self, n_components, eigenvectors):
        self.n_components = n_components
        self.eigenvectors = eigenvectors

    def fit(self, X, y=None):
        if self.eigenvectors is None:
            raise ValueError("Eigenvectors are not set.")
        return self

    def transform(self, X):
        #TODO: investigate this class further, is the dot product correct
        n_channels, n_times = X.shape
        selected_eigenvectors = self.eigenvectors.T[:self.n_components, :]
        X_transformed = np.dot(selected_eigenvectors, X)
        return X_transformed


def get_electrode_coordinates(subject=1):
    """ Get raw EEGMI data from website or locally if already downloaded.
        Filter only EEG channels to get 64 as we expect from Physionet data """
    raw_fnames = eegbci.load_data(subject, [6,10,14]) #all subjects have same coords
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    eegbci.standardize(raw)  # set channel names
    montage = make_standard_montage("standard_1020")
    raw.set_montage(montage)
    raw.set_eeg_reference(projection=True)
    raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")
    # Pick only EEG channels
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
    channel_names = [raw.ch_names[pick] for pick in picks]
    # Get positions of the selected channels
    montage_positions = montage.get_positions()
    xyz_coords = montage_positions['ch_pos']
    points = np.array([xyz_coords[ch_name] for ch_name in channel_names])
    return points


def remove_bottom_of_the_mesh(mesh, N=6):
    """ Visual inspection reveals that the bottom triangles we want to be rid
        of have the longest edgs. So we remove the N triangles with longest edge """
    le =  [length_of_longest_edge(t, mesh.vertlist) for t in mesh.trilist]
    ixs = np.argsort(le)[::-1][:N]
    ntriangles = [mesh.trilist[ix] for ix in range(len(mesh.trilist)) if ix not in ixs]
    mesh.trilist = np.array(ntriangles)
    return mesh


def resample_electrode_positions(xyz_coords, n_vertices, perturbation_scale):
    """ Resample the electrode positions to match the desired number of vertices """
    if len(xyz_coords) == n_vertices:
        return xyz_coords
    elif len(xyz_coords) > n_vertices:
        # Downsample by random selection
        indices = np.random.choice(len(xyz_coords), n_vertices, replace=False)
        return xyz_coords[indices]
    else:
        # Upsample by adding points within the convex hull
        return upsample_within_convex_hull(xyz_coords, n_vertices, perturbation_scale)


def upsample_within_convex_hull(xyz_coords, n_vertices, perturbation_scale):
    vertices = np.copy(xyz_coords)
    while len(vertices)<n_vertices:
        vertices = triple_res_via_perturbed_centroids(vertices, perturbation_scale)
    return vertices


def generate_outward_orthogonal_vector(triangle, hull_centroid, perturbation_scale):
    """Generate a small perturbation vector orthogonal to the plane of the triangle pointing outward."""
    vec1 = triangle[1] - triangle[0]
    vec2 = triangle[2] - triangle[0]
    normal = np.cross(vec1, vec2)
    normal = normal / np.linalg.norm(normal)
    
    # Ensure the normal is pointing outward
    triangle_centroid = np.mean(triangle, axis=0)
    direction_to_hull_centroid = hull_centroid - triangle_centroid
    if np.dot(normal, direction_to_hull_centroid) > 0:
        normal = -normal  # Invert direction if pointing inward
    
    return normal * perturbation_scale  # Small perturbation


def triple_res_via_perturbed_centroids(xyz_coords, perturbation_scale):
    """Upsample the coordinates by adding new points within the convex hull."""
    original_hull = ConvexHull(xyz_coords)
    hull_centroid = np.mean(original_hull.points, axis=0)
    current_hull = original_hull
    new_coords = np.copy(xyz_coords)
    simplex_index = 0
    print("Tripling res, original hull simplices: ", original_hull.simplices.shape)
    
    while simplex_index < len(original_hull.simplices):
            # Get current simplex and its vertices from original hull and coords
            current_simplex = original_hull.simplices[simplex_index]
            vertices = xyz_coords[current_simplex]
            
            #Generate a random point inside this simplex using barycentric coordinates or take centroid (leave one commented for now)
            new_point = np.mean(vertices, axis=0) #centroid
            #weights = np.random.dirichlet(np.ones(len(vertices)))
            #new_point = np.dot(weights, vertices)
                
            # Add a small perturbation orthogonal to the simplex and pointing outward
            perturbation = generate_outward_orthogonal_vector(vertices, hull_centroid, perturbation_scale)
            new_point = new_point + perturbation

            # Ensure the new point is inside the convex hull
            combined_points = np.vstack([new_coords, new_point])
            new_hull = ConvexHull(combined_points)

            if len(new_hull.simplices) > len(current_hull.simplices):
                new_coords = combined_points
                current_hull = new_hull  # Update the hull with the new point

            simplex_index += 1
    
    return new_coords


def length_of_longest_edge(triangle, vertices):
    p1 = vertices[triangle[0]]
    p2 = vertices[triangle[1]]
    p3 = vertices[triangle[2]]
    el1 = distance(p1, p2)
    el2 = distance(p1, p3)
    el3 = distance(p2, p3)
    return max(el1, el2, el3)


def distance(p1, p2):
    dx =(p1[0] - p2[0])
    dy =(p1[1] - p2[1])
    dz =(p1[2] - p2[2])
    distance = np.sqrt(dx**2 + dy**2 + dz**2)
    return distance


def plot_mesh(mesh):
    """ Expects a sphara TriMesh """
    vertices = np.array(mesh.vertlist)
    print("Plotted vertices: ", vertices.shape)
    triangles = np.array(mesh.trilist)
    print("Plotted triangles: ", triangles.shape)
    fig = plt.figure()
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('The triangulated EEG sensor setup')
    ax.view_init(elev=20., azim=80.)
    ax.set_aspect('auto')
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                    triangles=triangles, color='lightblue', edgecolor='black',
                    linewidth=0.5, shade=True, alpha=1)
    plt.show()


def plot_basis_functions(mesh):
    vertices = np.array(mesh.vertlist)
    triangles = np.array(mesh.trilist)
    eigenvectors, eigenvalues = compute_mesh_eigenvectors_and_values(mesh)

    figsb1, axes1 = plt.subplots(nrows=7, ncols=7, figsize=(8, 12),
                                 subplot_kw={'projection': '3d'})
    for i in range(np.size(axes1)):
        colors = np.mean(eigenvectors[triangles, i + 0], axis=1)
        ax = axes1.flat[i]
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(elev=60., azim=80.)
        ax.set_aspect('auto')
        trisurfplot = ax.plot_trisurf(vertices[:, 0], vertices[:, 1],
                                      vertices[:, 2], triangles=triangles,
                                      cmap=plt.cm.bwr,
                                      edgecolor='white', linewidth=0.)
        trisurfplot.set_array(colors)
        trisurfplot.set_clim(-0.15, 0.15)

    cbar = figsb1.colorbar(trisurfplot, ax=axes1.ravel().tolist(), shrink=0.85,
                           orientation='horizontal', fraction=0.05, pad=0.05,
                           anchor=(0.5, -4.5))

    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.08, top=1.0)
    plt.show()


if __name__ == '__main__':
    xyz_coords = get_electrode_coordinates()
    resampled_coords = resample_electrode_positions(xyz_coords, 150, 1e-3)
    mesh = create_triangular_dmesh(resampled_coords)
    eigenvectors, eigenvalues = compute_mesh_eigenvectors_and_values(mesh)
    plot_basis_functions(mesh)
    plot_mesh(mesh)



import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import mne
from scipy.spatial import Delaunay, ConvexHull
from eeg.laplacian import plot_mesh, plot_basis_functions, create_triangular_dmesh, resample_electrode_positions
from eeg.data import get_raw_data
import spharapy.trimesh as trimesh
from scipy.interpolate import Rbf, griddata
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D
from mne.transforms import apply_trans
physionet_runs = [6,10,14]

def compute_lead_field_matrix():
    fwd = compute_forward_solution()
    leadfield = fwd['sol']['data']
    print("\n#### Lead Field Matrix Computed ####\n")
    return leadfield

def compute_lead_field_matrix_():
    fwd = compute_forward_solution()
    fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True) #magic conversion that brings matrix back to Nchannels to Nsources (?)
    leadfield = fwd_fixed['sol']['data']
    print("\n#### Lead Field Matrix Computed ####\n")
    return leadfield

def compute_forward_solution(conductivity=(0.3, 0.006, 0.3)):
    raw = get_raw_data(1, physionet_runs)
    subject = "sample"
    subjects_dir = mne.datasets.sample.data_path() / 'subjects'
    model = mne.make_bem_model(subject=subject, #is this OK??
                               ico=None,
                               conductivity=conductivity,
                               subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)
    src = mne.setup_source_space(subject,
                                 spacing='oct4',
                                 add_dist=False,
                                 subjects_dir=subjects_dir)

    fiducials_trans= "/home/x-trialan/eeg/inverseproblem/phys.fif"
    fwd = mne.make_forward_solution(raw.info,
                                    trans=fiducials_trans,
                                    src=src,
                                    bem=bem,
                                    meg=False,
                                    eeg=True,
                                    mindist=5.0,
                                    n_jobs=-1)
    
    print("\n#### Forward Solution Computed ####\n")
    return fwd

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


def plot_vertices(vertices1, vertices2):
    fig = plt.figure()
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(vertices2[:, 0], vertices2[:, 1], vertices2[:, 2], color='red')
    ax.scatter(vertices1[:, 0], vertices1[:, 1], vertices1[:, 2], color='black')
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

    coords = np.array(coords)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    
    # Create a grid for interpolation
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    
    # Interpolate the surface using griddata
    zi = griddata((x, y), z, (xi, yi), method='cubic')
    
    # Flatten the grid for Delaunay triangulation
    points2D = np.vstack([xi.flatten(), yi.flatten()]).T
    points3D = np.vstack([xi.flatten(), yi.flatten(), zi.flatten()]).T
    
    # Remove NaN values which may appear due to interpolation
    mask = ~np.isnan(points3D[:, 2])
    points2D = points2D[mask]
    points3D = points3D[mask]
    
    # Perform Delaunay triangulation on the interpolated surface
    tri = Delaunay(points2D) 
    
      # Plot the original points and the interpolated surface with the Delaunay mesh
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the interpolated surface
    #ax.plot_surface(xi, yi, zi, cmap='viridis', alpha=0.7)

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


def setup_subject_source_space(subject, subjects_dir, spacing='oct6'):
    src = mne.setup_source_space(subject, spacing=spacing, subjects_dir=subjects_dir, add_dist=False)
    return src

def load_pial_surface(subject, subjects_dir, hemisphere):
    surface_path = f'{subjects_dir}/{subject}/surf/{hemisphere}.pial'
    vertices, triangles = mne.read_surface(surface_path)
    return vertices, triangles

def combine_hemisphere_surfaces(vertices_lh, triangles_lh, vertices_rh, triangles_rh):
    vertices = np.vstack((vertices_lh, vertices_rh))
    triangles = np.vstack((triangles_lh, triangles_rh + len(vertices_lh)))
    return vertices, triangles

def plot_cortical_surface_and_sources(vertices, triangles, src):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print(vertices)
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=triangles, edgecolor='k', linewidth=0.2, color='lightgrey', alpha=0.5)

    for hemi in src:
        rr = hemi['rr'][hemi['inuse'].astype(bool)]
        ax.scatter(rr[:, 0], rr[:, 1], rr[:, 2], s=30, c='b', alpha=0.7)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def plot_cortical_surface_and_sources_via_PyVista(vertices, triangles, src):
    # Create PyVista mesh for plotting
    mesh = pv.PolyData(vertices, np.hstack([np.full((triangles.shape[0], 1), 3), triangles]).astype(int))
    
    # Initialize plotter
    plotter = pv.Plotter()
    
    # Add mesh to plotter
    plotter.add_mesh(mesh, color='lightgrey', opacity=0.5, show_edges=True, edge_color='black', line_width=0.5)
    
    # Add source points to plotter
    for hemi in src:
        rr = hemi['rr'][hemi['inuse'].astype(bool)]
        plotter.add_points(rr, color='red', point_size=10)
    
    # Show plot
    plotter.show()


def plot_bem_and_sources(subject, subjects_dir, src):
        mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir, src=src, orientation='coronal')


def decimate_mesh(vertices, triangles, reduction_factor=0.5):
    # Create a PyVista mesh
    mesh = pv.PolyData(vertices, np.hstack([np.full((triangles.shape[0], 1), 3), triangles]).astype(int))

    # Decimate the mesh using vtk's decimate method
    decimated_mesh = mesh.decimate(target_reduction=reduction_factor)

    # Extract the decimated vertices and triangles
    decimated_vertices = decimated_mesh.points
    decimated_triangles = decimated_mesh.faces.reshape(-1, 4)[:, 1:]

    return decimated_vertices, decimated_triangles


def plot_mesh_and_sources(mesh, src_vertices):
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
    ax.set_title('SpharaPy Mesh')
    ax.view_init(elev=20., azim=80.)
    ax.set_aspect('auto')
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                    triangles=triangles, color='lightblue', edgecolor='black',
                    linewidth=0.5, shade=True, alpha=0.5)
    # Plot source locations
    ax.scatter(src_vertices[:, 0], src_vertices[:, 1], src_vertices[:, 2], color='r')
    print("Source vertices shape: ", src_vertices.shape)
    plt.show()


def read_xfm(xfm_path):
    matrix = []
    with open(xfm_path, 'r') as file:
        lines = file.readlines()
        matrix_lines_started = False
        for line in lines:
            if line.strip().startswith('Linear_Transform'):
                matrix_lines_started = True
                continue
            if matrix_lines_started:
                if line.strip() == '':
                    break
                values = [x for x in line.strip().replace(';', '').split() if x]
                matrix.append([float(x) for x in values])
    if len(matrix) != 3:
        raise ValueError(f"Unexpected number of lines for transformation matrix: {len(matrix)}")
    matrix.append([0, 0, 0, 1])
    print("head - mri matrix: ", np.array(matrix))
    return np.array(matrix)


def transform_to_mri_coordinates(vertices, subject, subjects_dir, trans):
    # Load the transformation matrix from the .xfm file
    #xfm_path = f'{subjects_dir}/{subject}/mri/transforms/talairach.xfm'
    #xfm_matrix = read_xfm(xfm_path)

    # Apply the transformation matrix
    vertices_mri = apply_trans(trans, vertices)
    return vertices_mri

def normalize_vertices(vertices):
    """
    Normalize the vertex coordinates to be centered around the origin and fit within a unit sphere.

    Parameters:
    vertices (numpy.ndarray): Array of vertex coordinates with shape (n_vertices, 3).

    Returns:
    numpy.ndarray: Normalized vertex coordinates.
    """
    # Center the vertices by subtracting the mean
    centered_vertices = vertices - np.mean(vertices, axis=0)

    # Scale the vertices to fit within a unit sphere
    max_distance = np.max(np.linalg.norm(centered_vertices, axis=1))
    normalized_vertices = centered_vertices / max_distance

    return normalized_vertices


if __name__ == '__main__': 
    """
    ###Testing and Plotting###
    
    # Get pial surface
    data_path = mne.datasets.sample.data_path()
    subjects_dir = data_path / 'subjects'
    subject = 'sample'
    vertices_lh, triangles_lh = load_pial_surface(subject, subjects_dir, 'lh')
    vertices_rh, triangles_rh = load_pial_surface(subject, subjects_dir, 'rh')
    vertices, triangles = combine_hemisphere_surfaces(vertices_lh, triangles_lh, vertices_rh, triangles_rh)
    
    # Get electrode positions
    raw = get_raw_data(1, physionet_runs)
    
    # Get source vertices from setup source space (as in compute_forward_solution())
    src = mne.setup_source_space(subject, spacing='oct4', surface='pial', subjects_dir=subjects_dir, add_dist=False)
    src_vertices = [s['rr'][s['vertno']] for s in src]
    src_vertices = np.vstack(src_vertices)
    
    # Get fiducials from head to mri coord transform and plot electrodes, pial surface, and sources
    fiducials_trans= "phys.fif"
    fig = mne.viz.plot_alignment(raw.info, trans = fiducials_trans, subject = subject, subjects_dir=subjects_dir, surfaces="pial", src=src, show_axes=True, coord_frame = 'mri', dig=True, eeg=True)

    ## Plotting sources on SpharaPy mesh ##

    # Transform pial surface vertices to head coordinates using the correct transformation file (without fiducials because no electrodes considered here!)
    mri_to_head_trans = mne.transforms.Transform("mri", "head")
    vertices_head = apply_trans(mri_to_head_trans, vertices)
    
    # Reduce mesh resolution and create SpharaPy mesh
    decimated_vert, decimated_tria = decimate_mesh(normalize_vertices(vertices_head), triangles, 0.95)
    decimated_sphara_mesh = trimesh.TriMesh(decimated_tria, decimated_vert)
    
    # Plot SpharaPy mesh and source space
    plot_mesh_and_sources(decimated_sphara_mesh, normalize_vertices(src_vertices))
    """
    
    
    ### Leadfield matrix computation ###
    leadfield = compute_lead_field_matrix_()
    print("SHAPE:",leadfield.shape)
   

    """
    plot_basis_functions(sp`hara_mesh)
    plot_mesh(decimated_sphara_mesh)
    subjects_dir = mne.datasets.sample.data_path() / 'subjects'
    for i,sphara_mesh in enumerate(sphara_meshes):
        print(f"\n#### SpharaPy Mesh {i+1} ####")
        plot_mesh(sphara_meshes[i])
        plot_basis_functions(sphara_meshes[i])
    """

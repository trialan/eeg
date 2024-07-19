import matplotlib.pyplot as plt
import numpy as np
import mne
from scipy.spatial import Delaunay, ConvexHull
from eeg.laplacian import plot_mesh, plot_basis_functions, create_triangular_dmesh
from eeg import physionet_runs
from eeg.data import get_raw_data
import spharapy.trimesh as trimesh

"""
- conductivity parameters: see Table (1) in "Global sensitivity of EEG
  source analysis to tissue conductivity uncertainties", https://doi.org/10.3389/fnhum.2024.1335212
    --> values (0.3, 0.006, 0.3), in Siemens/meter are roughly correct (good order of magnitude,
    good ratio of values). Could be tweaked a bit. Issue is that they can differ quite a bit
    for each subject (by a factor of 2-3 sometimes).

- ico parameter: set to None for highest resolution. This parameter controls
  downsampling.

- subject = "sample": 

"""

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
    return fwd

def return_source_locations(fwd):
    # Extract the source space coordinates from the forward solution
    src = fwd['src']
    vertices = [s['rr'][s['vertno']] for s in src]
    vertices = np.vstack(vertices)
    return vertices

def plot_source_surface(fwd):
    # Extract the source space coordinates from the forward solution
    src = fwd['src']
    vertices = [s['rr'][s['vertno']] for s in src]
    vertices = np.vstack(vertices)

    # Create a Delaunay triangulation of the vertices
    tri = Delaunay(vertices)

    # Create the plot using mayavi
    mlab.figure(size=(800, 800), bgcolor=(1, 1, 1))
    mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], tri.simplices,
                         colormap='Spectral', opacity=0.6)
    mlab.points3d(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                  color=(1, 0, 0), scale_factor=0.005)
    mlab.xlabel('X')
    mlab.ylabel('Y')
    mlab.zlabel('Z')
    mlab.show()

def generate_and_convert_bem_surfaces(subject, subjects_dir):
    """
    Generates BEM surfaces using MNE and converts them to SpharaPy format.
    
    Parameters:
    subject (str): Subject identifier.
    subjects_dir (str): Directory where the subject data is stored.
    
    Returns:
    sphara_meshes (list of SpharaPy Mesh): BEM surfaces in SpharaPy format.
    """
    # Set conductivity parameters for scalp, skull, and brain
    conductivity = (0.3, 0.006, 0.3)
    # Create BEM model surfaces
    bem_model = mne.make_bem_model(subject=subject,
                                      ico=4,
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

if __name__ == '__main__':
    #fwd = compute_forward_solution(3)
    #plot_source_surface(fwd)
    #source_vertices = return_source_locations(fwd)
    #mesh = create_triangular_dmesh(source_vertices)
    #plot_mesh(mesh)

    #fwd_fixed = mne.convert_forward_solution(
         #       fwd, surf_ori=True, force_fixed=True, use_cps=True
          #      )
    #leadfield = fwd_fixed["sol"]["data"]
    #print("Leadfield matrix shape: ", leadfield.shape)
    leadfield = compute_lead_field_matrix()
    print("SHAPE:",leadfield.shape)
    #subjects_dir = mne.datasets.sample.data_path() / 'subjects'
    #sphara_meshes = generate_and_convert_bem_surfaces('sample', subjects_dir)
    #for i,sphara_mesh in enumerate(sphara_meshes):
    #    print(f"\n#### SpharaPy Mesh {i+1} ####")
    #    plot_mesh(sphara_meshes[i])
    #    plot_basis_functions(sphara_meshes[i])


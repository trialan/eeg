import matplotlib.pyplot as plt
import numpy as np
import mne
from mne.datasets import sample
from eeg.laplacian import plot_mesh, plot_basis_functions
from eeg import physionet_runs
from eeg.data import get_raw_data
import spharapy.trimesh as trimesh

def generate_and_convert_bem_surfaces(subject='sample', subjects_dir=None):
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
    bem_surfaces = mne.make_bem_model(subject=subject,
                                      ico=None,
                                      conductivity=conductivity,
                                      subjects_dir=subjects_dir)
    
    # Convert BEM surfaces to SpharaPy mesh format
    sphara_meshes = []
    
    for surface in bem_surfaces:
        vertlist = surface['rr']  # Vertex coordinates
        trilist = surface['tris']   # Triangle faces
        
        # Convert to SpharaPy mesh format
        sphara_mesh = trimesh.TriMesh(trilist, vertlist)
        sphara_meshes.append(sphara_mesh)
    
    return sphara_meshes


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
    """ Same LF matrix for all subjects since we use avg brain + standard
        electrode positions, so use subject 1 """
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
                                 add_dist='patch',
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

    leadfield = fwd['sol']['data']
    print("\n#### Lead Field Matrix Computed ####\n")
    return leadfield


if __name__ == '__main__':
    #leadfield = compute_lead_field_matrix()
    subjects_dir = mne.datasets.sample.data_path() / 'subjects'
    sphara_meshes = generate_and_convert_bem_surfaces(subject='sample', subjects_dir=subjects_dir)
    for i,sphara_mesh in enumerate(sphara_meshes):
        print(f"\n#### SpharaPy Mesh {i+1} ####")
        plot_mesh(sphara_meshes[i])
        plot_basis_functions(sphara_meshes[i])


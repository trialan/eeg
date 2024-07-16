import matplotlib.pyplot as plt
import numpy as np
import mne
from mne.datasets import sample

from eeg.data import get_raw_data


"""
In this file I simply follow the tutorial at this page:
    https://mne.tools/stable/auto_tutorials/forward/30_forward.html

The object of the tutorial is to compute the lead field matrix with BEM.
The only difficult thing to do was to actually set up freesurfer. I
recommend following the video tutorial here:
    https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall?action=AttachFile&do=get&target=installFS_demo.mp4

    PS: for me her solution to allowing the pkg installer to run didn't work, instead i had to run:

    sudo xattr -rd com.apple.quarantine /path/to/freesurfer


Correct runs are [6,10,14] but bad practice to hard code them, need to clean this up

"""

def compute_lead_field_matrix():
    """ Same LF matrix for all subjects since we use avg brain + standard
        electrode positions, so use subject 1 """
    raw = get_raw_data(1, [6, 10, 14])
    subject = "sample"
    subjects_dir = mne.datasets.sample.data_path() / 'subjects'
    conductivity = (0.3, 0.006, 0.3)  # for three layers
    model = mne.make_bem_model(subject=subject, #is this OK??
                               ico=4,
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
    return leadfield


if __name__ == '__main__':
    leadfield = compute_lead_field_matrix()



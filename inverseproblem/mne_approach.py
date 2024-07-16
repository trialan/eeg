import matplotlib.pyplot as plt
import numpy as np

import mne
from mne.datasets import sample


"""
In this file I simply follow the tutorial at this page:
    https://mne.tools/stable/auto_tutorials/forward/30_forward.html

The object of the tutorial is to compute the lead field matrix with BEM.
The only difficult thing to do was to actually set up freesurfer. I
recommend following the video tutorial here:
    https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall?action=AttachFile&do=get&target=installFS_demo.mp4

    PS: for me her solution to allowing the pkg installer to run didn't work, instead i had to run:

    sudo xattr -rd com.apple.quarantine /path/to/freesurfer
"""

data_path = sample.data_path()

# the raw file containing the channel location + types
sample_dir = data_path / "MEG" / "sample"
raw_fname = sample_dir / "sample_audvis_raw.fif"
# The paths to Freesurfer reconstructions
subjects_dir = data_path / "subjects"
subject = "sample"

plot_bem_kwargs = dict(
    subject=subject,
    subjects_dir=subjects_dir,
    brain_surfaces="white",
    orientation="coronal",
    slices=[50, 100, 150, 200],
)

mne.viz.plot_bem(**plot_bem_kwargs)
plt.show()

# The transformation file obtained by coregistration
trans = sample_dir / "sample_audvis_raw-trans.fif"

info = mne.io.read_info(raw_fname)
# Here we look at the dense head, which isn't used for BEM computations but
# is useful for coregistration.
mne.viz.plot_alignment(
    info,
    trans,
    subject=subject,
    dig=True,
    meg=["helmet", "sensors"],
    subjects_dir=subjects_dir,
    surfaces="head-dense",
)


src = mne.setup_source_space(
    subject, spacing="oct4", add_dist="patch", subjects_dir=subjects_dir
)
print(src)

#this plots the surface-based source space (so sources on the surface?)
mne.viz.plot_bem(src=src, **plot_bem_kwargs)
plt.show()



#this is plotting possible source dipole positions in a sphere
sphere = (0.0, 0.0, 0.04, 0.09)
vol_src = mne.setup_volume_source_space(
    subject,
    subjects_dir=subjects_dir,
    sphere=sphere,
    sphere_units="m",
    add_interpolator=False,
)  # just for speed!
print(vol_src)

mne.viz.plot_bem(src=vol_src, **plot_bem_kwargs)
plt.show()


#this makes sure the source dipoles are inside the brain
surface = subjects_dir / subject / "bem" / "inner_skull.surf"
vol_src = mne.setup_volume_source_space(
    subject, subjects_dir=subjects_dir, surface=surface, add_interpolator=False
)  # Just for speed!
print(vol_src)

mne.viz.plot_bem(src=vol_src, **plot_bem_kwargs)
plt.show()



#this is just a 3D view of the same thing afaik
fig = mne.viz.plot_alignment(
    subject=subject,
    subjects_dir=subjects_dir,
    surfaces="white",
    coord_frame="mri",
    src=src,
)
mne.viz.set_3d_view(
    fig,
    azimuth=173.78,
    elevation=101.75,
    distance=0.30,
    focalpoint=(-0.03, -0.01, 0.03),
)


conductivity = (0.3, 0.006, 0.3)  # for three layers
model = mne.make_bem_model(
    subject="sample", ico=4, conductivity=conductivity, subjects_dir=subjects_dir
)
bem = mne.make_bem_solution(model)


fwd = mne.make_forward_solution(
    raw_fname,
    trans=trans,
    src=src,
    bem=bem,
    meg=False,
    eeg=True,
    mindist=5.0,
    n_jobs=None,
    verbose=True,
)
print(fwd)


print(f"Before: {src}")
print(f'After:  {fwd["src"]}')


leadfield = fwd["sol"]["data"]
print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)






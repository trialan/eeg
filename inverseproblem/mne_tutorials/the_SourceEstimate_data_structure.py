from mne import read_source_estimate
from mne.datasets import sample

print(__doc__)

# Paths to example data
sample_dir_raw = sample.data_path()
sample_dir = sample_dir_raw / "MEG" / "sample"
subjects_dir = sample_dir_raw / "subjects"

fname_stc = sample_dir / "sample_audvis-meg"

stc = read_source_estimate(fname_stc, subject="sample")

# Define plotting parameters
surfer_kwargs = dict(
    hemi="lh",
    subjects_dir=subjects_dir,
    clim=dict(kind="value", lims=[8, 12, 15]),
    views="lateral",
    initial_time=0.09,
    time_unit="s",
    size=(800, 800),
    smoothing_steps=5,
)

# Plot surface
brain = stc.plot(**surfer_kwargs)

# Add title
brain.add_text(0.1, 0.9, "SourceEstimate", "title", font_size=16)


"""
If every vertex represents the spatial location of a time series, the time series and spatial location can be written into a matrix, where to each vertex (rows) at multiple time points (columns) a value can be assigned. This value is the strength of our signal at a given point in space and time. Exactly this matrix is stored in stc.data.
"""

shape = stc.data.shape

print(f"The data has {shape} vertex locations with {shape} sample points each.")


"""
To get access to the vertices: stc.vertices. This is two arrays, one per hemisphere. "A value of 42 would hence map to the x,y,z coordinates of the mesh with index 42. See next section on how to get access to the positions in a mne.SourceSpaces object."
"""


"""
As mentioned above, src carries the mapping from stc to the surface. The surface is built up from a triangulated mesh for each hemisphere.
"""











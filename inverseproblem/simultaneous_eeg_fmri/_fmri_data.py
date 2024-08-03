import os
import glob
import numpy as np
import nibabel as nib
import scipy.io
from scipy import signal
from scipy.ndimage import gaussian_filter
from nilearn.image import clean_img
from tqdm import tqdm
from bvbabel.vtc import read_vtc

from eeg.inverseproblem.simultaneous_eeg_fmri.data_utils import (get_DS116_paths,
                                                                 load_events,
                                                                 get_bv_paths)

"""
This fMRI data is:
    - motion corrected (source: "mcf" in file name)
    - has brain artifacts removed (source: openfrmi comments section)
    - spatially smoothed (we do this)
    - high pass filtered (we do this)
    - slice time corrected (we do this)


Questions:
    - Should I do some "unentangling" of the fMRI data due to mismatch
      in timescales? (HRF is slow, EEG is fast, stimuli are every 2-3s)
    - Should I use a canonical HRF instead of my peak-a-5.5s-and-nothing-else
      model? Probably, just not sure how. Discuss with Nyx. Read the litt.


Experiment notes:
    CNN on non-resampled dataset: 80% (i.e. same as uniform 0 pred, 80% of labels are 0.)
    CNN on resampled dataset    : 85% (i.e crushed it because here random is 50%)
"""


tr = 2.0  # The repetition time, TR is standard naming


def get_bv_fmri_data(root_dir):
    bold_paths, _, event_paths = get_bv_paths(root_dir)
    slice_order = np.loadtxt(os.path.join(root_dir, "DS116/ds116_metadata/supplementary/slice_order.txt"))
    Xs = []
    ys = []
    for bp, ep in tqdm(list(zip(bold_paths, event_paths))):
        x = load_bv_file(bp)
        events = load_events(ep)
        fmri, labels = get_run_data(x, events)
        fmri = preprocess_fmri(fmri, tr, slice_order)
        assert fmri.shape == (125, 56, 69, 56)
        Xs.extend(fmri)
        ys.extend(labels)
    return np.array(Xs), np.array(ys)


def load_bv_file(path):
    info, data = read_vtc(path)
    return data


"""
    Code below here is for reading the processed data downloaded from
    openfmri. Code above is for reading the BrainVoyager files generated
    by Nyx after preprocessing them manually.
"""


def get_raw_fmri_data(root_dir):
    bold_paths, _, event_paths = get_DS116_paths(root_dir)
    slice_order = np.loadtxt(root_dir + "ds116_metadata/supplementary/slice_order.txt")
    Xs = []
    ys = []
    for bp, ep in tqdm(list(zip(bold_paths, event_paths))):
        x = load_bold_run_data(bp)
        events = load_events(ep)
        fmri, labels = get_run_data(x, events)
        xp = preprocess_fmri(fmri, tr, slice_order)
        Xs.extend(xp)
        ys.extend(labels)
    return np.array(Xs), np.array(ys)


def get_run_data(fmri_data, events):
    fmri = [get_event_fmri(fmri_data, e[0]) for e in events]
    labels = np.array([e[2] for e in events])
    return np.array(fmri), labels


def load_bold_run_data(run_path):
    img = nib.load(run_path)
    data = img.get_fdata()
    return data


def get_event_fmri(fmri_data, event_time):
    event_ix = convert_time_to_fmri_ix(event_time)
    fmri_eeg = fmri_data[:, :, :, event_ix]
    return fmri_eeg


def convert_time_to_fmri_ix(t):
    """Assume it takes 5.5s for haemodynamic response to peak"""
    fmri_ixs = np.arange(170)
    fmri_times = fmri_ixs * 2.0
    differences = np.abs(fmri_times - (t + 5.5))
    ix = np.argmin(differences)
    return ix


def preprocess_fmri(fmri_data, tr, slice_order):
    fmri_data_highpass = clean_img(
        nib.Nifti1Image(fmri_data, affine=np.eye(4)), high_pass=0.01, t_r=tr
    ).get_fdata()
    fmri_data_smoothed = gaussian_filter(fmri_data_highpass, sigma=(1.5, 1.5, 1.5, 0))
    fmri_data_reshaped = np.transpose(fmri_data_smoothed, (1, 2, 3, 0))
    fmri_data_stc = slice_timing_correction(fmri_data_reshaped, slice_order, tr)
    return np.transpose(fmri_data_stc, (3, 0, 1, 2))


def slice_timing_correction(fmri_data, slice_order, tr):
    n_slices, n_volumes = fmri_data.shape[2], fmri_data.shape[-1]
    slice_time = (np.array(slice_order) - 1) * tr / n_slices
    time_points = np.arange(n_volumes) * tr

    corrected_data = np.zeros_like(fmri_data)

    for s in range(n_slices):
        slice_data = fmri_data[:, :, s, :]
        time_courses = slice_data.reshape(-1, n_volumes)

        corrected_time_courses = []
        for tc in time_courses:
            # Interpolate to correct for slice timing
            corrected_time_points = time_points - slice_time[s]
            corrected_tc = np.interp(time_points, corrected_time_points, tc)
            corrected_time_courses.append(corrected_tc)

        corrected_slice = np.array(corrected_time_courses).T.reshape(slice_data.shape)
        corrected_data[:, :, s, :] = corrected_slice

    return corrected_data




import os
import glob
import numpy as np
import nibabel as nib
import scipy.io
from scipy import signal
from scipy.ndimage import gaussian_filter
from nilearn.image import clean_img
from tqdm import tqdm

import multiprocessing as mp


"""
This fMRI data is:
    - motion corrected (source: "mcf" in file name)
    - has brain artifacts removed (source: openfrmi comments section)
    - spatially smoothed (we do this)
    - high pass filtered (we do this)
    - slice time corrected (we do this)

The EEG data is:
    - re-referenced (source: README)
    - gradient artifacts removed (source: README)
    - BCG effects removed (source: email to author in openfmri comments)

Question:
    - Should I do some "unentangling" of the fMRI data due to mismatch
      in timescales? (HRF is slow, EEG is fast, stimuli are every 2-3s)
"""


root_dir = "/root/DS116/"
slice_order = np.loadtxt(root_dir + "ds116_metadata/supplementary/slice_order.txt")
fmri_tr = 2.0  # The repetition time, TR is standard naming


def write_data_to_disk(output_dir="processed_data", batch_size=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    X_filename = os.path.join(output_dir, "X_data.npy")
    Y_filename = os.path.join(output_dir, "Y_data.npy")

    N_files = len(get_paths()[0])
    total_sanples = 125 * N_files
    X_shape = (125 * N_files, 34, 1000)
    Y_shape = (125 * N_files, 1)

    X_memmap = np.memmap(
        X_filename, dtype="float32", mode="w+", shape=(total_samples, *X_shape)
    )
    Y_memmap = np.memmap(
        Y_filename, dtype="float32", mode="w+", shape=(total_samples, *Y_shape)
    )

    start_idx = 0
    while start_idx < total_samples:
        end_idx = min(start_idx + batch_size, total_samples)
        for batch_X, batch_Y in get_data(start_idx, end_idx):
            batch_end_idx = start_idx + len(batch_X)
            X_memmap[start_idx:batch_end_idx] = batch_X
            Y_memmap[start_idx:batch_end_idx] = batch_Y.reshape(-1, *Y_shape)
            start_idx = batch_end_idx

    # Flush the memmaps to ensure all data is written to disk
    del X_memmap
    del Y_memmap

    print(f"Data written to {X_filename} and {Y_filename}")
    return X_filename, Y_filename


def get_data(start_idx, end_idx, batch_size=5):
    bold_paths, eeg_paths, event_time_paths = get_paths()
    bold_paths = bold_paths[start_idx:end_idx]
    eeg_paths = eeg_paths[start_idx:end_idx]
    event_time_paths = event_time_paths[start_idx:end_idx]
    for i in range(0, len(bold_paths), batch_size):
        X = []
        Y = []
        batch_paths = zip(eeg_paths[i:i+batch_size], bold_paths[i:i+batch_size], event_time_paths[i:i+batch_size])
        for eeg, bold, eventtimes in tqdm(batch_paths):
            x, y = load_run_data(eeg, bold, eventtimes)
            X.extend(x)
            Y.extend(y)
        yield np.array(X), np.array(Y)


def get_paths():
    def _get_paths(path_type, root):
        assert path_type in ["BOLD", "EEG", "behav"]
        paths = []
        file_name = {
            "BOLD": "bold_mcf_brain.nii.gz",
            "EEG": "EEG_rereferenced.mat",
            "behav": "behavdata.txt",
        }[path_type]

        subject_pattern = os.path.join(root, "sub*")

        for subject_dir in glob.glob(subject_pattern):
            pattern = os.path.join(subject_dir, path_type, "task002_run*", file_name)
            paths.extend(glob.glob(pattern))

        return np.array(paths)
    bold_paths = _get_paths("BOLD", root="/root/DS116")
    eeg_paths = _get_paths("EEG", root="/root/DS116")
    event_paths = _get_paths("behav", root="/root/DS116")
    assert len(event_paths) == len(eeg_paths) == len(bold_paths)
    return bold_paths, eeg_paths, event_paths


def load_run_data(eeg_path, bold_path, event_time_path):
    # eeg_data = load_eeg_run_data(eeg_path)
    fmri_data = load_bold_run_data(bold_path)
    events = load_events(event_time_path)
    # x_kHz, y_mcf_brain = pair_eeg_fmri(eeg_data, fmri_data, events)
    y_mcf_brain = pair_eeg_fmri(fmri_data, events)
    # x = resample_eeg(x_kHz)
    x = preprocess_fmri(y_mcf_brain, fmri_tr, slice_order)
    y = np.array([e.label for e in events])
    return x, y


def load_bold_run_data(run_path):
    img = nib.load(run_path)
    data = img.get_fdata()
    print("fMRI data shape:", data.shape)
    return data


def load_events(file_path):
    """ load in mne Raw format: [time, 0., label] """
    event_data = []
    with open(file_path, "r") as file:
        next(file)
        for line in file:
            components = line.strip().split()
            if len(components) == 4:
                time = float(components[0])
                event_type = int(components[1])
                event_data.append(np.array([time, event_type]))
    event_data = np.array(event_data)
    mne_fmt = np.concatenate((event_data[:, :1],
                              np.zeros((len(event_data),1)),
                              event_data[:, 1:]), axis=1)
    return mne_fmt



# def pair_eeg_fmri(eeg_data, fmri_data, events, tr=2, eeg_fs=1000):
def pair_eeg_fmri(fmri_data, events, tr=2, eeg_fs=1000):
    # n_channels, n_timepoints = eeg_data.shape
    n_volumes = fmri_data.shape[-1]
    # print("EEG duration (seconds):", n_timepoints / eeg_fs)
    print("fMRI duration (seconds):", n_volumes * tr)

    x_list = []
    y_list = []

    for event in events:
        # event_eeg = get_event_eeg(eeg_data, event.time)
        event_fmri = get_event_fmri(fmri_data, event.time)
        # x_list.append(event_eeg)
        y_list.append(event_fmri)

    # return np.array(x_list), np.array(y_list)
    return np.array(y_list)



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



if __name__ == "__main__":
    x, y = write_data_to_disk()

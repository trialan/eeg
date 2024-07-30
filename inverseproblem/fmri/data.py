import os
import glob
import numpy as np
import nibabel as nib
import scipy.io
from scipy import signal
from scipy.ndimage import gaussian_filter
from nilearn.image import clean_img
from tqdm import tqdm


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


root_dir = "/Users/thomasrialan/Documents/code/DS116/"
slice_order = np.loadtxt(root_dir + "ds116_metadata/supplementary/slice_order.txt")
fmri_tr = 2.0 #The repetition time, TR is standard naming


def write_data_to_disk(output_dir='processed_data', batch_size=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    X_filename = os.path.join(output_dir, 'X_data.npy')
    Y_filename = os.path.join(output_dir, 'Y_data.npy')

    total_samples, X_shape, Y_shape = get_total_samples_and_shape()

    X_memmap = np.memmap(X_filename, dtype='float32', mode='w+', shape=(total_samples, *X_shape))
    Y_memmap = np.memmap(Y_filename, dtype='float32', mode='w+', shape=(total_samples, *Y_shape))

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


def get_total_samples_and_shape():
    total_samples = 0
    X_shape = None
    Y_shape = None
    for batch_X, batch_Y in get_data(0, 5):  # Just get the first batch
        total_samples += len(batch_X)
        if X_shape is None:
            X_shape = batch_X.shape[1:]
            Y_shape = batch_Y.shape[1:] if len(batch_Y.shape) > 1 else (1,)
    return total_samples, X_shape, Y_shape


def get_data(start_idx, end_idx, batch_size=5):
    bold_paths, eeg_paths, event_time_paths = get_paths()[start_idx:end_idx]
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
    bold_paths = []
    eeg_paths = []
    event_time_paths = []

    subject_pattern = os.path.join(root_dir, "sub*")

    for subject_dir in glob.glob(subject_pattern):
        bold_pattern = os.path.join(subject_dir, "BOLD", "task002_run*", "bold_mcf_brain.nii.gz")
        bold_paths.extend(glob.glob(bold_pattern))

        eeg_pattern = os.path.join(subject_dir, "EEG", "task002_run*", "EEG_rereferenced.mat")
        eeg_paths.extend(glob.glob(eeg_pattern))

        event_pattern = os.path.join(subject_dir, "behav", "task002_run*", "behavdata.txt")
        event_time_paths.extend(glob.glob(event_pattern))

    assert len(bold_paths) == len(eeg_paths) == len(event_time_paths)
    return bold_paths, eeg_paths, event_time_paths


def load_run_data(eeg_path, bold_path, event_time_path):
    #eeg_data = load_eeg_run_data(eeg_path)
    fmri_data = load_bold_run_data(bold_path)
    events = load_events(event_time_path)
    #x_kHz, y_mcf_brain = pair_eeg_fmri(eeg_data, fmri_data, events)
    y_mcf_brain = pair_eeg_fmri(fmri_data, events)
    #x = resample_eeg(x_kHz)
    x = preprocess_fmri(y_mcf_brain, fmri_tr, slice_order)
    y = np.array([e.label for e in events])
    return x, y


def load_bold_run_data(run_path):
    img = nib.load(run_path)
    data = img.get_fdata()
    print("fMRI data shape:", data.shape)
    return data


def load_eeg_run_data(run_path):
    data = scipy.io.loadmat(run_path)
    eeg_data = data["data_reref"][:34, :]  # Use only the first 34 channels (EEG data)
    eeg_data = eeg_data.astype(np.float64)
    print("EEG data shape:", eeg_data.shape)
    return eeg_data


def load_events(file_path):
    event_data = []
    with open(file_path, "r") as file:
        next(file)
        for line in file:
            components = line.strip().split()
            if len(components) == 4:
                time = float(components[0])
                event_type = int(components[1])
                event_data.append(Event(time, event_type))
    return np.array(event_data)


class Event:
    def __init__(self, time, label):
        self.time = time
        self.label = label


#def pair_eeg_fmri(eeg_data, fmri_data, events, tr=2, eeg_fs=1000):
def pair_eeg_fmri(fmri_data, events, tr=2, eeg_fs=1000):
    #n_channels, n_timepoints = eeg_data.shape
    n_volumes = fmri_data.shape[-1]
    #print("EEG duration (seconds):", n_timepoints / eeg_fs)
    print("fMRI duration (seconds):", n_volumes * tr)

    x_list = []
    y_list = []

    for event in events:
        #event_eeg = get_event_eeg(eeg_data, event.time)
        event_fmri = get_event_fmri(fmri_data, event.time)
        #x_list.append(event_eeg)
        y_list.append(event_fmri)

    #return np.array(x_list), np.array(y_list)
    return np.array(y_list)


def get_event_eeg(eeg_data, event_time):
    event_ix = convert_time_to_eeg_ix(event_time)
    event_eeg = eeg_data.T[event_ix - 500 : event_ix + 500].T
    return event_eeg


def get_event_fmri(fmri_data, event_time):
    event_ix = convert_time_to_fmri_ix(event_time)
    fmri_eeg = fmri_data[:, :, :, event_ix]
    return fmri_eeg


def convert_time_to_eeg_ix(t):
    """EEG sampled at 1kHz"""
    return int(t * 1000)


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
    return np.transpose(fmri_data_stc, (3,0,1,2))


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


def resample_eeg(eeg_data, original_freq=1000, target_freq=500):
    num_trials, num_channels, num_time_points = eeg_data.shape
    new_num_time_points = int(num_time_points * (target_freq / original_freq))
    resampled_data = np.zeros((num_trials, num_channels, new_num_time_points))
    for trial in range(num_trials):
        for channel in range(num_channels):
            resampled_data[trial, channel] = signal.resample(eeg_data[trial, channel], new_num_time_points)
    return resampled_data


if __name__ == "__main__":
    x,y = write_data_to_disk()
    #x, y = collect_all_data()
    1/0
    from eeg.plot_reproduction import assemble_classifier_CSPLDA
    from eeg.utils import get_cv, results
    clf = assemble_classifier_CSPLDA(10)
    cv = get_cv()
    score = results(clf, x, y, cv)
    print(score)



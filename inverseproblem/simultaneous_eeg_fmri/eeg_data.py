import mne
import os
import glob
import scipy
import numpy as np
from tqdm import tqdm
from sklearn.utils import resample

from eeg.inverseproblem.simultaneous_eeg_fmri.data_utils import (get_paths,
                                                                 load_events)


SAMPLING_FREQ = 1000  # Hz
TMIN = -0.2
TMAX = 0.8


def get_data():
    _, eeg_paths, event_paths = get_paths(
        root="/Users/thomasrialan/Documents/code/DS116"
    )
    Xs = []
    ys = []
    for eegp, eventp in list(zip(eeg_paths, event_paths)):
        X, y = get_formatted_data(eegp, eventp)
        Xs.extend(X)
        ys.extend(y)
    homogenous_ixs = [i for i in range(len(Xs)) if Xs[i].shape == (34, 1001)]
    Xs_homogenous = [Xs[i] for i in homogenous_ixs]
    ys_homogenous = [ys[i] for i in homogenous_ixs]
    print(f"\n #### Got data: {len(Xs_homogenous)} of {len(Xs)} EEG recordings ####\n")
    return np.array(Xs_homogenous, dtype=np.float64), np.array(ys_homogenous)


def get_formatted_data(eeg_file, event_file):
    eeg_data = load_eeg_run_data(eeg_file)
    raw = create_mne_raw(eeg_data)

    raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")

    events = load_events(event_file)

    # Create epochs
    picks = mne.pick_types(
        raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
    )
    event_id = {"standard": 0, "oddball": 1}  # Adjust based on your event coding

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=TMIN,
        tmax=TMAX,
        proj=True,
        picks=picks,
        baseline=None,
        preload=True,
    )

    X = epochs.get_data()
    y = epochs.events[:, -1]

    return X, y


def create_mne_raw(eeg_data):
    ch_names = [f"EEG{i:03d}" for i in range(1, eeg_data.shape[0] + 1)]
    ch_types = ["eeg"] * eeg_data.shape[0]
    info = mne.create_info(ch_names=ch_names, sfreq=SAMPLING_FREQ, ch_types=ch_types)
    raw = mne.io.RawArray(eeg_data, info)
    return raw


def load_eeg_run_data(run_path):
    data = scipy.io.loadmat(run_path)
    eeg_data = data["data_reref"][:34, :]  # Use only the first 34 channels (EEG data)
    eeg_data = eeg_data.astype(np.float64)
    print("EEG data shape:", eeg_data.shape)
    return eeg_data


def resample_eeg(eeg_data, target_freq=500):
    num_trials, num_channels, num_time_points = eeg_data.shape
    new_num_time_points = int(num_time_points * (target_freq / SAMPLING_FREQ))
    resampled_data = np.zeros((num_trials, num_channels, new_num_time_points))
    for trial in range(num_trials):
        for channel in range(num_channels):
            resampled_data[trial, channel] = signal.resample(
                eeg_data[trial, channel], new_num_time_points
            )
    return resampled_data


def get_event_eeg(eeg_data, event_time):
    event_eeg = eeg_data.T[
        event_time - TMIN * SAMPLING_FREQ : event_time + TMAX * SAMPLING_FREQ
    ].T
    return event_eeg


def balance_and_shuffle(X, y):
    # Separate majority and minority classes
    X_majority = X[y == 0]
    y_majority = y[y == 0]
    X_minority = X[y == 1]
    y_minority = y[y == 1]

    N = len(X_minority)
    # Upsample minority class
    X_minority_upsampled, y_minority_upsampled = resample(
        X_minority, y_minority, replace=False, n_samples=N, random_state=42
    )

    # Combine majority class with upsampled minority class
    X_balanced = np.vstack((X_majority[:N], X_minority_upsampled))
    y_balanced = np.hstack((y_majority[:N], y_minority_upsampled))

    # Shuffle the balanced dataset
    shuffle_indices = np.random.permutation(len(y_balanced))
    X_balanced_shuffled = X_balanced[shuffle_indices]
    y_balanced_shuffled = y_balanced[shuffle_indices]
    print(X_balanced.shape)
    print(y_balanced.shape)

    return X_balanced_shuffled, y_balanced_shuffled


if __name__ == "__main__":
    x, y = get_data()

    X_resampled, y_resampled = balance_and_shuffle(x, y)

    from eeg.plot_reproduction import assemble_classifier_CSPLDA
    from eeg.utils import get_cv, results

    clf = assemble_classifier_CSPLDA(10)
    cv = get_cv()
    score = results(clf, X_resampled, y_resampled, cv)
    print(score)

    """
    import pyriemann

    Xcov = pyriemann.estimation.Covariances("oas").fit_transform(X_resampled)
    FgMDM = pyriemann.classification.FgMDM()
    FgMDM_score = results(FgMDM, Xcov, y_resampled, cv)
    print(FgMDM_score)
    """


    """ Experiment notes:

        On non-resampled dataset:

        CSP+LDA (n=10) : 79% (i.e. random)
        CSP+LDA (n=30) : 80% (i.e. random)
        FgMDM          : 71% (i.e. ultra shit)

        On resampled dataset:

        CSP+LDA (n=10) : 57%
        CSP+LDA (n=30) : 61%
        FgMDM          : 78%

    --> this makes sense, it's actually pretty easy to classify these
        ERPs just looking at the Cz electrode visuall (c.f. Wikipedia
        figure 2 on P300 page).
    """











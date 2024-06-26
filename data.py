from mne import Epochs, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
import numpy as np


tmin, tmax = 1.0, 2.0
runs = [6, 10, 14]


def get_data(n_subjects=109):
    """ Set n_subjects to a small number for testing, 109 for full dataset """
    Xs = []
    ys = []
    for subject in range(1, n_subjects+1):
        X, y = get_formatted_data(subject)
        Xs.extend(X)
        ys.extend(y)
    homogenous_ixs = [i for i in range(len(Xs)) if Xs[i].shape==(64, 161)]
    Xs_homogenous = [Xs[i] for i in homogenous_ixs]
    ys_homogenous = [ys[i] for i in homogenous_ixs]
    print(f"We kept {len(Xs_homogenous)} of {len(Xs)} EEG recordings")
    return np.array(Xs_homogenous, dtype=np.float64), np.array(ys_homogenous)


def get_formatted_data(subject):
    """ Band pass filter + pick only EEG channels + format as Epoch objects """
    raw = get_raw_data(subject)
    raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
    epochs = Epochs(
        raw,
        event_id=["hands", "feet"],
        tmin=tmin,
        tmax=tmax,
        proj=True,
        picks=picks,
        baseline=None,
        preload=True,
    )
    X = epochs.copy().get_data(copy=False)
    y = epochs.events[:, -1] - 2
    return X, y


def get_raw_data(subject):
    """ Get raw EEGMI data from website or locally if already downloaded """
    raw_fnames = eegbci.load_data(subject, runs)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    eegbci.standardize(raw)  # set channel names
    montage = make_standard_montage("standard_1020") #or 1005?
    raw.set_montage(montage)
    raw.annotations.rename(dict(T1="hands", T2="feet"))
    raw.set_eeg_reference(projection=True)
    return raw




from mne import Epochs, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
import numpy as np

from eeg import physionet_runs


"""
    Physionet (dataset): https://physionet.org/content/eegmmidb/1.0.0/
    Tutorial: https://mne.tools/dev/auto_examples/decoding/decoding_csp_eeg.html
"""


tmin, tmax = 1.0, 2.0


def get_formatted_data(subject, bandpass, runs):
    """ Band pass filter + pick only EEG channels + format as Epoch objects """
    raw = get_raw_data(subject, runs)
    if bandpass:
        raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
    if runs == [1]:
        event_id = "rest"
    else:
        event_id = ["hands", "feet"]
    epochs = Epochs(
        raw,
        event_id=event_id,
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


def get_data(n_subjects=109, bandpass=True, reststate=False, formatter=get_formatted_data):
    """ Set n_subjects to a small number for testing, 109 for full dataset """
    if reststate:
        runs = [1]
    else:
        runs = physionet_runs
    Xs = []
    ys = []
    for subject in range(1, n_subjects+1):
        X, y = formatter(subject, bandpass, runs)
        Xs.extend(X)
        ys.extend(y)
    homogenous_ixs = [i for i in range(len(Xs)) if Xs[i].shape==(64, 161)]
    Xs_homogenous = [Xs[i] for i in homogenous_ixs]
    ys_homogenous = [ys[i] for i in homogenous_ixs]
    print(f"\n #### Got data: {len(Xs_homogenous)} of {len(Xs)} EEG recordings ####\n")
    return np.array(Xs_homogenous, dtype=np.float64), np.array(ys_homogenous)


def get_raw_data(subject, runs):
    """ Get raw EEGMI data from website or locally if already downloaded """
    raw_fnames = eegbci.load_data(subject, runs)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    eegbci.standardize(raw)  # set channel names
    montage = make_standard_montage("standard_1020") #or 1005?
    raw.set_montage(montage)
    if runs == [1]:
        raw.annotations.rename(dict(T0="rest"))
    else:
        raw.annotations.rename(dict(T1="hands", T2="feet"))
    raw.set_eeg_reference(projection=True)
    return raw



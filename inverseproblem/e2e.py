import mne
from mne import Epochs, pick_types
from mne.minimum_norm import apply_inverse_epochs, make_inverse_operator
import pyriemann

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from eeg import physionet_runs
from eeg.data import get_raw_data, tmax, tmin, get_data
from eeg.utils import results, get_cv, avg_power_matrix
from eeg.inverseproblem.leadfield import compute_forward_solution
from eeg.plot_reproduction import assemble_classifier_CSPLDA

method = "eLORETA"
snr = 3.0 #why??
lambda2 = 1.0 / snr**2


def get_J_y(n_subjects=109):
    J = []
    y = []
    for subject in range(1, n_subjects+1):
        sub_J, sub_y = get_subject_J(subject)
        y.extend(sub_y)
        J.extend(sub_J)

    homogenous_ixs = [i for i in range(len(J)) if True]#J[i].shape==(474, 161)]
    J_homogenous = [J[i] for i in homogenous_ixs]
    y_homogenous = [y[i] for i in homogenous_ixs]
    return np.array(J_homogenous), np.array(y_homogenous)


def get_subject_J(subject):
    raw = get_raw_data(subject, physionet_runs)
    raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
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
    noise_cov = mne.compute_covariance(epochs,
                                       tmax=tmax,
                                       method=["shrunk", "empirical"],
                                       rank=None, verbose=True)
    inverse_operator = make_inverse_operator(
        epochs.info, fwd, noise_cov, loose=0.2, depth=0.8)
    stcs = apply_inverse_epochs(
        epochs,
        inverse_operator,
        lambda2,
        method=method,
    )
    y = epochs.events[:, -1] - 2
    J = np.array([stc.data for stc in stcs])
    del noise_cov
    del inverse_operator
    del epochs
    del raw
    return J, y

"""
LDA on J (oct6)              ~ 49%
LDA on J (oct4)              ~ 49%
LDA on X                     ~ 49%
CSP+LDA (30 components) on X ~ 61.5%
"""


if __name__ == '__main__':
    fwd = compute_forward_solution()
    J, y = get_J_y()
    cv = get_cv()
    clf = assemble_classifier_CSPLDA(30)
    J_score = results(clf, J, y, cv) 
    print(f"CSP+LDA on J: {J_score}")



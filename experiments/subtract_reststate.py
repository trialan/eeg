
from mne.datasets import eegbci
from mne import Epochs, pick_types
from mne.io import concatenate_raws, read_raw_edf
from mne.channels import make_standard_montage

import matplotlib.pyplot as plt

from tqdm import tqdm
from mne.decoding import CSP, UnsupervisedSpatialFilter

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline

import pyriemann
from eeg.laplacian import (get_electrode_coordinates,
                           compute_scalp_eigenvectors_and_values,
                           create_triangular_dmesh, ED)
from eeg.ml import results
from eeg.data import get_formatted_data, get_data

"""
Goal of this experiment: download the rest-state data and re-run
Laplacian + FgMDM on the normalised (X - X_resting) data.

Runs: [6,10,14] are motor-imagery data, run 1 is baseline
Source: https://physionet.org/content/eegmmidb/1.0.0/

Subtracting the mean signal like this:

    mean_X_rs = np.mean(X_rs, axis=0)
    X_corrected = X - mean_X_rs

gives a score: 0.6148846960167715 (i.e. worse). Doing the ratio

    X_corrected = X / mean_X_rs

gives a score: 0.619916142557652 (i.e. same as without ratio).

This could be because the mean signal is averaged over all subjects, but
perhaps their individual means aren't so similar.

This current version of the code subtracts the individual's baseline
state of activity.

"""


def get_reststate_data(n_subjects=109, bandpass=True):
    return get_data(n_subjects, reststate=True)


def get_normalised_data(X_rs, n_subjects=109, bandpass=True):
    runs = [6, 10, 14]
    Xs = []
    ys = []
    for subject in range(1, n_subjects+1):
        X, y = get_formatted_data(subject, bandpass, runs)
        if X.shape == (64, 161):
            X = X / X_rs[subject - 1]
        Xs.extend(X)
        ys.extend(y)
    homogenous_ixs = [i for i in range(len(Xs)) if Xs[i].shape==(64, 161)]
    Xs_homogenous = [Xs[i] for i in homogenous_ixs]
    ys_homogenous = [ys[i] for i in homogenous_ixs]
    print(f"We kept {len(Xs_homogenous)} of {len(Xs)} EEG recordings")
    return np.array(Xs_homogenous, dtype=np.float64), np.array(ys_homogenous)

if __name__ == '__main__':
    X_rs, y_rs = get_reststate_data()
    X, y = get_normalised_data(X_rs)
    cv = ShuffleSplit(5, test_size=0.2, random_state=42)

    print("#############  FgMDM  #############")
    Xcov = pyriemann.estimation.Covariances('oas').fit_transform(X)
    FgMDM = pyriemann.classification.FgMDM()
    FgMDM_score = results(FgMDM, Xcov, y, cv)
    plt.axhline(y=FgMDM_score, linestyle='--', label='FgMDM (norm. subj.)')
    plt.legend()
    plt.show()



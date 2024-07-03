
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
from eeg.data import get_data

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
"""

def get_reststate_data(n_subjects=109, bandpass=True):
    return get_data(n_subjects, reststate=True)


if __name__ == '__main__':
    X_rs, y_rs = get_reststate_data()
    X, y = get_data()
    cv = ShuffleSplit(5, test_size=0.2, random_state=42)

    mean_X_rs = np.mean(X_rs, axis=0)
    X_corrected = X / mean_X_rs

    print("#############  FgMDM  #############")
    Xcov = pyriemann.estimation.Covariances('oas').fit_transform(X)
    FgMDM = pyriemann.classification.FgMDM()
    FgMDM_score = results(FgMDM, Xcov, y, cv)
    plt.axhline(y=FgMDM_score, linestyle='--', label='FgMDM (tmin/max=-1-4)')
    plt.legend()
    plt.show()



from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import pyriemann

from mne import Epochs, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.decoding import CSP, UnsupervisedSpatialFilter
from mne.io import concatenate_raws, read_raw_edf

from tqdm import tqdm

"""
First we define our data-related parameters
    - we set tmin and tmax as in the tutorial linked below
    - runs are [6,10,14] as in the tutorial, [5,9,13] tempting to use but
      represent action, not imagination
    - use a single subject (subject 1) for pipeline setup

Physionet page: https://physionet.org/content/eegmmidb/1.0.0/
Tutorial: https://mne.tools/dev/auto_examples/decoding/decoding_csp_eeg.html
"""

tmin, tmax = 1.0, 2.0
runs = [6, 10, 14]


def get_data():
    Xs = []
    ys = []
    for subject in range(1, 109+1):
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


def assemble_classifier_PCA_XGB(n_components):
    pca = PCA(n_components)
    xgb = XGBClassifier()
    clf = Pipeline([("PCA", pca), ("XGB", xgb)])
    return clf


def results(clf, X, y, cv):
    scores = cross_val_score(clf, X, y, cv=cv, n_jobs=None)
    class_balance = np.mean(y == y[0])
    class_balance = max(class_balance, 1.0 - class_balance)
    #print(f"Accuracy: {np.mean(scores)} / Chance level: {class_balance}")
    return np.mean(scores)


if __name__ == '__main__':
    X, y = get_data()
    cv = ShuffleSplit(5, test_size=0.2, random_state=42)
    component_numbers = list(range(1, 50, 5))

    print("PCA+XGB")
    X_rs = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
    scores = []
    for n_components in tqdm(component_numbers):
        clf = assemble_classifier_PCA_XGB(n_components)
        score = results(clf, X_rs, y, cv)
        scores.append(score)
    plt.plot(component_numbers, scores, marker='o', linestyle='-', label='PCA+XGB')
    plt.show()



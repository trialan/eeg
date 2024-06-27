from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline

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


@lru_cache(maxsize=None)
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
    montage = make_standard_montage("standard_1005")
    raw.set_montage(montage)
    raw.annotations.rename(dict(T1="hands", T2="feet"))
    raw.set_eeg_reference(projection=True)
    return raw


def assemble_classifer_PCACSPLDA(n_components):
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    pca = UnsupervisedSpatialFilter(PCA(n_components), average=False)
    clf = Pipeline([("PCA", pca), ("CSP", csp), ("LDA", lda)])
    return clf


def assemble_classifer_CSPLDA(n_components):
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    clf = Pipeline([("CSP", csp), ("LDA", lda)])
    return clf


def assemble_classifer_PCAFgMDM(n_components):
    pca = UnsupervisedSpatialFilter(PCA(n_components), average=False)
    FgMDM = pyriemann.classification.FgMDM()
    clf = Pipeline([("PCA", pca), ("FgMDM", FgMDM)])
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

    print("CSP+LDA")
    scores = []
    for n_components in tqdm(component_numbers):
        clf = assemble_classifer_CSPLDA(n_components)
        score = results(clf, X, y, cv)
        scores.append(score)
    plt.plot(component_numbers, scores, marker='o', linestyle='-', label='CSP+LDA')


    print("PCA+FgMDM")
    scores = []
    for n_components in tqdm(component_numbers):
        n_epochs, n_channels, n_times = X.shape
        X_reshaped = X.reshape(n_times * n_epochs, n_channels)
        pca = PCA(n_components=n_components)
        pca.fit(X_reshaped)

        X_pca = np.array([pca.transform(epoch.T).T for epoch in X])
        Xcov = pyriemann.estimation.Covariances('oas').fit_transform(X_pca)

        FgMDM = pyriemann.classification.FgMDM()
        score = results(FgMDM, Xcov, y, cv)
        scores.append(score)
    plt.plot(component_numbers, scores, marker='o', linestyle='-', label='PCA+FgMDM')

    #'oas' because: https://github.com/pyRiemann/pyRiemann/issues/65
    print("FgMDM")
    Xcov = pyriemann.estimation.Covariances('oas').fit_transform(X)
    FgMDM = pyriemann.classification.FgMDM()
    FgMDM_score = results(FgMDM, Xcov, y, cv)
    plt.axhline(y=FgMDM_score, linestyle='--', label='FgMDM')

    print("PCA+CSP+LDA")
    scores = []
    for n_components in tqdm(component_numbers):
        clf = assemble_classifer_PCACSPLDA(n_components)
        score = results(clf, X, y, cv)
        scores.append(score)
    plt.plot(component_numbers, scores, marker='o', linestyle='-', label='PCA+CSP+LDA')


    plt.xlabel("Number of components")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.show()



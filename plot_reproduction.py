from eeg.data import get_data
import matplotlib.pyplot as plt

from tqdm import tqdm
from mne.decoding import CSP, UnsupervisedSpatialFilter

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

import pyriemann

from eeg.laplacian import compute_scalp_eigenvectors_and_values
from eeg.utils import results, get_cv
from eeg.experiments.eigen_fgmdm import OldED

"""
    This script reproduces fig 3(a) from Xu et. al.
    https://hal.science/hal-03477057/documen://hal.science/hal-03477057/document
"""


def assemble_classifer_PCACSPLDA(n_components):
    lda = LinearDiscriminantAnalysis()
    pca = UnsupervisedSpatialFilter(PCA(n_components), average=False)
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    clf = Pipeline([("PCA", pca), ("CSP", csp), ("LDA", lda)])
    return clf


def assemble_classifier_CSPLDA(n_components):
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    clf = Pipeline([("CSP", csp), ("LDA", lda)])
    return clf


def assemble_classifier_LaplacianCSPLDA(n_components, eigenvectors):
    ed = UnsupervisedSpatialFilter(OldED(n_components, eigenvectors), average=False)
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    clf = Pipeline([("ED", ed), ("CSP", csp), ("LDA", lda)])
    return clf


if __name__ == '__main__':
    X, y = get_data()
    cv = get_cv()
    eigenvectors, eigenvals = compute_scalp_eigenvectors_and_values()

    component_numbers = list(range(2, 64))

    print("Laplacian+FgMDM")
    scores = []
    for n_components in tqdm(component_numbers):
        n_epochs, n_channels, n_times = X.shape
        X_reshaped = X.reshape(n_times * n_epochs, n_channels)
        ed = OldED(n_components, eigenvectors)
        X_ed = np.array([ed.transform(epoch.T).T for epoch in X])
        Xcov = pyriemann.estimation.Covariances('oas').fit_transform(X_ed)

        FgMDM = pyriemann.classification.FgMDM()
        score = results(FgMDM, Xcov, y, cv)
        scores.append(score)
    plt.plot(component_numbers, scores, marker='o', linestyle='-', label='Laplacian+FgMDM')


    print("Laplacian+CSP+LDA")
    scores = []
    for n_components in tqdm(component_numbers):
        clf = assemble_classifier_LaplacianCSPLDA(n_components, eigenvectors)
        score = results(clf, X, y, cv)
        scores.append(score)
    plt.plot(component_numbers, scores, marker='o', linestyle='-', label='Laplacian+CSP+LDA')


    print("CSP+LDA")
    scores = []
    for n_components in tqdm(component_numbers):
        clf = assemble_classifier_CSPLDA(n_components)
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



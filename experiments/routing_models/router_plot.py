import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import numpy as np
import sklearn
from sklearn.decomposition import PCA
import pyriemann

from eeg.experiments.ensemble import EDFgMDM, OldED
from eeg.laplacian import compute_scalp_eigenvectors_and_values
from eeg.ml import (assemble_classifer_PCACSPLDA,
                    assemble_classifer_CSPLDA)
from eeg.utils import (results, get_covariances,
                       get_cv, set_seed, avg_power_matrix)
from mne.decoding import CSP, UnsupervisedSpatialFilter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline


def assemble_classifier_LaplacianCSPLDA(n_components, eigenvectors):
    ed = UnsupervisedSpatialFilter(OldED(n_components, eigenvectors), average=False)
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    clf = Pipeline([("ED", ed), ("CSP", csp), ("LDA", lda)])
    return clf


if __name__ == '__main__':

    with h5py.File('router_data.h5', 'r') as f:
        X_router = f['X_router'][:]
        y_router = f['y_router'][:]

    X_router_cov = get_covariances(X_router)
    cv = get_cv()
    eigenvectors, eigenvals = compute_scalp_eigenvectors_and_values()
    component_numbers = list(range(2, 25))

    set_seed()

    print("Laplacian + FgMDM (64)")
    ed_fgmdm_scores = []
    for n_components in tqdm(component_numbers):
        ed_fgmdm = EDFgMDM(n_components=n_components,
                           eigenvectors=eigenvectors)
        score = results(ed_fgmdm, X_router_cov, y_router, cv)
        ed_fgmdm_scores.append(score)
    plt.plot(component_numbers, ed_fgmdm_scores,
             marker='o', linestyle='-', label='Laplacian + FgMDM')

    print("CSP+LDA")
    csp_lda_scores = []
    for n_components in tqdm(component_numbers):
        router = assemble_classifer_CSPLDA(n_components)
        score = results(router, X_router, y_router, cv)
        csp_lda_scores.append(score)
    plt.plot(component_numbers, csp_lda_scores,
             marker='o', linestyle='-', label='CSP+LDA')

    print("Laplacian+CSP+LDA")
    lap_csp_lda_scores = []
    for n_components in tqdm(component_numbers):
        clf = assemble_classifier_LaplacianCSPLDA(n_components, eigenvectors)
        score = results(clf, X_router, y_router, cv)
        lap_csp_lda_scores.append(score)
    plt.plot(component_numbers, lap_csp_lda_scores,
             marker='o', linestyle='-', label='Laplacian+CSP+LDA')


    print("FgMDM")
    FgMDM = pyriemann.classification.FgMDM()
    FgMDM_score = results(FgMDM, X_router_cov, y_router, cv)
    plt.axhline(y=FgMDM_score, color='r', linestyle='--', label='FgMDM')


    print("Laplacian+LDA")
    ed_lda_scores = []
    for n_components in tqdm(component_numbers):
        ed = OldED(n_components, eigenvectors[::-1])
        X_ed = np.array([ed.transform(epoch.T).T for epoch in X_router])
        X_ap = np.array([avg_power_matrix(m) for m in X_ed])
        lda = LinearDiscriminantAnalysis()
        score = results(lda, X_ap, y_router, cv)
        ed_lda_scores.append(score)
    plt.plot(component_numbers, ed_lda_scores,
             marker='o', linestyle='-', label='Laplacian+LDA')


    plt.legend()
    plt.show()



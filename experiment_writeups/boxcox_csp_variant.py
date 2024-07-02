import random
from eeg.data import get_data
import matplotlib.pyplot as plt

from tqdm import tqdm
from mne.decoding import CSP, UnsupervisedSpatialFilter

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox

import pyriemann
from eeg.laplacian import (get_electrode_coordinates,
                           compute_scalp_eigenvectors_and_values,
                           get_256D_eigenvectors,
                           create_triangular_dmesh, ED)
from eeg.ml import results
from eeg.utils import avg_power_matrix

"""
Experimenting with variants of CSP:
    1. Varying log = {True, None} like this:

        def assemble_classifer_CSPLDA(n_components, log):
            lda = LinearDiscriminantAnalysis()
            csp = CSP(n_components=n_components,
                      reg=None,
                      log=log, #key line
                      norm_trace=False)
            clf = Pipeline([("CSP", csp), ("LDA", lda)])
            return clf

       makes no difference at all. This makes sense because reading
       the mne code we see that when transform_into ='average_power', then
       regardless of the log argument, a log transform is taken to
       standardise the features. The log arg must be None when
       transform_into == 'csp_space', but when we do that, we aren't
       dropping the time dimension, so we can't use LDA/SVC/normal CLFs.
       However we could use FgMDM. Let's try this next.
    2. FgMDM + CSP space: didn't do great, see plot_CSPFgMDM.
    3. "LSP" (invented term): Laplacian Spatial Patterns:
        a) project the data into the eigenbasis
        b) take the average power (this drops the time dimension)
        c) boxcox it for normalisation / scaling
        d) LDA classifier

        Results are plotted on the GitHub, interesting that they are linear
        in the number of components.
"""


def plot_CSPFgMDM(X, y, cv, component_numbers):
    #'oas' because: https://github.com/pyRiemann/pyRiemann/issues/65
    print("CSP+FgMDM")
    scores = []
    for n_components in tqdm(component_numbers):
        csp = CSP(n_components=n_components, reg=None,
                  log=None, norm_trace=False, transform_into='csp_space')
        X_csp = csp.fit_transform(X, y)
        Xcov = pyriemann.estimation.Covariances('oas').fit_transform(X_csp)

        FgMDM = pyriemann.classification.FgMDM()
        score = results(FgMDM, Xcov, y, cv)
        scores.append(score)
    plt.plot(component_numbers, scores, marker='o', linestyle='-', label='CSP+FgMDM')

    plt.axhline(y=0.6199, linestyle='--', label='FgMDM')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    X, y = get_data()
    cv = ShuffleSplit(5, test_size=0.2, random_state=42)
    component_numbers = list(range(2, 64))
    print(f"Shape: {X.shape}")

    xyz_coords = get_electrode_coordinates()
    mesh = create_triangular_dmesh(xyz_coords)
    #eigenvectors = get_256D_eigenvectors()
    eigenvectors, eigenvals = compute_scalp_eigenvectors_and_values(mesh)

    np.random.seed(3)
    print("Laplacian+FgMDM")
    for j in range(50):
        print(f"{j} of 4")
        np.random.shuffle(eigenvectors)
        scores = []
        for n_components in tqdm(component_numbers):
            n_epochs, n_channels, n_times = X.shape
            X_reshaped = X.reshape(n_times * n_epochs, n_channels)
            ed = ED(n_components, eigenvectors)
            X_ed = np.array([ed.transform(epoch.T).T for epoch in X])
            Xcov = pyriemann.estimation.Covariances('oas').fit_transform(X_ed)

            FgMDM = pyriemann.classification.FgMDM()
            score = results(FgMDM, Xcov, y, cv)
            scores.append(score)
        print(max(scores))
        plt.plot(component_numbers, scores, marker='o', linestyle='-', label='Laplacian+FgMDM')
    plt.show()





    1/0
    for j in range(5):
        print(f"{j} of 4")
        np.random.shuffle(eigenvectors)
        scores = []
        for n_components in tqdm(list(range(2,64))):
            ed = ED(n_components, eigenvectors)
            X_ed = np.array([ed.transform(sub_X.T).T for sub_X in X])
            X_ap = np.array([avg_power_matrix(sub_X) for sub_X in X_ed])

            lda = LinearDiscriminantAnalysis()
            score = results(lda, X_ap, y, cv)
            scores.append(score)

        plt.plot(component_numbers, scores, marker='o', linestyle='-')
    plt.show()


    improving_components = [0]
    for i, score in enumerate(scores):
        if i > 0:
            if score > scores[i-1]:
                improving_components.append(i)


    nscores = []
    for n_components in tqdm(range(2, len(improving_components))):
        ed = ED(n_components, eigenvectors)
        X_ed = np.array([ed.transform(sub_X.T).T[:, improving_components]
                         for sub_X in X])
        X_ap = np.array([avg_power_matrix(sub_X) for sub_X in X_ed])

        lda = LinearDiscriminantAnalysis()
        score = results(lda, X_ap, y, cv)
        nscores.append(score)

    plt.plot(list(range(2,len(improving_components))), nscores, marker='o', linestyle='-', label='LSP+LDA')
    plt.legend()
    plt.show()




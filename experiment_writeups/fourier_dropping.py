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
File for experimenting with alternative ways to drop the time dimension.
At present we drop the time dimension with avg_power_matrix which takes
in a matrix of time series and returns a vector of power averages. So here
we experiment with other ways of collapsing the time dimension.

"""



if __name__ == '__main__':
    X, y = get_data(2)
    cv = ShuffleSplit(5, test_size=0.2, random_state=42)

    xyz_coords = get_electrode_coordinates()
    mesh = create_triangular_dmesh(xyz_coords)
    eigenvectors, eigenvals = compute_scalp_eigenvectors_and_values(mesh)

    component_numbers = list(range(2, 64))
    scores = []
    for n_components in tqdm(component_numbers):
        ed = ED(n_components, eigenvectors)
        X_ed = np.array([ed.transform(sub_X.T).T for sub_X in X])
        X_ap = np.array([avg_power_matrix(sub_X) for sub_X in X_ed])

        lda = LinearDiscriminantAnalysis()
        score = results(lda, X_ap, y, cv)
        scores.append(score)

    plt.plot(component_numbers, scores, marker='o', linestyle='-')
    plt.show()




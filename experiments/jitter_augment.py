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
                           get_256D_eigenvectors,
                           compute_scalp_eigenvectors_and_values,
                           create_triangular_dmesh, ED)
from eeg.ml import results
from eeg.data import get_formatted_data, get_data
from eeg.utils import avg_power_matrix, jitter
from gtda.time_series import TakensEmbedding



def augment_subX(subX):
    assert subX.shape == (64, 161)
    new_subX = list(subX)
    for j in range(256 - 64):
        ix = j%64
        se = np.std(subX[ix]) / np.sqrt(161)
        new_electrode = jitter(subX[ix], sigma=se/300)
        new_subX.append(new_electrode)
    new_subX = np.array(new_subX)
    assert new_subX.shape == (256, 161)
    return new_subX


def transform_data(X):
    X_prime = np.array([augment_subX(subX) for subX in tqdm(X)])
    return X_prime



if __name__ == '__main__':
    Xr, y = get_data()
    X = transform_data(Xr)
    cv = ShuffleSplit(5, test_size=0.2, random_state=42)

    eigenvectors = get_256D_eigenvectors()
    xyz_coords = get_electrode_coordinates()
    mesh = create_triangular_dmesh(xyz_coords)
    eigenvectors, eigenvals = compute_scalp_eigenvectors_and_values(mesh)

    components = list(range(1,len(eigenvectors)))
    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    i1/0
    scores = []
    components = list(range(50,len(eigenvectors)))
    for n_components in tqdm(components):
        ed = ED(n_components, eigenvectors)
        X_ed = np.array([ed.transform(sub_X.T).T for sub_X in X])
        X_ap = np.array([avg_power_matrix(sub_X) for sub_X in X_ed])
        clf = LinearDiscriminantAnalysis()
        #clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        score = results(clf, X_ap, y, cv)
        print(score)
    plt.plot(components, scores, marker='o', linestyle='-', label='LSP+LDA (256D)')
    plt.legend()
    plt.show()




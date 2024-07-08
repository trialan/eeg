from eeg.data import get_data
import math
from scipy.special import sph_harm
import matplotlib.pyplot as plt

from tqdm import tqdm
from mne.decoding import CSP, UnsupervisedSpatialFilter

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

import pyriemann
from eeg.ml import results, assemble_classifer_CSPLDA

from gtda.time_series import TakensEmbedding
from eeg.utils import avg_power_matrix

from eeg.laplacian import (get_electrode_coordinates,
                           compute_scalp_eigenvectors_and_values,
                           get_256D_eigenvectors,
                           create_triangular_dmesh)

def augment_subX(subX, TE):
    assert subX.shape == (64, 161)
    new_subX = []#list(subX)
    for ts in subX:
        new_electrodes = takens_aug(ts, TE)
        for new_electrode in new_electrodes:
            new_subX.append(new_electrode)
    new_subX = np.array(new_subX[:256])
    #assert new_subX.shape == (192, 56)
    return new_subX


def transform_data(X, TE):
    X_prime = np.array([augment_subX(subX, TE) for subX in tqdm(X)])
    return X_prime


class ED(BaseEstimator, TransformerMixin):
    """ This is like sklearn's PCA class, but for Eigen-decomposition (ED). """
    def __init__(self, n_components, eigenvectors):
        self.n_components = n_components
        self.eigenvectors = eigenvectors

    def fit(self, X, y=None):
        if self.eigenvectors is None:
            raise ValueError("Eigenvectors are not set.")
        return self

    def transform(self, X):
        n_channels, n_times = X.shape
        selected_eigenvectors = self.eigenvectors[:, :self.n_components]
        X_transformed = selected_eigenvectors.T @ X
        return X_transformed


def takens_aug(ts, TE):
    x  = TE.transform(ts.reshape(1, -1))
    return x[0]


def get_sh_eigenvectors(num_harmonics):
    l_max = math.floor(math.sqrt(num_harmonics) - 1)

    def compute_spherical_harmonics(l_max, theta, phi):
        harmonics = []
        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                Y_lm = sph_harm(m, l, phi, theta)
                harmonics.append(Y_lm)
        return harmonics

    def generate_high_res_mesh(num_points):
        phi = np.linspace(0, 2 * np.pi, num_points)
        theta = np.linspace(0, np.pi/2, num_points)
        phi, theta = np.meshgrid(phi, theta)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return x, y, z, theta, phi

    num_points = 10  # Increase this number for higher resolution
    x, y, z, theta, phi = generate_high_res_mesh(num_points)

    spherical_harmonics = compute_spherical_harmonics(l_max, theta, phi)
    return spherical_harmonics



if __name__ == '__main__':
    Xr, y = get_data()
    cv = ShuffleSplit(5, test_size=0.2, random_state=42)

    xyz_coords = get_electrode_coordinates()
    mesh = create_triangular_dmesh(xyz_coords)
    eigenvectors, eigenvals = compute_scalp_eigenvectors_and_values(mesh)

    TE = TakensEmbedding(time_delay=25, dimension=4)
    TE.fit(Xr[0]) #This does nothing

    X = transform_data(Xr, TE)
    print("FgMDM")
    Xcov = pyriemann.estimation.Covariances('oas').fit_transform(X)
    FgMDM = pyriemann.classification.FgMDM()
    FgMDM_score = results(FgMDM, Xcov, y, cv)
    print(FgMDM_score)

    """
    #eigenvectors = get_256D_eigenvectors()
    components = list(range(1,len(eigenvectors)))

    scores = []
    for n_components in tqdm(components):
        ed = ED(n_components, eigenvectors)
        x_ed = np.array([ed.transform(sub_x) for sub_x in Xr])
        x_ap = np.array([avg_power_matrix(sub_x) for sub_x in x_ed])
        clf = LinearDiscriminantAnalysis()
        score = results(clf, x_ap, y, cv)
        print(score)
    plt.plot(components, scores, marker='o', linestyle='-', label='LSP+LDA (256D)')
    plt.legend()
    plt.show()
    """



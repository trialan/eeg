import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import spharapy.trimesh as tm
from scipy.spatial import Delaunay, ConvexHull
import spharapy.trimesh as trimesh
import spharapy.spharabasis as sb
import spharapy.datasets as sd
import random
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin

from eeg.data import *
from eeg.laplacian import (get_electrode_coordinates,
                           create_triangular_dmesh,
                           compute_scalp_eigenvectors_and_values,
                           ED)


def get_coefficients(eigenvectors, subjects):
    ed = ED(64, eigenvectors) #prepare for decomposition
    X, y = get_data(subjects)
    label = y # recording label, ie 'hands' or 'feet' with 0,1
    coeffs0_vs_t_for_epoch = [] #coefficients through time corresponding to label 0
    coeffs1_vs_t_for_epoch = [] #...to label 1
    for i in range(len(label)):
        if (not label[i]):
            coeffs0_vs_t_for_epoch.append(ed.transform(X[i].T).T)
        else:
            coeffs1_vs_t_for_epoch.append(ed.transform(X[i].T).T)
    Coeffs0_vs_t_for_epoch = np.array(coeffs0_vs_t_for_epoch)
    Coeffs1_vs_t_for_epoch = np.array(coeffs1_vs_t_for_epoch)
    return Coeffs0_vs_t_for_epoch, Coeffs1_vs_t_for_epoch


if __name__ == '__main__':
    xyz_coords = get_electrode_coordinates()
    mesh = create_triangular_dmesh(xyz_coords)
    eigenvectors, eigenvalues = compute_scalp_eigenvectors_and_values(mesh)
    coeffs0_vs_t_for_epoch, coeffs1_vs_t_for_epoch = get_coefficients(eigenvectors, 1)

    Coeffs0 = coeffs0_vs_t_for_epoch
    Coeffs1 = coeffs1_vs_t_for_epoch
    fourier0 = np.mean(np.fft.fft(Coeffs0), axis = 0)
    fourier1 = np.mean(np.fft.fft(Coeffs1), axis = 0)
    freq = np.fft.fftfreq(fourier0.shape[-1])

    print(fourier0.shape)

    log_power_spectrum0 = np.log10(np.abs(fourier0[2])**2)
    log_power_spectrum1 = np.log10(np.abs(fourier1[2])**2)
    plt.plot(freq, log_power_spectrum0, label = '0')
    plt.plot(freq, log_power_spectrum1, label = '1')
    plt.legend()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit, cross_val_score
from eeg.laplacian import (get_electrode_coordinates,
                           create_triangular_dmesh,
                           compute_scalp_eigenvectors_and_values,
                           ED)
from tqdm import tqdm

import pyriemann
from eeg.laplacian import (get_electrode_coordinates,
                           compute_scalp_eigenvectors_and_values,
                           create_triangular_dmesh, ED)
from eeg.ml import results
from eeg.data import get_formatted_data, get_data


def get_hands_feet_coefficients(eigenvectors, subjects):
    ed = ED(64, eigenvectors) #prepare for decomposition
    X, y = get_data()
    label = y # recording label, ie 'hands' or 'feet' with 0,1
    coeffs0_vs_t_for_epoch = [] #coefficients through time corresponding to label 0
    coeffs1_vs_t_for_epoch = [] #...to label 1
    for i in tqdm(range(len(label)), desc="getting coefficients"):
        if (not label[i]):
            coeffs0_vs_t_for_epoch.append(ed.transform(X[i].T).T)
        else:
            coeffs1_vs_t_for_epoch.append(ed.transform(X[i].T).T)
    Coeffs0_vs_t_for_epoch = np.array(coeffs0_vs_t_for_epoch)
    Coeffs1_vs_t_for_epoch = np.array(coeffs1_vs_t_for_epoch)
    return Coeffs0_vs_t_for_epoch, Coeffs1_vs_t_for_epoch


def get_fourier_data(eigenvectors):
    ed = ED(64, eigenvectors) #prepare for decomposition
    X, y = get_data()
    label = y # recording label, ie 'hands' or 'feet' with 0,1
    coeffs_vs_t_for_epoch = []
    for i in tqdm(range(len(label)), desc="getting coefficients"):
        coeffs_vs_t_for_epoch.append(ed.transform(X[i].T).T)
    coeffs = np.array(np.fft.fft(coeffs_vs_t_for_epoch))
    return coeffs, y



if __name__ == '__main__':
    yz_coords = get_electrode_coordinates()
    mesh = create_triangular_dmesh(xyz_coords)
    eigenvectors, eigenvalues = compute_scalp_eigenvectors_and_values(mesh)

    X, y = get_fourier_data(eigenvectors)

    cv = ShuffleSplit(5, test_size=0.2, random_state=42)

    print("#############  FgMDM  #############")
    Xcov = pyriemann.estimation.Covariances('oas').fit_transform(np.abs(X))
    FgMDM = pyriemann.classification.FgMDM()
    FgMDM_score = results(FgMDM, Xcov, y, cv)
    plt.axhline(y=FgMDM_score, linestyle='--', label='FgMDM (norm. subj.)')
    plt.legend()
    plt.show()

    1/0
    #Below are Anthony's plots:
    coeffs0_vs_t_for_epoch, coeffs1_vs_t_for_epoch = get_hands_feet_coefficients(eigenvectors, 1)

    Coeffs0 = coeffs0_vs_t_for_epoch
    Coeffs1 = coeffs1_vs_t_for_epoch

    fourier0 = np.mean(np.fft.fft(Coeffs0), axis = 0)
    fourier1 = np.mean(np.fft.fft(Coeffs1), axis = 0)
    freq = np.fft.fftfreq(fourier0.shape[-1])

    print(fourier0.shape)

    log_power_spectrum0 = np.log10(np.abs(fourier0[2])**2) #picked mode 2 because of Xu et al paper
    log_power_spectrum1 = np.log10(np.abs(fourier1[2])**2)
    plt.plot(freq, log_power_spectrum0, label = '0')
    plt.plot(freq, log_power_spectrum1, label = '1')
    plt.axvline(x=-0.13043, color='r', linestyle='dashed', linewidth=0.5, label='Vertical Line')

    plt.legend()
    plt.show()



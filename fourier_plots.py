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
    X, y = get_data(subjects)
    label = y # recording label, ie 'hands' or 'feet' with 0,1
    print(label.shape)
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
    xyz_coords = get_electrode_coordinates()
    mesh = create_triangular_dmesh(xyz_coords)
    eigenvectors, eigenvalues = compute_scalp_eigenvectors_and_values(mesh)


    #Below are Anthony's plots:
    coeffs0_vs_t_for_epoch, coeffs1_vs_t_for_epoch = get_hands_feet_coefficients(eigenvectors, 1)

    fourier0 = np.mean(np.fft.fft(coeffs0_vs_t_for_epoch), axis=0)
    fourier1 = np.mean(np.fft.fft(coeffs1_vs_t_for_epoch), axis=0)
    
    num_eigenmodes = 20 #about the number of 'good' components

    # Determine the grid size for plotting
    cols = 4  # Number of columns in the grid
    rows = int(np.ceil(num_eigenmodes / cols))  # Number of rows, calculated to fit all signals

    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))

    # Flatten axes array for easy iteration
    axes = axes.flatten()

    # Loop through each signal in Coeffs0, compute the Fourier transform, and plot it
    for i in range(num_eigenmodes):
        signal0 = fourier0[i]
        signal1 = fourier1[i]
        freqs = np.fft.fftfreq(signal0.shape[-1], 0.01)  # Assuming 10ms intervals
        log_power_spectrum0 = np.log10(np.abs(signal0)**2)
        log_power_spectrum1 = np.log10(np.abs(signal1)**2)

        axes[i].plot(freqs, log_power_spectrum0)
        axes[i].plot(freqs, log_power_spectrum1)
        axes[i].set_title(f'FFT of Eigenmode {i}')
        axes[i].set_xlabel('Frequency (Hz)')
        axes[i].set_ylabel('Log Power Spectrum')

    # Hide any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()



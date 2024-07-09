import numpy as np
import pywt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from eeg.laplacian import (get_electrode_coordinates,
                           create_triangular_dmesh,
                           compute_scalp_eigenvectors_and_values,
                           ED)
from tqdm import tqdm

import pyriemann
from eeg.laplacian import (get_electrode_coordinates,
                           compute_scalp_eigenvectors_and_values,
                           create_triangular_dmesh, ED)
from scipy import signal
from scipy.ndimage import uniform_filter
from eeg.ml import results
from eeg.data import get_formatted_data, get_data
from eeg.utils import avg_power_matrix

def get_hands_feet_coefficients(eigenvectors, subjects, X, y):
    ed = ED(64, eigenvectors) #prepare for decomposition
    label = y # recording label, ie 'hands' or 'feet' with 0,1
    coeffs0_vs_t_for_epoch = [] #coefficients through time corresponding to label 0
    coeffs1_vs_t_for_epoch = [] #...to label 1
    for i in tqdm(range(len(label)), desc="getting coefficients"):
        if (not label[i]):
            coeffs0_vs_t_for_epoch.append(ed.transform(X[i]))
        else:
            coeffs1_vs_t_for_epoch.append(ed.transform(X[i]))
    Coeffs0_vs_t_for_epoch = np.array(coeffs0_vs_t_for_epoch)
    Coeffs1_vs_t_for_epoch = np.array(coeffs1_vs_t_for_epoch)
    return Coeffs0_vs_t_for_epoch, Coeffs1_vs_t_for_epoch

def wavelet_coherence(signal1, signal2, wavelet='morl', scales=np.arange(1, 128), sampling_frequency=100):
    cwt1, freqs = pywt.cwt(signal1, scales, wavelet, sampling_period=1/sampling_frequency)
    cwt2, _ = pywt.cwt(signal2, scales, wavelet, sampling_period=1/sampling_frequency)
    
    # Compute the cross wavelet transform
    Wxy = cwt1 * np.conj(cwt2)
    
    # Compute the wavelet coherence
    S1 = np.abs(cwt1)**2
    S2 = np.abs(cwt2)**2
    Sxy = np.abs(Wxy)**2
    # Smooth the wavelet power spectra
    S1_smooth = uniform_filter(S1, size=(10, 10))
    S2_smooth = uniform_filter(S2, size=(10, 10))
    Sxy_smooth = uniform_filter(Sxy, size=(10, 10))

    coherence = Sxy_smooth / (S1_smooth * S2_smooth)

    return coherence, freqs

def compute_wavelet_spectrum(signal, wavelet='morl', scales=np.arange(1, 128), sampling_frequency=100):
    # Compute the wavelet transform
    cwt_coeffs, freqs = pywt.cwt(signal, scales, wavelet, sampling_period=1/sampling_frequency)
    
    # Collapse the time dimension (average over time)
    spectrum = np.mean(np.abs(cwt_coeffs), axis=1)
    
    return spectrum, freqs

def decompose_signal(signal, wavelet='db4', level=4):
    """
    Decompose an EEG signal using Daubechies wavelet.
    
    Parameters:
    - signal: Time series data (EEG signal).
    - wavelet: Type of wavelet to use (default is 'db4' for Daubechies wavelet).
    - level: Level of decomposition.
    
    Returns:
    - coeffs: Wavelet decomposition coefficients.
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return coeffs

if __name__ == '__main__':
    xyz_coords = get_electrode_coordinates()
    mesh = create_triangular_dmesh(xyz_coords)
    eigenvectors, eigenvalues = compute_scalp_eigenvectors_and_values()
    subjects = 50
    X, y = get_data(subjects)
    coeffs0_vs_t_for_epoch, coeffs1_vs_t_for_epoch = get_hands_feet_coefficients(eigenvectors, subjects, X, y)
    label = y
    #signal1 = avg_power_matrix(coeffs0_vs_t_for_epoch)[15] 

    signal1  = np.mean(coeffs0_vs_t_for_epoch[:,5]/np.linalg.norm(coeffs0_vs_t_for_epoch[:,5],axis=0), axis=0)
    signal2  = coeffs0_vs_t_for_epoch[401][5]/np.linalg.norm(coeffs0_vs_t_for_epoch[401],axis=0)
    print(label[402],label[401])
    #signal1  = coeffs0_vs_t_for_epoch[220][5]
    #signal2  = coeffs1_vs_t_for_epoch[220][5]
    print(signal1.shape)
    t = range(0,len(signal1))
    # Compute wavelet spectrum
    spectrum1, freqs = compute_wavelet_spectrum(signal1)
    spectrum2, freqs = compute_wavelet_spectrum(signal2)
    print(spectrum1.shape)
    
    # Computer coeffs
    coeffs = decompose_signal(signal2)
    # Plot decomposition
    plt.figure(figsize=(12, 8))
    plt.subplot(len(coeffs) + 1, 1, 1)
    plt.plot(t, signal2, label='Original Signal')
    plt.legend(loc='upper right')
    plt.title('EEG Signal and Wavelet Decomposition (db4)')

    for i, coeff in enumerate(coeffs):
        plt.subplot(len(coeffs) + 1, 1, i + 2)
        plt.plot(coeff, label=f'Detail Coefficients (Level {i})' if i != 0 else 'Approximation Coefficients (Level 0)')
        plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    # Plot the wavelet spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, spectrum1)
    plt.plot(freqs, spectrum2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Wavelet Spectrum (Collapsed over Time)')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, which="both", ls="--")
    plt.show()
  
    # Plot wavelet coherence
    
    # Determine the grid size for plotting
    num_eigenmodes=10
    cols = 3  # Number of columns in the grid
    rows = int(np.ceil(num_eigenmodes / cols))  # Number of rows, calculated to fit all signals
     # Create a figure with a grid of subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))

    # Flatten axes array for easy iteration
    axes = axes.flatten()
    for i in range(num_eigenmodes):
        split = int(len(coeffs0_vs_t_for_epoch)*0.2)
        print(split)
        print(coeffs0_vs_t_for_epoch.shape)
        signal1 = coeffs0_vs_t_for_epoch[212][i]
        signal2  = np.mean(coeffs0_vs_t_for_epoch[201:210],axis=0)[i]
        coherence, freqs = wavelet_coherence(signal1, signal2)
        vmin = 0.
        vmax = 4.
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        axes[i].contourf(t, freqs, coherence, 100, cmap='viridis', norm=norm)
        #axes[i].colorbar(label='Wavelet Coherence')
        #fig.colorbar(contour, ax=axes[i], label='Wavelet Coherence')
        axes[i].set_ylabel('Frequency (Hz)')
        axes[i].set_xlabel('Time (s)')
        #axes[i].title('Wavelet Coherence')
    #plt.yscale('log')
    plt.tight_layout()
    plt.show()


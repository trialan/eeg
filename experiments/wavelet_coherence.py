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

def get_hands_feet_coefficients(eigenvectors, subjects):
    ed = ED(64, eigenvectors) #prepare for decomposition
    X, y = get_data(subjects)
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

def wavelet_coherence(signal1, signal2, wavelet='morl', scales=np.arange(1, 128), sampling_frequency=0.1):
    """
    Compute wavelet coherence between two signals.
    
    Parameters:
    - signal1: First time series data.
    - signal2: Second time series data.
    - wavelet: Type of wavelet to use (default is 'morl' for Morlet wavelet).
    - scales: Scales to use for the wavelet transform.
    - sampling_frequency: Sampling frequency of the signals.
    
    Returns:
    - coherence: Wavelet coherence values.
    - freqs: Corresponding frequencies.
    """
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

if __name__ == '__main__':
    xyz_coords = get_electrode_coordinates()
    mesh = create_triangular_dmesh(xyz_coords)
    eigenvectors, eigenvalues = compute_scalp_eigenvectors_and_values()

    coeffs0_vs_t_for_epoch, coeffs1_vs_t_for_epoch = get_hands_feet_coefficients(eigenvectors, 20)

    signal1  = np.mean(coeffs0_vs_t_for_epoch, axis=0)[17]
    signal2  = np.mean(coeffs1_vs_t_for_epoch, axis=0)[17]
    print(signal1.shape)
    t = range(0,len(signal1))

    # Compute wavelet coherence
    coherence, freqs = wavelet_coherence(signal1, signal2)
    vmin = 0.01
    vmax = 10.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    # Plot wavelet coherence
    plt.figure(figsize=(10, 6))
    plt.contourf(t, freqs, coherence, 161, cmap='viridis', norm=norm)
    plt.colorbar(label='Wavelet Coherence')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Wavelet Coherence')
    plt.yscale('log')
    plt.show()


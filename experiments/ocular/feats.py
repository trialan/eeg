import numpy as np
import pywt
from pyentrp import entropy as ent
from scipy.signal import coherence, hilbert
from mne.time_frequency import psd_array_welch


def band_power(data, sf, band, window_sec=None, relative=False):
    """Compute the average power of the signal in a specific frequency band."""
    band = np.asarray(band)
    low, high = band

    # Calculate PSD using Welch's method
    psds, freqs = psd_array_welch(data, sfreq=sf, fmin=low, fmax=high, n_per_seg=window_sec*sf if window_sec else None)

    # Select the frequencies of interest
    freq_res = freqs[1] - freqs[0]  # Frequency resolution
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Compute band power
    band_power = psds[:, idx_band].mean(axis=1)

    if relative:
        band_power /= psds.sum(axis=1, keepdims=True)

    return band_power


def dwt_band_power(data, wavelet='db4', level=4):
    """Compute the band power using Discrete Wavelet Transform."""
    coeffs = pywt.wavedec(data, wavelet, level=level)
    power = np.array([np.sum(c ** 2) for c in coeffs[1:]])
    return power


def dwt_coherence(data1, data2, sf, wavelet='db4', level=4):
    """Compute coherence using DWT between two signals."""
    coeffs1 = pywt.wavedec(data1, wavelet, level=level)
    coeffs2 = pywt.wavedec(data2, wavelet, level=level)
    coherence_values = []
    for c1, c2 in zip(coeffs1[1:], coeffs2[1:]):
        f, Cxy = coherence(c1, c2, sf)
        coherence_values.append(np.mean(Cxy))
    return np.array(coherence_values)


def dwt_plv(data1, data2, wavelet='db4', level=4):
    """Compute phase locking value (PLV) using DWT between two signals."""
    coeffs1 = pywt.wavedec(data1, wavelet, level=level)
    coeffs2 = pywt.wavedec(data2, wavelet, level=level)
    plv_values = []
    for c1, c2 in zip(coeffs1[1:], coeffs2[1:]):
        phase1 = np.angle(hilbert(c1))
        phase2 = np.angle(hilbert(c2))
        plv = np.abs(np.mean(np.exp(1j * (phase1 - phase2))))
        plv_values.append(plv)
    return np.array(plv_values)


def wavelet_entropy(data, wavelet='db4', level=4):
    """Compute the wavelet entropy of the signal."""
    coeffs = pywt.wavedec(data, wavelet, level=level)

    def shannon_entropy(signal):
        """Compute the Shannon entropy of a signal."""
        signal = np.abs(signal)
        signal = signal[signal > 0]  # Filter out zero values to avoid log(0)
        prob = signal / signal.sum()
        return -np.sum(prob * np.log2(prob))

    entropy = np.array([shannon_entropy(c) for c in coeffs[1:]])
    return entropy.mean()


def hjorth_parameters(data):
    """Compute Hjorth parameters: activity, mobility, and complexity."""
    first_deriv = np.diff(data)
    second_deriv = np.diff(first_deriv)
    activity = np.var(data)
    mobility = np.sqrt(np.var(first_deriv) / activity)
    complexity = np.sqrt(np.var(second_deriv) / np.var(first_deriv)) / mobility
    return activity, mobility, complexity


def fractal_dimension(data):
    """Compute the fractal dimension of the signal."""
    L = []
    x = np.array(range(1, len(data) + 1))
    y = data
    N = len(x)
    for k in range(2, int(N/2)):
        Lk = []
        for m in range(k):
            Lmk = 0
            for i in range(1, int(np.floor((N-m)/k))):
                Lmk += abs(y[m+i*k] - y[m+(i-1)*k])
            Lmk = (Lmk * (N - 1) / (np.floor((N - m) / k) * k)) / k
            Lk.append(Lmk)
        L.append(np.log(np.mean(Lk)))
    return np.polyfit(np.log(range(2, int(N/2))), L, 1)[0]



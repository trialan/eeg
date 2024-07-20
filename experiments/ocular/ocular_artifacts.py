from eeg.experiments.eigen_fgmdm import OldED
import xgboost as xgb
import lightgbm as lgb

from eeg.laplacian import compute_scalp_eigenvectors_and_values
import numpy as np
import mne
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import pywt
from mne.time_frequency import psd_array_welch
from pyentrp import entropy as ent
from scipy.signal import coherence, hilbert
from mne import Epochs, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

from eeg.data import get_raw_data, tmin, tmax, get_data
from eeg import physionet_runs
from eeg.plot_reproduction import assemble_classifer_PCACSPLDA
from eeg.utils import results, get_cv, avg_power_matrix
from eeg.experiments.ocular.custom_csp import CustomCSP

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline
from mne.decoding import UnsupervisedSpatialFilter


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.sf = 160 #sampling frequency in Hz

    def fit(self, X, y=None):
        # No fitting necessary, so we just return self
        return self

    def transform(self, X):
        return extract_features(X, self.sf)


def assemble_classifier_LaplacianCFELDA(n_components, eigenvectors):
    ed = UnsupervisedSpatialFilter(OldED(n_components, eigenvectors), average=False)
    lda = LinearDiscriminantAnalysis()
    csp = CustomCSP(n_components=n_components, reg=None, log=None, norm_trace=False)
    clf = Pipeline([("ED", ed), ("CSP", csp), ("LDA", lda)])
    return clf


def get_ocular_clean_data(subject, bandpass, runs):
    raw = get_raw_data(subject, physionet_runs)
    raw.filter(1.0, 30.0, fir_design="firwin", skip_by_annotation="edge")
    # Apply Common Average Reference (CAR) filtering
    #raw.set_eeg_reference(ref_channels='average', projection=True)
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
    event_id = ["hands", "feet"]

    #remove the ocular artifacts
    prefrontal_channels = ['Fp1', 'Fp2']
    prefrontal_indices = [raw.ch_names.index(ch) for ch in prefrontal_channels if ch in raw.ch_names]

    ica = FastICA(n_components=64, random_state=0)
    ica_sources = ica.fit_transform(raw.get_data().T).T

    features = compute_features(ica_sources, prefrontal_indices)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
    labels = kmeans.labels_

    cluster_0_mean_ccorr = np.mean(features[labels == 0], axis=0)[0]
    cluster_1_mean_ccorr = np.mean(features[labels == 1], axis=0)[0]

    # Assuming the cluster with higher mean correlation values is the ocular artifacts
    if cluster_0_mean_ccorr > cluster_1_mean_ccorr:
        ocular_components = np.where(labels == 0)[0]
    else:
        ocular_components = np.where(labels == 1)[0]


    non_ocular_components = np.setdiff1d(np.arange(64), ocular_components)

    reconstructed_eeg = ica.mixing_[:, non_ocular_components].dot(ica_sources[non_ocular_components, :])

    # Create a new Raw object with the cleaned data
    cleaned_raw = mne.io.RawArray(reconstructed_eeg, raw.info)
    cleaned_raw.set_annotations(raw.annotations)
    #split into epochs
    epochs = Epochs(
        cleaned_raw,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        proj=True,
        picks=picks,
        baseline=None,
        preload=True,
    )
    X = epochs.copy().get_data(copy=False)
    y = epochs.events[:, -1] - 2
    return X, y


def compute_features(ica_sources, fp_indices):
    """Compute features for each IC for Kmeans clustering."""
    features = []
    for i, source in enumerate(ica_sources):
        # Cross-correlation with prefrontal electrodes
        cross_corr = np.mean([np.correlate(source, ica_sources[j], mode='valid') for j in fp_indices])
        # Distribution ratio
        dist_ratio = np.max(source) / np.std(source)
        # Maximum value
        max_value = np.max(source)
        features.append([cross_corr, dist_ratio, max_value])
    return np.array(features)


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


def extract_features(X, sf):
    """Extract features from the EEG data."""
    N, C, T = X.shape
    band = (8, 30)  # Mu and Beta rhythms
    window_sec = 2
    features = np.zeros((N, 12))
    for i in range(N):
        # Band Power
        bp = band_power(X[i], sf, band, window_sec)
        features[i, 0] = bp.mean()

        # DWT-Band Power
        dwt_bp = dwt_band_power(X[i, 8])  # C3
        features[i, 1] = dwt_bp.mean()

        # DWT-Coherence
        dwt_coh = dwt_coherence(X[i, 8], X[i, 12], sf)  # C3 and C4
        features[i, 2] = dwt_coh.mean()

        dwt_plv_val = dwt_plv(X[i, 8], X[i, 23])  # C3 and Fz
        features[i, 3] = dwt_plv_val.mean()

        # 5. Wavelet Entropy
        we = wavelet_entropy(X[i, 8])  # C3
        features[i, 4] = we

        # 6. Hjorth Parameters
        activity, mobility, complexity = hjorth_parameters(X[i, 8])  # C3
        features[i, 5] = activity
        features[i, 6] = mobility
        features[i, 7] = complexity

        # 7. Fractal Dimension
        fd = fractal_dimension(X[i, 8])  # C3
        features[i, 8] = fd

        # 8. PSD in Theta Band (4-7 Hz)
        theta_bp = band_power(X[i], sf, (4, 7), window_sec)
        features[i, 9] = theta_bp.mean()

        # 9. PAC (Phase-Amplitude Coupling)
        phase_amp_coupling = np.abs(np.mean(np.exp(1j * (np.angle(hilbert(X[i, 8]))) * np.abs(hilbert(X[i, 8])))))  # C3
        features[i, 10] = phase_amp_coupling

        # 10. Signal-to-Noise Ratio (SNR) in Mu and Beta Band
        snr = np.mean(bp) / np.mean(X[i, 8])
        features[i, 11] = snr

    return features


if __name__ == '__main__':

    from eeg.plot_reproduction import assemble_classifier_LaplacianCSPLDA

    X, y = get_data(2)
    cv = get_cv()
    eigenvectors, eigenvals = compute_scalp_eigenvectors_and_values()

    #LaplacianCFELDA: 62.01%
    #LaplacianCSPLDA: 62.39%
    #LaplacianCFELGB: 61.03% <-- LDA is good.

    """
    It's not really surprising that LaplacianCFELDA doesn't work
    because the feats are on the channels. But with Laplacian i've
    changed the basis --> so this doesn't really make sense anymore.
    Perhaps what I need to do is:
        - find the laplacian components most correlated to 
    """

    sf = 160  # Sampling frequency
    #features = extract_features(X, sf)
    #print("Extracted features shape:", features.shape)

    #LDA with eyes removed + feats : 56.39% --> it's the feats that help
    #this is probably bc they already removed this from the ds
    #Vanilla LDA + feats           : 56.50%
    #Vanilla LDA + avg_power_matrix: 48.85% <VERY INTERESTING!>
    #XGB + feats                   : 53.60%  <-- should be 60%+?
    #XGB + avg_power_matrix        : 53.48%
    #LGB + feats                   : 56.67%  <-- it's barely better than LDA
    #LGB + avg_power_matrix        : 48.85%  <-- more along what i expected
    #no we try removing features
    #LGB + feats, no dwt_plv_val   : 55.77%
    #LGB + band power + dwt bandpow: 55.68%
    #LGB + band power              : 48.68%  <-- DWT-BP is key feature,
    #                                            7% improvement from this.
    # --> this suggests keeping LGB, adding 10 new good feats.
    #LGB + 12 feats                : 56.31%  <-- Nope, ok too bad.
    #We can just remember that these feats do help LDA a lot.
    #The shame is that the main good idea from this paper is removing the
    #ocular artifacts, and that doesn't seem to actually work.
    #Laplcian + feats + LDA        : 52.22%



    """
    score = results(clf, X - np.mean(X), y, cv)
    print(score) #62.24%

    score = results(clf, (X - np.mean(X))/np.std(X), y, cv)
    print(score) #62.47% (OK whitening is a ~1SE improvement I think,
    # better than 62.33% without whitening.
    # --> this is a #TODO for later, un-important for current analysis,
    #but should provide a small performance increase in meta_clf.py as it
    # --> It did not! How odd.
    """




    """
    # Assuming `eeg_data` is your (64, 161) EEG signal
    # Bandpass filter the data
    raw = mne.io.RawArray(eeg_data, mne.create_info(ch_names=64, sfreq=160, ch_types="eeg"))
    raw.filter(1, 30, fir_design='firwin')

    # Apply CAR filtering
    eeg_data_filtered = raw.get_data() - np.mean(raw.get_data(), axis=0)
    """

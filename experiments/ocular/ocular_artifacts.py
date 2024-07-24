import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
import numpy as np
from tqdm import tqdm

import mne
from mne import Epochs, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.decoding import CSP, UnsupervisedSpatialFilter

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from eeg.plot_reproduction import assemble_classifer_PCACSPLDA
from eeg.utils import results, get_cv
from eeg.experiments.ocular.custom_csp import CustomCSP, FeatureExtractor
from eeg.experiments.eigen_fgmdm import OldED
from eeg.experiments.ocular.feats import *

from eeg import physionet_runs
from eeg.data import get_data, get_raw_data
from eeg.experiments.eigen_fgmdm import EDFgMDM
from eeg.laplacian import compute_scalp_eigenvectors_and_values
from eeg.utils import results, get_cv, get_covariances, get_fraction
from eeg.plot_reproduction import (
    assemble_classifer_CSPLDA,
)
from eeg.experiments.channel_selection.channels import get_sorted_channels


def assemble_classifier_FELDA(extractor=FeatureExtractor):
    fe = FeatureExtractor(extract_features)
    lda = LinearDiscriminantAnalysis()
    clf = Pipeline([("FE", fe), ("LDA", lda)])
    return clf


def get_channel_indices(channels):
    raw = get_raw_data(1, physionet_runs)
    raw.filter(1.0, 30.0, fir_design="firwin", skip_by_annotation="edge")
    # Apply Common Average Reference (CAR) filtering
    #raw.set_eeg_reference(ref_channels='average', projection=True)
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

    #remove the ocular artifacts
    indices = [raw.ch_names.index(ch) for ch in channels if ch in raw.ch_names]
    return indices


def find_most_correlated_indices(X_original, X_eigenbasis, target_indices):
    N, C, T = X_original.shape
    correlated_indices = []
    for target_idx in target_indices:
        target_channel = X_original[:, target_idx, :]
        correlations = []
        for i in range(C):
            eigen_component = X_eigenbasis[:, i, :]
            correlation = np.mean([np.corrcoef(target_channel[j], eigen_component[j])[0, 1] for j in range(N)])
            correlations.append(correlation)
        print(f"Max correlation: {max(correlations)}")
        most_correlated_idx = np.argmax(np.abs(correlations))
        correlated_indices.append(most_correlated_idx)
    return correlated_indices


def extract_features(X, sf=160):
    """Extract features from the EEG data."""
    N, C, T = X.shape

    pca = UnsupervisedSpatialFilter(PCA(64), average=False)
    X_pca = pca.fit_transform(X)

    ica = UnsupervisedSpatialFilter(FastICA(64), average=False)
    X_ica = pca.fit_transform(X)

    band = (8, 30)  # Mu and Beta rhythms
    window_sec = 2
    features = np.zeros((N, 14+3*C))

    target_indices = [8, 12, 23] #C3, C4, Fz
    #correlated_indices = find_most_correlated_indices(X, X_eigenbasis, target_indices)
    c3_index = target_indices[0]
    c4_index = target_indices[1]
    fz_index = target_indices[2]
    cz_index = 10

    for i in range(N):
        # Band Power
        bp = band_power(X[i], sf, band, window_sec)
        features[i, 0] = bp.mean()

        # DWT-Band Power
        dwt_bp = dwt_band_power(X[i, c3_index])  # C3
        features[i, 1] = dwt_bp.mean()

        dwt_bp = dwt_band_power(X[i, c4_index])
        features[i, 2] = dwt_bp.mean()

        dwt_bp = dwt_band_power(X[i, cz_index])
        features[i, 3] = dwt_bp.mean()

        # DWT-Coherence
        dwt_coh = dwt_coherence(X[i, c3_index], X[i, c4_index], sf)  # C3 and C4
        features[i, 4] = dwt_coh.mean()

        dwt_plv_val = dwt_plv(X[i, c3_index], X[i, fz_index])  # C3 and Fz
        features[i, 5] = dwt_plv_val.mean()

        # 5. Wavelet Entropy
        we = wavelet_entropy(X[i, c3_index])  # C3
        features[i, 6] = we

        # 6. Hjorth Parameters
        activity, mobility, complexity = hjorth_parameters(X[i, c3_index])  # C3
        features[i, 7] = activity
        features[i, 8] = mobility
        features[i, 9] = complexity

        # 7. Fractal Dimension
        fd = fractal_dimension(X[i, c3_index])  # C3
        features[i, 10] = fd

        # 8. PSD in Theta Band (4-7 Hz)
        theta_bp = band_power(X[i], sf, (4, 7), window_sec)
        features[i, 11] = theta_bp.mean()

        # 9. PAC (Phase-Amplitude Coupling)
        phase_amp_coupling = np.abs(np.mean(np.exp(1j * (np.angle(hilbert(X[i, c3_index]))) * np.abs(hilbert(X[i, c3_index])))))  # C3
        features[i, 12] = phase_amp_coupling

        # 10. Signal-to-Noise Ratio (SNR) in Mu and Beta Band
        snr = np.mean(bp) / np.mean(X[i, c3_index])
        features[i, 13] = snr

        # 11. Avg log power (natural basis)
        avg_power = (X[i] ** 2).mean(axis=1)
        avg_power = np.log(avg_power)
        features[i,14:14+C] = avg_power

        # 12. Avg log power (PCA basis)
        avg_power = (X_pca[i] ** 2).mean(axis=1)
        avg_power = np.log(avg_power)
        features[i,14+C:14+2*C] = avg_power

        # 13. Avg log power (ICA basis)
        avg_power = (X_ica[i] ** 2).mean(axis=1)
        avg_power = np.log(avg_power)
        features[i,14+2*C:] = avg_power

    return features


if __name__ == "__main__":
    X, y = get_data()
    cv = get_cv()
    eigenvectors, eigenvals = compute_scalp_eigenvectors_and_values()
    lda = LinearDiscriminantAnalysis()
    feats = extract_features(X)
    score = results(lda, feats, y, cv)
    print(score)



import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

import mne
from mne import Epochs, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf


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
from eeg.laplacian import compute_scalp_eigenvectors_and_values
from eeg.experiments.ocular.feats import *

from tqdm import tqdm
from venn import venn
import matplotlib.pyplot as plt
import numpy as np
import pyriemann
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from eeg.data import get_data
from eeg.experiments.eigen_fgmdm import EDFgMDM
from eeg.laplacian import compute_scalp_eigenvectors_and_values
from eeg.utils import results, get_cv, get_covariances, get_fraction
from eeg.plot_reproduction import (
    assemble_classifer_CSPLDA,
)
from eeg.experiments.channel_selection.channels import get_sorted_channels
from eeg.experiments.routing_models.meta_clf import generate_datasets


def assemble_classifier_FELDA(extractor=FeatureExtractor):
    fe = FeatureExtractor(extract_features)
    lda = lgb.LGBMClassifier()#LinearDiscriminantAnalysis()
    clf = Pipeline([("FE", fe), ("LDA", lda)])
    return clf


def extract_features(X, sf=160):
    """Extract features from the EEG data."""
    N, C, T = X.shape
    band = (8, 30)  # Mu and Beta rhythms
    window_sec = 2
    features = np.zeros((N, 12+C))
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

        # Log average power of each channel
        avg_power = (X[i] ** 2).mean(axis=1)
        #avg_power = np.log(avg_power)
        features[i, 12:] = avg_power

    return features


if __name__ == "__main__":
    X, y = get_data()
    cv = get_cv()
    eigenvectors, eigenvals = compute_scalp_eigenvectors_and_values()
    datasets = generate_datasets(X, y, cv)
    sorted_channels = get_sorted_channels(X, y, cv)

    meta_clf_scores = []
    edf_scores = []
    scf_scores = []
    f_scores = []
    pcl_scores = []
    cl_scores = []
    router_scores = []

    for i, (X_train, y_train, X_test, y_test) in tqdm(enumerate(datasets)):

        # Train a EDFgMDM classifier (laplacian + fgmdm)
        print("Training EDFgMDM")
        edf = EDFgMDM(n_components=24, eigenvectors=eigenvectors)
        edf.fit(X_train, y_train)
        edf_y_pred = edf.predict(X_test)
        edf_score = accuracy_score(edf_y_pred, y_test)
        edf_scores.append(edf_score)

        # Train a 20-channel FgMDM classifier
        print("20C-FgMDM")
        channel_X_train = X_train[:, sorted_channels[:20], :]
        channel_X_train_cov = get_covariances(channel_X_train)

        channel_X_test = X_test[:, sorted_channels[:20], :]
        channel_X_test_cov = get_covariances(channel_X_test)

        scf = pyriemann.classification.FgMDM()
        scf.fit(channel_X_train_cov, y_train)
        scf_y_pred = scf.predict(channel_X_test_cov)
        scf_score = accuracy_score(scf_y_pred, y_test)
        scf_scores.append(scf_score)

        # Train FgMDM
        print("Training FgMDM")
        Xcov = get_covariances(X_train)
        Xcov_test = get_covariances(X_test)
        f = pyriemann.classification.FgMDM()
        f.fit(Xcov, y_train)
        f_y_pred = f.predict(Xcov_test)
        f_score = accuracy_score(f_y_pred, y_test)
        f_scores.append(f_score)

        # Train PCA+CSP+LDA
        print("Training PCL")
        pcl = assemble_classifer_PCACSPLDA(n_components=42)
        pcl.fit(X_train, y_train)
        pcl_y_pred = pcl.predict(X_test)
        pcl_score = accuracy_score(pcl_y_pred, y_test)
        pcl_scores.append(pcl_score)

        # Train CSP+LDA
        print("Training CL")
        cl = assemble_classifer_CSPLDA(n_components=10)
        cl.fit(X_train, y_train)
        cl_y_pred = cl.predict(X_test)
        cl_score = accuracy_score(cl_y_pred, y_test)
        cl_scores.append(cl_score)


        models_test_preds = np.array([cl_y_pred,
                                 pcl_y_pred,
                                 f_y_pred,
                                 scf_y_pred,
                                 edf_y_pred])

        models_train_preds = np.array([
            cl.predict(X_train),
            pcl.predict(X_train),
            f.predict(Xcov),
            scf.predict(channel_X_train_cov),
            edf.predict(X_train)])

        scaler = StandardScaler()
        test_og_feats = extract_features(X_test)
        test_combined_feats = np.hstack((test_og_feats,
                                         models_test_preds.T))

        train_og_feats = extract_features(X_train)
        train_combined_feats = np.hstack((train_og_feats,
                                          models_train_preds.T))

        train_combined_feats_scaled = scaler.fit_transform(train_combined_feats)
        test_combined_feats_scaled = scaler.transform(test_combined_feats)

        meta_clf = LogisticRegression(max_iter=200)
        meta_clf.fit(train_combined_feats_scaled, y_train)
        y_pred = meta_clf.predict(test_combined_feats_scaled)

        score = accuracy_score(y_pred, y_test)
        meta_clf_scores.append(score)


    print(f"Mean Meta-clf score: {np.mean(meta_clf_scores)}")
    print(f"Mean Laplacian + FgMDM (n=24) model score: {np.mean(edf_scores)}")
    print(f"Mean 24-channel FgMDM score: {np.mean(scf_scores)}")
    print(f"Mean FgMDM score: {np.mean(f_scores)}")
    print(f"Mean PCL score: {np.mean(pcl_scores)}")
    print(f"Mean CL score: {np.mean(cl_scores)}")

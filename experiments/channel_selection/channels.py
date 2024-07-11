import numpy as np
from tqdm import tqdm
import pyriemann

import matplotlib.pyplot as plt
from venn import venn

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F

from eeg.data import get_data
from eeg.utils import (results, get_cv,
                       avg_power_vector,
                       get_covariances,
                       set_seed)
from eeg.experiments.eigen_fgmdm import EDFgMDM, OldED
from eeg.laplacian import compute_scalp_eigenvectors_and_values
from eeg.ml import (assemble_classifer_PCACSPLDA,
                    assemble_classifer_CSPLDA,
                    PCA)


"""
The max score of FgMDM sorted by N best channels is 0.637945 with 24 channels. At least this was the case when I first ran this, I can't seem to get
the same channel ordering. Old channel ordering is saved in the experiment
write-upds. This current approach also improves on FgMDM. I think it would
be interesting to try to combine this with Laplacian and get a higher score.
"""

n_channels = 64

def get_sorted_channels(X, y, cv):
    #A bit sketch to do this on train+test, but we'll fix that later
    channels = np.array(range(0, n_channels))
    channel_scores = []
    for channel in channels:
        X_channel = X[:, channel, :]
        X_ap = [avg_power_vector(v) for v in X_channel]
        X_ap = (np.array(X_ap) - np.std(X_ap)) / np.mean(X_ap)
        X_ap = np.array(X_ap).reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(X_ap, y)
        scores = cross_val_score(lr, X_ap, y, cv=cv, n_jobs=None)
        channel_scores.append(np.mean(scores))

    sorted_indices = np.argsort(channel_scores)[::-1]
    sorted_channels = channels[sorted_indices]
    return sorted_channels



if __name__ == '__main__':
    X, y = get_data()
    cv = get_cv()
    eigenvectors, eigenvals = compute_scalp_eigenvectors_and_values()
    sorted_channels = get_sorted_channels(X, y, cv)

    1/0

    """
    scores = []
    for i in tqdm(range(1, n_channels)):
        sub_X = X[:, sorted_channels[:20], :]
        sub_Xcov = get_covariances(sub_X)
        fgmdm = pyriemann.classification.FgMDM()
        score = results(fgmdm, sub_Xcov, y, cv)
        scores.append(score)

    plt.plot(list(range(1, n_channels)), scores,
             marker='o', linestyle='-', label='Best N channels FgMDM')
    plt.axhline(y=0.6199, linestyle='--', label="Traditional FgMDM",
                color='b')
    plt.axhline(y=0.6424, linestyle='--', label="Xu et al. top score",
                color='r')
    plt.axhline(y=0.6606, linestyle='--', label="Our top score",
                color='g')
    plt.title("FgMDM performance with best N channels")
    plt.xlabel("Best N channels (determined by linear reg.)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    """

def transform_to_eigenbasis(matrix, eigenvectors):
    """ eigenvectors are rows """
    assert matrix.shape[0] == eigenvectors.shape[0]
    assert eigenvectors.shape[0] == eigenvectors.shape[1]
    transformed_matrix = np.dot(eigenvectors, matrix) #? not .T
    return transformed_matrix

    scores = []
    for n_components in tqdm(list(range(10, 35))):
        channel_X = X[:, sorted_channels[:n_components], :]
        n_epochs, n_channels, n_times = channel_X.shape
        X_ed = np.array([transform_to_eigenbasis(epoch, eigenvectors[:n_components, :n_components]) for epoch in channel_X])
        Xcov = pyriemann.estimation.Covariances('oas').fit_transform(X_ed)

        FgMDM = pyriemann.classification.FgMDM()
        score = results(FgMDM, Xcov, y, cv)
        scores.append(score)
    plt.plot(list(range(10,35)), scores, marker='o', linestyle='-', label='Laplacian+FgMDM')
    plt.legend()
    plt.show()



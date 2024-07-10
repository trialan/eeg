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
from eeg.experiments.ensemble import EDFgMDM, OldED
from eeg.laplacian import compute_scalp_eigenvectors_and_values
from eeg.ml import (assemble_classifer_PCACSPLDA,
                    assemble_classifer_CSPLDA,
                    PCA)


"""
The max score of FgMDM sorted by N best channels is 0.637945 with 24 channels
"""


if __name__ == '__main__':
    X, y = get_data()
    cv = get_cv()
    eigenvectors, eigenvals = compute_scalp_eigenvectors_and_values()

    n_channels = X.shape[1]
    channels = np.array(range(0, n_channels))

    channel_scores = []
    for channel in channels:
        X_channel = X[:, channel, :]
        X_ap = [avg_power_vector(v) for v in X_channel]
        X_ap = (np.array(X_ap) - np.std(X_ap)) / np.mean(X_ap)
        X_ap = X_ap.reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(X_ap, y)
        scores = cross_val_score(lr, X_ap, y, cv=cv, n_jobs=None)
        channel_scores.append(score)

    sorted_indices = np.argsort(channel_scores)[::-1]
    sorted_channels = channels[sorted_indices]

    scores = []
    for i in tqdm(range(1, n_channels)):
        sub_X = X[:, sorted_channels[:i], :]
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





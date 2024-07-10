
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
Following up on our `channels.py` results, we train a LSP+FgMDM model
with 24 scalp eigenmodes and the 24 best channels only.
"""



if __name__ == '__main__':
    X, y = get_data()
    cv = get_cv()
    eigenvectors, eigenvals = compute_scalp_eigenvectors_and_values()

    sorted_channels = np.array([63, 62, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1, 30, 31, 32, 48, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 47, 33, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34,  0])

    edf = EDFgMDM(n_components=24, eigenvectors=eigenvectors)
    fgmdm = pyriemann.classification.FgMDM()
    sub_X = X[:, sorted_channels[:24], :]
    sub_Xcov = get_covariances(sub_X)
    edf_score = results(edf, sub_Xcov, y, cv)
    vanilla_score = results(fgmdm, sub_Xcov, y, cv)
    print(f"EDF score: {edf_score}")
    print(f"vanilla score: {vanilla_score}")



from mne.decoding import CSP
import numpy as np
from tqdm import tqdm
import pyriemann
from scipy.stats import boxcox

import matplotlib.pyplot as plt
from venn import venn

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from eeg.data import get_data
from eeg.utils import (results, get_cv,
                       avg_power_matrix,
                       get_covariances,
                       set_seed)
from eeg.experiments.eigen_fgmdm import EDFgMDM, OldED
from eeg.laplacian import compute_scalp_eigenvectors_and_values
from eeg.ml import (assemble_classifer_PCACSPLDA,
                    assemble_classifer_CSPLDA,
                    PCA)
from sklearn.base import BaseEstimator, TransformerMixin


"""
Top score for:
    - LSP+LDA with avg. power: 58.4%
    - LSP+LDA with log-var   : 58.5%

Motivation for this experiment: https://doi.org/10.1016/j.bspc.2021.103101
claim to get 90% accuracy on motor-imagery classification on the Berlin BCI
competition. afaik the only difference between what they do and what we do
is the log-variance way of dropping the time dimension. I am very suspicious
of their claims.

    "Brain signals were projected on the CSP plane, yielding 6 channels. And
    only three of those CSP filters were selected according to the
    neurophysiological plausibility, mainly in the primary motor cortex (C3, Cz,
    C5). Final feature vectors were represented by the logarithm of the
    variance of each CSP channel" -- page 4

Perhaps this CSP + log-variance is a good idea? They claim over 70% accuracy
using SVM with this data. I try this below, get far from this. Not going to
dig into this further, seems like nonsense paper.
"""


def logvar_matrix(m):
    """ Drop the time dimension on matrix by log-variance """
    assert len(m.shape) == 2
    y = np.array([np.log(np.var(row)) for row in m])
    return y


class LogVarTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X is expected to be of shape (n_epochs, p, q)
        # Calculate the log of the variance along the last dimension (q)
        log_var = np.log(np.var(X, axis=2))
        return log_var


def assemble_classifer_CSPLDA(n_components, target_space):
    assert target_space in ["average_power", "csp_space"]
    svc = SVC()
    csp = CSP(n_components=n_components, transform_into=target_space,
              reg=None, log=None)
    if target_space == "average_power":
        clf = Pipeline([("CSP", csp), ("LDA", svc)])
    else:
        print("Using log-var transform")
        lvt = LogVarTransformer()
        clf = Pipeline([("CSP", csp), ("LVT", lvt), ("LDA", svc)])
    return clf


if __name__ == '__main__':
    X, y = get_data()
    cv = get_cv()
    eigenvectors, eigenvals = compute_scalp_eigenvectors_and_values()

    component_numbers = list(range(2, 25))

    clf = assemble_classifer_CSPLDA(15, target_space="average_power")
    score = results(clf, X, y, cv)
    print(score)

    clf = assemble_classifer_CSPLDA(15, target_space="csp_space")
    score = results(clf, X, y, cv)
    print(score)


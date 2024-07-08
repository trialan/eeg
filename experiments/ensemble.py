import numpy as np
import pyriemann

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.metrics import accuracy_score

from eeg.data import get_data
from eeg.utils import results, get_cv
from eeg.ml import (assemble_classifer_PCACSPLDA,
                    assemble_classifer_CSPLDA,
                    PCA)
from eeg.laplacian import compute_scalp_eigenvectors_and_values


"""
What we would like to beat: 63.38%.

Experiment 1: score = 62.81%
    - EnsembleClassifier with:
        - PCA + CSP + LDA with 30 components
        - CSP + LDA with 10 components
        - OldED + FgMDM with 24 components
    - weights: 1/3 for each model

Experiment 2: score = 62.73%
    - EnsembleClassifier with:
        - PCA + CSP + LDA with 30 components
        - CSP + LDA with 10 components
        - OldED + FgMDM with 24 components
        - weights = np.array([0.6338, 0.6252, 0.6174]) / sum_of_scores)
          where sum_of_scores = 0.6338 + 0.6252 + 0.6174, i.e. we have
          weighted the models' contributions according to their scores.

--> it seems clear from the two experiments above that simple ensembling
    will not do. We need to have a router model that routes to the best
    model. But perhpas we should first check if the hypothesis "model A
    is good at detecting subject S1, but model B does better on subject
    S2 even though A is overall the better model"
"""


class OldED(BaseEstimator, TransformerMixin):
    """ Old version of our Eigen-Decomp """
    def __init__(self, n_components, eigenvectors):
        self.n_components = n_components
        self.eigenvectors = eigenvectors

    def fit(self, X, y=None):
        if self.eigenvectors is None:
            raise ValueError("Eigenvectors are not set.")
        return self

    def transform(self, X):
        n_channels, n_times = X.shape
        selected_eigenvectors = self.eigenvectors[:, :self.n_components]
        X_transformed = np.dot(X, selected_eigenvectors)
        return X_transformed


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """ t_classifiers are 'traditional', r_classifiers need cov. matrix """
    def __init__(self, t_classifiers, r_classifiers, weights=None):
        self.t_classifiers = t_classifiers
        self.r_classifiers = r_classifiers
        self.classifiers = []
        self.classifiers.extend(t_classifiers)
        self.classifiers.extend(r_classifiers)
        N = len(t_classifiers) + len(r_classifiers)
        self.weights = weights if weights is not None else np.ones(N) / N

    def fit(self, X, y):
        for clf in self.t_classifiers:
            clf.fit(X, y)
        for clf in self.r_classifiers:
            clf.fit(X, y)
        return self

    def predict(self, X):
        return np.array([round(b) for (a,b) in self.predict_proba(X)])

    def predict_proba(self, X):
        """ returns [(a_0,b_0), ...] where a_i=prob(0), b_i=prob(1) """
        final_prediction = np.zeros((X.shape[0], 2))
        for clf, weight in zip(self.classifiers, self.weights):
            predictions = clf.predict_proba(X)
            final_prediction += weight * predictions
        return final_prediction


class EDFgMDM(BaseEstimator):
    def __init__(self, n_components):
        self.n_components = n_components
        self.clf = FgMDM = pyriemann.classification.FgMDM()

    def get_ED_covariance_data(self, X):
        n_epochs, n_channels, n_times = X.shape
        X_reshaped = X.reshape(n_times * n_epochs, n_channels)
        ed = OldED(self.n_components, eigenvectors)
        X_ed = np.array([ed.transform(epoch.T).T for epoch in X])
        Xcov = pyriemann.estimation.Covariances('oas').fit_transform(X_ed)
        return Xcov

    def fit(self, X, y):
        Xcov = self.get_ED_covariance_data(X)
        self.clf.fit(Xcov, y)

    def predict_proba(self, X):
        Xcov = self.get_ED_covariance_data(X)
        return self.clf.predict_proba(Xcov)

    def predict(self, X):
        Xcov = self.get_ED_covariance_data(X)
        return self.clf.predict(Xcov)

    def score(self, X, y):
        y_pred = self.predict(X)
        score = accuracy_score(y_pred, y)
        return score


if __name__ == '__main__':
    X, y = get_data()
    cv = get_cv()
    eigenvectors, eigenvals = compute_scalp_eigenvectors_and_values()

    print("Laplacian+FgMDM with 24 components (score: 63.38%)")
    ed_fgmdm = EDFgMDM(n_components=24)
    #score = results(clf, X, y, cv)


    print("PCA+CSP+LDA with 30 components (score: 62.52%)")
    n_components = 30
    pca_csp_lda = assemble_classifer_PCACSPLDA(n_components)
    #score = results(pca_csp_lda, X, y, cv)


    print("CSP+LDA with 10 components (score: 61.74%)")
    n_components = 10
    csp_lda = assemble_classifer_CSPLDA(n_components)
    #score = results(csp_lda, X, y, cv)

    sum_of_scores = 0.6338 + 0.6252 + 0.6174
    ensemble = EnsembleClassifier(t_classifiers = [pca_csp_lda, csp_lda],
                                  r_classifiers = [ed_fgmdm],
                                  weights = np.array([0.6338, 0.6252, 0.6174]) / sum_of_scores)
    score = results(ensemble, X, y, cv)
    print(score)




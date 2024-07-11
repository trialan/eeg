import numpy as np
import pyriemann
from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.metrics import accuracy_score


class EDFgMDM(BaseEstimator):
    def __init__(self, n_components, eigenvectors):
        self.n_components = n_components
        self.eigenvectors = eigenvectors
        self.clf = FgMDM = pyriemann.classification.FgMDM()

    def get_ED_covariance_data(self, X):
        n_epochs, n_channels, n_times = X.shape
        X_reshaped = X.reshape(n_times * n_epochs, n_channels)
        ed = OldED(self.n_components, self.eigenvectors)
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


class OldED(BaseEstimator, TransformerMixin):
    """ Old version of our Eigen-Decomp """
    def __init__(self, n_components, eigenvectors, hack=False):
        self.n_components = n_components
        self.eigenvectors = eigenvectors
        self.hack = hack

    def fit(self, X, y=None):
        if self.eigenvectors is None:
            raise ValueError("Eigenvectors are not set.")
        return self

    def transform(self, X):
        n_channels, n_times = X.shape
        selected_eigenvectors = self.eigenvectors[:, :self.n_components]
        # I do not understand why this is necessary :-(
        if self.hack:
            X_transformed = np.dot(selected_eigenvectors, X)
        else:
            X_transformed = np.dot(X, selected_eigenvectors)
        return X_transformed



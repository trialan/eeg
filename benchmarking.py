import moabb
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from moabb.datasets import PhysionetMI
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import LeftRightImagery, MotorImagery
from pyriemann.spatialfilters import CSP
from pyriemann.estimation import Covariances
from pyriemann.classification import FgMDM
from sklearn.pipeline import make_pipeline

from eeg.utils import get_cv, avg_power_vector


moabb.set_log_level("error")
pipelines = {}
n_channels = 64

from eeg.experiments.eigen_fgmdm import OldED
from eeg.laplacian import compute_scalp_eigenvectors_and_values


class EigenDecomp(BaseEstimator, TransformerMixin):
    def __init__(self, eigenvecs, n_components):
        self.eigenvecs = eigenvecs
        self.n_components = n_components

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        ed = OldED(self.n_components, self.eigenvecs)
        X_ed = np.array([ed.transform(epoch.T).T for epoch in X])
        return X_ed


if __name__ == '__main__':
    dataset = PhysionetMI()
    paradigm = LeftRightImagery()
    evecs, _ = compute_scalp_eigenvectors_and_values()
    datasets = [dataset]
    overwrite = True
    evaluation = WithinSessionEvaluation(
        paradigm=paradigm, datasets=datasets, suffix="examples", overwrite=overwrite
    )

    scores = []
    std_errs = []
    for n_evecs in range(3,64):
        pipelines["Eigen-FgMDM"] = make_pipeline(EigenDecomp(evecs, n_evecs), Covariances("oas"), FgMDM())

        results = evaluation.process(pipelines)
        score = results.groupby("pipeline").score.mean()
        std_err = results.groupby("pipeline").score.sem()

        print(n_evecs)
        scores.append(score.iloc[0])
        print(score.iloc[0])
        std_errs.append(std_err.iloc[0])
        print(std_err.iloc[0])
        print("##########")

    x = list(range(3, 64))
    plt.plot(x, scores)
    plt.savefig("eigen_fgmdm_range.png")
    print(max(scores))
    print(np.argmax(scores))
    print(x[np.argmax(scores)])

    """
    Result: 0.720683 (0.017255) -- this is SoTA on Aug. 12th, 2024.
    Benchmark: 0.6887513 (0.0179015) -- this is FgMDM
    """


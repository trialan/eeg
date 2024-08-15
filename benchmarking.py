import numpy as np
import moabb
from moabb.datasets import PhysionetMI, Shin2017A, AlexMI
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import LeftRightImagery, MotorImagery
from pyriemann.estimation import Covariances
from pyriemann.classification import FgMDM
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from eeg.utils import get_cv, avg_power_vector

pipelines = {}

from eeg.experiments.eigen_fgmdm import OldED
from eeg.laplacian import compute_scalp_eigenvectors_and_values

from moabb.utils import set_download_dir
scratch_dir = "/anvil/scratch/x-trialan"
set_download_dir(scratch_dir)


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

    #pipelines["FgMDM"] = make_pipeline(EigenDecomp(evecs, 20), Covariances("oas"), FgMDM())
    pipelines["pipe"] = make_pipeline(TangentSpace(metric="riemann"), LogisticRegression())

    results = evaluation.process(pipelines)
    score = results.groupby("pipeline").score.mean()
    std_err = results.groupby("pipeline").score.sem()

    print(score.iloc[0])
    print(std_err.iloc[0])
    print("##########")

    """
    PhysionetMI
    Result: 0.720683 (0.017255) -- this is SoTA on Aug. 12th, 2024.
    Benchmark: 0.6887513 (0.0179015) -- this is FgMDM
    """



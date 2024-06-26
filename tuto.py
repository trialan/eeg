import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline

import pyriemann

from mne import Epochs, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.decoding import CSP, UnsupervisedSpatialFilter
from mne.io import concatenate_raws, read_raw_edf

"""
First we define our data-related parameters
    - we set tmin and tmax as in the tutorial linked below
    - runs are [6,10,14] as in the tutorial, but based on physionet page,
      it would make sense to include [5,9,13] as well.
    - use a single subject (subject 1) for pipeline setup

Physionet page: https://physionet.org/content/eegmmidb/1.0.0/
Tutorial: https://mne.tools/dev/auto_examples/decoding/decoding_csp_eeg.html
"""

tmin, tmax = 1.0, 2.0
subject = 1
runs = [6, 10, 14]


def get_raw_data():
    """ Get raw EEGMI data from website or locally if already downloaded """
    raw_fnames = eegbci.load_data(subject, runs)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    eegbci.standardize(raw)  # set channel names
    montage = make_standard_montage("standard_1005")
    raw.set_montage(montage)
    raw.annotations.rename(dict(T1="hands", T2="feet"))
    raw.set_eeg_reference(projection=True)
    return raw


def get_formatted_data():
    """ Band pass filter + pick only EEG channels + format as Epoch objects """
    raw = get_raw_data()
    raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
    epochs = Epochs(
        raw,
        event_id=["hands", "feet"],
        tmin=tmin,
        tmax=tmax,
        proj=True,
        picks=picks,
        baseline=None,
        preload=True,
    )
    X = epochs.copy().get_data(copy=False)
    y = epochs.events[:, -1] - 2
    return X, y


def assemble_classifer(n_components):
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    pca = UnsupervisedSpatialFilter(PCA(n_components), average=False)
    clf = Pipeline([("PCA", pca), ("CSP", csp), ("LDA", lda)])
    return clf


def print_results(clf, X, y, cv):
    scores = cross_val_score(clf, X, y, cv=cv, n_jobs=None)
    class_balance = np.mean(y == y[0])
    class_balance = max(class_balance, 1.0 - class_balance)
    print(f"Accuracy: {np.mean(scores)} / Chance level: {class_balance}")


if __name__ == '__main__':
    X, y = get_formatted_data()
    cv = ShuffleSplit(5, test_size=0.2, random_state=42)

    #'oas' because: https://github.com/pyRiemann/pyRiemann/issues/65
    X = pyriemann.estimation.Covariances('oas').fit_transform(X)
    clf = pyriemann.classification.FgMDM()

    #clf = assemble_classifer(n_components=5)
    print_results(clf, X, y, cv)



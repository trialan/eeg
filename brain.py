import matplotlib.pyplot as plt
import pickle
import numpy as np
from tqdm import tqdm
from mne.decoding import CSP, UnsupervisedSpatialFilter

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

import pyriemann

from eeg.data import get_data
from eeg.laplacian import compute_scalp_eigenvectors_and_values
from eeg.utils import results, get_cv, avg_power_matrix
from eeg.experiments.eigen_fgmdm import OldED
from eeg.plot_reproduction import assemble_classifer_CSPLDA
from tqdm import tqdm


def load_np_array_from_pkl(file_path):
    with open(file_path, 'rb') as file:
        np_array = pickle.load(file)
    return np_array


if __name__ == '__main__':
    _, y = get_data()
    cv = get_cv()

    S_raw = load_np_array_from_pkl('array_data.pkl')
    S = np.array([avg_power_matrix(m) for m in S_raw])

    n_components = 10
    clf = assemble_classifer_CSPLDA(n_components)
    score = results(clf, S, y, cv)

    """
    files = glob.glob("covariance_matrices/*.pkl")
    np_arrays = []
    for file in files:
        array = load_np_array_from_pkl(file)
        np_arrays.append(array)
    Xcov = np.array(np_arrays)

    FgMDM = pyriemann.classification.FgMDM()
    FgMDM_score = results(FgMDM, Xcov, y, cv)
    print(FgMDM_score)
    """

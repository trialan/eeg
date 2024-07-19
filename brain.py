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
from eeg.utils import results, get_cv
from eeg.experiments.eigen_fgmdm import OldED
from tqdm import tqdm


def load_np_array_from_pkl(file_path):
    with open(file_path, 'rb') as file:
        np_array = pickle.load(file)
    return np_array


if __name__ == '__main__':
    #_, y = get_data()
    cv = get_cv()

    X = load_np_array_from_pkl("array_data.pkl")

    print(f"LENGTH: {len(X)}")
    sub_X = np.array_split(X, 10)

    sub_covs = []
    for i, sub in tqdm(enumerate(sub_X)):
        Xcov = pyriemann.estimation.Covariances('oas').fit_transform(sub)
        with open(f'cov_data_{i}.pkl', 'wb') as file:
            pickle.dump(Xcov, file)

    #FgMDM = pyriemann.classification.FgMDM()
    #FgMDM_score = results(FgMDM, Xcov, y, cv)
    #print(FgMDM_score)
    """
    import glob
    files = glob.glob("*.pkl")
    np_arrays = []
    files = sorted(files, key=lambda x: int(x.split('_')[2].split('.')[0]))
    for file_path in files[:-1]:
        with open(file_path, 'rb') as file:
            np_array = pickle.load(file)
            np_arrays.append(np_array)
    """




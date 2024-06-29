import random
from eeg.data import get_data
import matplotlib.pyplot as plt

from tqdm import tqdm
from mne.decoding import CSP, UnsupervisedSpatialFilter

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import pyriemann
from eeg.laplacian import (get_electrode_coordinates,
                           compute_scalp_eigenvectors_and_values,
                           create_triangular_dmesh, ED)
from eeg.ml import results


"""
Data augmentation experiment:How does FgMDM perform with a
fraction of the data.

# Approach 1: (EEG, target) augmentation / reduction

First way we can reduce the data is fewer (EEG, target) pairs as such:


    X, y = get_data()
    frac=0.9
    n_samples = int(frac*len(X))
    X_frac = X[:n_samples]
    y_frac = y[:n_samples]


Fraction  |   FgMDM_score
-------------------------
  0.5     |   0.6247
  0.9     |   0.6291
  1.0     |   0.6199 #weird that this is lower. perhaps noise from CV.

--> It seems like augmenting the number of (EEG, target) pairs isn't
    likely to get us very far. That's quite nice actually because this
    would be expensive.


# Approach 2: n_channel augmentation / reduction

The second way we can reduce the data is fewer channels: let's randomly
pick a subset of channels, and see if more channels would be helpful. If
so, we could then follow arXiv:2403.05645v2 and use Taken's theorem to
get synthetic channel data. We can keep n channels as such:


    n = 3
    channel_indices = random.sample(range(64), n)
    X_frac = np.array([sub_X[channel_indices, :] for sub_X in X])


Fraction  |   Avg. FgMDM_score
-------------------------
   3      |   0.5583  (kept channels [20, 49, 33]
   3      |   0.5734  (kept channels [19, 3, 13]
   3      |   0.5493  (kept channels [58, 6, 32]
Mean for 3 is: 0.5603 (0.0057)
   6      |   0.5681
   6      |   0.5392
   6      |   0.6080
Mean for 6 is: 0.5718 (0.0162), i.e. one SE better.
  32      |   0.6176
  32      |   0.6040
  32      |   0.6124
  64      |   0.6199


--> It seems like perhaps a bit of wishful thinking to assume that doing
    channel augmentation would help. But of course in the arXiv paper
    cited above it is crucial because they start by picking only three
    electrodes, and in that case increasing the number of channels is
    clearly going to yield results.

"""


if __name__ == '__main__':
    X, y = get_data()
    cv = ShuffleSplit(5, test_size=0.2, random_state=42)

    n = 3
    channel_indices = random.sample(range(64), n)
    X_frac = np.array([sub_X[channel_indices, :] for sub_X in X])

    print("FgMDM")
    Xcov = pyriemann.estimation.Covariances('oas').fit_transform(X_frac)
    FgMDM = pyriemann.classification.FgMDM()
    FgMDM_score = results(FgMDM, Xcov, y_frac, cv)
    print(FgMDM_score)


import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker

def tucker_decompose_dataset(X, ranks=(None, None, None)):
    """
    Perform Tucker decomposition on a dataset of fMRI images.

    Parameters:
    X : numpy array of shape (N, 64, 64, 32)
        The fMRI dataset with N samples
    ranks : tuple of 3 integers or None
        The ranks for each mode of the Tucker decomposition.
        If None, the rank will be automatically determined.

    Returns:
    X_decomposed : numpy array of shape (N, p, q, r)
        The dataset after Tucker decomposition
    """
    tl.set_backend('numpy')
    N, height, width, depth = X.shape
    decomposed_samples = []
    for i in range(N):
        core, factors = tucker(X[i], rank=ranks)
        p, q, r = core.shape
        decomposed_samples.append(core)
    X_decomposed = np.stack(decomposed_samples)
    return X_decomposed



import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
from sklearn.feature_selection import mutual_info_classif


def tucker_decompose_dataset(X, ranks=(None, None, None)):
    tl.set_backend('numpy')
    N, height, width, depth = X.shape
    decomposed_samples = []
    for i in range(N):
        core, factors = tucker(X[i], rank=ranks)
        p, q, r = core.shape
        decomposed_samples.append(core)
    X_decomposed = np.stack(decomposed_samples)
    return X_decomposed


def keep_top_MI_voxels(X, y, N):
    top_voxels = _select_top_voxels(X, y, N)
    mask = create_voxel_mask(top_voxels)
    X_masked = X * mask[np.newaxis, :, :, :]
    return X_masked


def _select_top_voxels(X, y, N):
    X_reshaped = X.reshape(X.shape[0], -1)
    mi_scores = mutual_info_classif(X_reshaped, y)
    mi_scores_3d = mi_scores.reshape(64, 64, 32)
    top_indices = np.argpartition(mi_scores_3d.ravel(), -N)[-N:]
    indices = np.array(np.unravel_index(top_indices, (64, 64, 32))).T
    return indices


def _create_voxel_mask(top_voxels, shape=(64, 64, 32)):
    mask = np.zeros(shape, dtype=bool)
    mask[top_voxels[:, 0], top_voxels[:, 1], top_voxels[:, 2]] = True
    return mask



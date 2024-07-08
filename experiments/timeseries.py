import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from eeg.laplacian import (get_electrode_coordinates,
                           compute_scalp_eigenvectors_and_values,
                           get_256D_eigenvectors,
                           create_triangular_dmesh, ED)
from eeg.ml import results
from eeg.data import get_data
import pandas as pd
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.model_selection import train_test_split
from sktime.datatypes._panel._convert import from_2d_array_to_nested
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sklearn.metrics import accuracy_score


"""
Idea for this experiment:
    - train a time-series classifier for each electrode
    - ensemble the predictions for each electrode-level clf

Accuracy: 0.5324947589098532

The highest score for any one electrode was 0.5401
"""


def get_train_test_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=42)
    X_train = make_sktime_compatible(X_train)
    X_test = make_sktime_compatible(X_test)
    return X_train, X_test, y_train, y_test


def make_sktime_compatible(X):
    """ sktime works with pd.Series objects """
    assert len(X.shape) == 2
    pd_series = [pd.Series(ts) for ts in X]
    X_df = pd.DataFrame({f'time_series': pd_series})
    return X_df


if __name__ == '__main__':
    X, y = get_data(2)
    cv = ShuffleSplit(5, test_size=0.2, random_state=42)

    xyz_coords = get_electrode_coordinates()
    mesh = create_triangular_dmesh(xyz_coords)
    eigenvectors, eigenvals = compute_scalp_eigenvectors_and_values(mesh)

    n_dims = 5
    ed = ED(n_dims, eigenvectors)
    X_ed = np.array([ed.transform(sub_X.T).T for sub_X in X])

    xyz_coords = get_electrode_coordinates()
    mesh = create_triangular_dmesh(xyz_coords)
    eigenvectors, eigenvals = compute_scalp_eigenvectors_and_values(mesh)

    ### Make the time-series datasets
    ts_ds = []
    for j in range(X_ed.shape[1]):
        dim_ts_ds = X_ed[:, j, :]
        ts_ds.append(dim_ts_ds)

    ts_ds = np.array(ts_ds)
    print(ts_ds.shape)

    ### Train a classifier per channel
    scores = []
    classifiers = []
    for i in tqdm(range(n_dims), desc="Training electrode-level clf"):
        clf = TimeSeriesForestClassifier(n_jobs=-1, random_state=42)
        score = cross_val_score(clf, ts_ds[i], y, cv=cv)
        clf.fit(ts_ds[i], y)
        scores.append(score)
        classifiers.append(clf)

    gscores = []
    for n_components in tqdm(range(2, n_dims), desc="Training LDA"):
        ed = ED(n_components, eigenvectors)
        X_ed = np.array([ed.transform(sub_X.T).T for sub_X in X])
        import pdb;pdb.set_trace() 
        ### Replace each time-series with the output of the relevant classifier
        X_ap = np.array([classifiers[i].predict(sub_X) for i,sub_X in enumerate(X_ed)])

        lda = LinearDiscriminantAnalysis()
        score = results(lda, X_ap, y, cv)
        gscores.append(score)

    plt.plot(list(range(2, n_dims)), gscores, marker='o', linestyle='-', label='LSP+LDA')
    plt.legend()
    plt.show()



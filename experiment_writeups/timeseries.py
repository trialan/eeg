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
from sklearn.model_selection import train_test_split
from sktime.datatypes._panel._convert import from_2d_array_to_nested
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sklearn.metrics import accuracy_score


"""
Idea for this experiment:
    - train a time-series classifier for each electrode
    - ensemble the predictions for each electrode-level clf

Accuracy: 0.5324947589098532
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
    X, y = get_data()

    ### Make the time-series datasets
    ts_ds = []
    for j in range(X.shape[1]):
        dim_ts_ds = X[:, j, :]
        ts_ds.append(dim_ts_ds)

    ts_ds = np.array(ts_ds)
    print(ts_ds.shape)

    ### Train a classifier per channel
    scores = []
    classifiers = []
    preds = []
    for i in tqdm(range(64), desc="Training electrode-level clf"):
        X_train, X_test, y_train, y_test = get_train_test_split(ts_ds[i], y)

        clf = TimeSeriesForestClassifier(n_jobs=-1,
                                         random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score = accuracy_score(y_test, y_pred)

        preds.append(y_pred)
        scores.append(score)
        classifiers.append(clf)

    voting_preds = sum(preds) / len(preds)
    int_voting_preds = [round(vp) for vp in voting_preds]
    print(accuracy_score(int_voting_preds, y_test))


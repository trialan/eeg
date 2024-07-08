import numpy as np
from tqdm import tqdm
import pyriemann

import matplotlib.pyplot as plt
from matplotlib_venn import venn3

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F

from eeg.data import get_data
from eeg.utils import results, get_cv, avg_power_matrix
from eeg.experiments.ensemble import EDFgMDM
from eeg.laplacian import compute_scalp_eigenvectors_and_values
from eeg.ml import (assemble_classifer_PCACSPLDA,
                    assemble_classifer_CSPLDA,
                    PCA)


def predict_with_router(row, cov_matrix, router):
    chosen_classifier = get_best_classifier(row, cov_matrix, router)
    if chosen_classifier == 0:
        return edf.predict(np.array([row]))[0]
    elif chosen_classifier == 1:
        return pcl.predict(np.array([row]))[0]
    else:
        return cl.predict(np.array([row]))[0]


def get_best_classifier(row, cov_matrix, router):
    input_tensor = np.array([cov_matrix])
    router_output = router.predict(input_tensor)[0]
    return router_output


if __name__ == '__main__':
    X, y = get_data()
    cv = get_cv()
    eigenvectors, eigenvals = compute_scalp_eigenvectors_and_values()

    X_temp, X_test, y_temp, y_test = train_test_split(X, y,
                                                      test_size=0.15,
                                                      random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp,
                                                      test_size=0.1765,
                                                      random_state=42)

    print(f"Train X size: {X_train.shape}")
    print(f"Train y size: {y_train.shape}")

    print(f"Val X size: {X_val.shape}")
    print(f"Val y size: {y_val.shape}")

    print(f"Test X size: {X_test.shape}")
    print(f"Test y size: {y_test.shape}")

    edf = EDFgMDM(n_components=24, eigenvectors=eigenvectors)
    edf.fit(X_train, y_train)
    edf_y_pred = edf.predict(X_test)
    score = accuracy_score(edf_y_pred, y_test)
    print(f"###### Laplacian + FgMDM score: {score}\n") #0.6425

    pcl = assemble_classifer_PCACSPLDA(n_components=30)
    pcl.fit(X_train, y_train)
    pcl_y_pred = pcl.predict(X_test)
    score = accuracy_score(pcl_y_pred, y_test)
    print(f"###### PCA+CSP+LDA score: {score}\n") #0.6285

    cl = assemble_classifer_CSPLDA(n_components=10)
    cl.fit(X_train, y_train)
    cl_y_pred = cl.predict(X_test)
    score = accuracy_score(cl_y_pred, y_test)
    print(f"###### CSP+LDA score: {score}\n") #0.6327

    #Make the Venn diagram
    good_edf_subjects = np.where(edf_y_pred == y_test)[0]
    good_pcl_subjects = np.where(pcl_y_pred == y_test)[0]
    good_cl_subjects = np.where(cl_y_pred == y_test)[0]

    edf_set = set(good_edf_subjects)
    pcl_set = set(good_pcl_subjects)
    cl_set = set(good_cl_subjects)

    plt.figure(figsize=(8, 8))
    venn = venn3([edf_set, pcl_set, cl_set],
                 ('Laplacian + FgMDM (24 components)',
                  'PCA+CSP+LDA (30 components)',
                  'CSP+LDA (10 components)'))

    #Make the routing dataset using the validation dataset
    edf_preds = edf.predict_proba(X_val)
    pcl_preds = pcl.predict_proba(X_val)
    cl_preds = cl.predict_proba(X_val)

    X_router = X_val
    y_router = np.zeros(len(X_val))

    for i in range(len(X_val)):
        prob_1 = edf_preds[i][1]
        prob_2 = pcl_preds[i][1]
        prob_3 = cl_preds[i][1]

        diff_1 = abs(prob_1 - y_val[i])
        diff_2 = abs(prob_2 - y_val[i])
        diff_3 = abs(prob_3 - y_val[i])

        best_classifier = np.argmin([diff_1, diff_2, diff_3])
        y_router[i] = best_classifier

    def get_covariances(M):
        cov = pyriemann.estimation.Covariances('oas').fit_transform(M)
        return cov

    Xcov = get_covariances(X_router)

    router = EDFgMDM(n_components=64, eigenvectors=eigenvectors)
    router.fit(Xcov, y_router)
    score = results(router, Xcov, y_router, cv)
    print(f"###### Router score: {score}\n")

    X_test_cov = get_covariances(X_test)

    y_pred = []
    for ix, row in enumerate(X_test):
        cov_matrix = X_test_cov[ix]
        out = predict_with_router(row, cov_matrix, router)
        y_pred.append(out)
    score = accuracy_score(y_pred, y_test)
    print(f"###### Meta-clf score: {score}\n")



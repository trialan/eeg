from tqdm import tqdm
from venn import venn
import matplotlib.pyplot as plt
import numpy as np
import pyriemann
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from eeg.data import get_data
from eeg.experiments.eigen_fgmdm import EDFgMDM
from eeg.laplacian import compute_scalp_eigenvectors_and_values
from eeg.utils import results, get_cv, get_covariances
from eeg.ml import (assemble_classifer_PCACSPLDA,
                    assemble_classifer_CSPLDA)

"""
At first I thought that this improved the accuracy past 66%, but
actually this was only the case for a single fold. When you do
5-fold CV as in the paper you get these results:

    Mean Meta-clf score: 0.6159
    Mean router score: 0.3606
    Mean Laplacian + FgMDM (n=24) model score: 0.6230
    Mean CSP+LDA model score: 0.6092
    Mean PCA+CSP+LDA model score: 0.6140

66% was achieved with this train-val-test split:

    X_temp, X_test, y_temp, y_test = train_test_split(X, y,
                                                      test_size=0.15,
                                                      random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp,
                                                      test_size=0.1765,
                                                      random_state=42)
"""


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


def generate_datasets(X, y, cv):
    datasets = []

    for train_index, test_index in cv.split(X):
        X_train_full, X_test = X[train_index], X[test_index]
        y_train_full, y_test = y[train_index], y[test_index]

        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)
        datasets.append((X_train, y_train, X_val, y_val, X_test, y_test))
    return datasets


if __name__ == '__main__':
    X, y = get_data()
    cv = get_cv()
    eigenvectors, eigenvals = compute_scalp_eigenvectors_and_values()
    datasets = generate_datasets(X, y, cv)

    meta_clf_scores = []
    edf_scores = []
    pcl_scores = []
    cl_scores = []
    router_scores = []


    for i, (X_train, y_train, X_val, y_val, X_test, y_test) in tqdm(enumerate(datasets)):

        #Train a EDFgMDM classifier
        edf = EDFgMDM(n_components=24, eigenvectors=eigenvectors)
        edf.fit(X_train, y_train)
        edf_y_pred = edf.predict(X_test)
        edf_score = accuracy_score(edf_y_pred, y_test)
        edf_scores.append(edf_score)

        #Train a PCA+CSP+LDA classifier
        pcl = assemble_classifer_PCACSPLDA(n_components=30)
        pcl.fit(X_train, y_train)
        pcl_y_pred = pcl.predict(X_test)
        pcl_score = accuracy_score(pcl_y_pred, y_test)
        pcl_scores.append(pcl_score)

        #Train a CSP + LDA classifier
        cl = assemble_classifer_CSPLDA(n_components=10)
        cl.fit(X_train, y_train)
        cl_y_pred = cl.predict(X_test)
        cl_score = accuracy_score(cl_y_pred, y_test)
        cl_scores.append(cl_score)

        #Make a Venn diagram of which model is predicting what correctly
        good_edf_subjects = np.where(edf_y_pred == y_test)[0]
        good_pcl_subjects = np.where(pcl_y_pred == y_test)[0]
        good_cl_subjects = np.where(cl_y_pred == y_test)[0]

        edf_set = set(good_edf_subjects)
        pcl_set = set(good_pcl_subjects)
        cl_set = set(good_cl_subjects)

        data = {"Laplacian + FgMDM": edf_set,
                "PCA + CSP + LDA": pcl_set,
                "CSP + LDA": cl_set}
        venn(data)
        #plt.show()

        #Make a routing dataset. Which model did best on each
        #row in the validation set?
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

        Xcov = get_covariances(X_router)

        #Train a router using this dataset
        router = EDFgMDM(n_components=64, eigenvectors=eigenvectors)
        router.fit(Xcov, y_router)
        router_score = results(router, Xcov, y_router, cv)
        router_scores.append(router_score)

        X_test_cov = get_covariances(X_test)

        #Get a meta-clf score
        y_pred = []
        for ix, row in enumerate(X_test):
            cov_matrix = X_test_cov[ix]
            out = predict_with_router(row, cov_matrix, router)
            y_pred.append(out)
        score = accuracy_score(y_pred, y_test)
        meta_clf_scores.append(score)

    print(f"Mean Meta-clf score: {np.mean(meta_clf_scores)}")
    print(f"Mean router score: {np.mean(router_scores)}")
    print(f"Mean Laplacian + FgMDM (n=24) model score: {np.mean(edf_scores)}")
    print(f"Mean CSP+LDA model score: {np.mean(cl_scores)}")
    print(f"Mean PCA+CSP+LDA model score: {np.mean(pcl_scores)}")



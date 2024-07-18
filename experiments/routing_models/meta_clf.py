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
from eeg.utils import results, get_cv, get_covariances, get_fraction
from eeg.plot_reproduction import (
    assemble_classifer_PCACSPLDA,
    assemble_classifer_CSPLDA,
)
from eeg.experiments.channel_selection.channels import get_sorted_channels

"""
At first I thought that this improved the accuracy past 66%, but
actually this was only the case for a single fold. When you do
5-fold CV as in the paper you get these results:

When the router is trained on the val set:
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)

    Mean Meta-clf score: 0.6159
    Mean router score: 0.3606
    Mean Laplacian + FgMDM (n=24) model score: 0.6230
    Mean CSP+LDA model score: 0.6092
    Mean PCA+CSP+LDA model score: 0.6140

When the router is trained on the full train set (pretty sure we can do that):

    Mean Meta-clf score: 0.6283
    Mean router score: 0.3844
    Mean Laplacian + FgMDM (n=24) model score: 0.6338 #seems a bit low?
    Mean CSP+LDA model score: 0.6174
    Mean PCA+CSP+LDA model score: 0.6235

66% was achieved with this train-val-test split:

    X_temp, X_test, y_temp, y_test = train_test_split(X, y,
                                                      test_size=0.15,
                                                      random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp,
                                                      test_size=0.1765,
                                                      random_state=42)


Mean Meta-clf score: 0.630188679245283
Mean router score: 0.5301570680628271 (this is not great)
Mean Laplacian + FgMDM (n=24) model score: 0.6337526205450734
Mean 24-channel FgMDM score: 0.6377358490566039 (this is better than Xu et al by a tiny bit)


Finally by using cutoffs of 95th percentile, and ensemble when not sure:

Mean Meta-clf score: 0.6444444444444445 (this is 1.07% better than laplacian+FgMDM).
Mean router score: 0.5301570680628271
Mean Laplacian + FgMDM (n=24) model score: 0.6337526205450734
Mean 24-channel FgMDM score: 0.6377358490566039

Doing the same thing with thrown in FgMDM (only ensemble top 2 models when un-sure)
    Mean Meta-clf score: 0.6486373165618449
    Mean router score: 0.3660732984293193
    Mean Laplacian + FgMDM (n=24) model score: 0.6371069182389937
    Mean 24-channel FgMDM score: 0.6377358490566039
    Mean FgMDM score: 0.619916142557652
    using these cutoffs:
    cutoff_EDF = 0.3350226332121478
    cutoff_24C = 0.3346899916548326
    cutoff_F = 0.3343728619094913

And again doing the same thing with PCL and CL models thrown in too:
    Mean Meta-clf score: 0.6486373165618449
    Mean router score: 0.3120942408376964
    Mean Laplacian + FgMDM (n=24) model score: 0.6371069182389937
    Mean 24-channel FgMDM score: 0.6377358490566039
    Mean FgMDM score: 0.619916142557652
    Mean PCL score: 0.6316561844863731
    Mean CL score: 0.6174004192872118
    using these cutoffs:
    cutoff_EDF = 0.2055192066980013
    cutoff_24C = 0.2037104615796535
    cutoff_F = 0.20484568672504827
    cutoff_PCL = 0.20558124514075884
    cutoff_CL = 0.2054181690938479

    where I had SEs:
    In [26]: np.std(meta_clf_scores) / np.sqrt(5)
    Out[26]: 0.0048299994139485705

    In [27]: np.std(edf_scores) / np.sqrt(5)
    Out[27]: 0.0027237537557923204
"""


cutoff_EDF = 0.2055192066980013
cutoff_24C = 0.2037104615796535
cutoff_F = 0.20484568672504827
cutoff_PCL = 0.20558124514075884
cutoff_CL = 0.2054181690938479


def predict_with_router(row, cov_matrix, channel_cov_matrix, router):
    chosen_classifier = get_best_classifier(row, cov_matrix, router)
    if chosen_classifier == 0:
        # edf model
        return edf.predict(np.array([row]))[0]
    elif chosen_classifier == 1:
        return scf.predict(np.array([channel_cov_matrix]))[0]
    elif chosen_classifier == 2:
        return f.predict(np.array([cov_matrix]))[0]
    elif chosen_classifier == 3:
        return pcl.predict(np.array([row]))[0]
    elif chosen_classifier == 4:
        return cl.predict(np.array([row]))[0]
    else:
        edf_proba = edf.predict_proba(np.array([row]))[0]
        scf_proba = scf.predict_proba(np.array([channel_cov_matrix]))[0]
        # look at probability it's 1, if that's over 0.5 -> answer is 1
        # else answer is zero
        ensemble_proba = (edf_proba[1] + scf_proba[1]) / 2.0
        return round(ensemble_proba)


def get_best_classifier(row, cov_matrix, router):
    input_tensor = np.array([cov_matrix])
    router_probas = router.predict_proba(input_tensor)[0]
    if router_probas[0] >= cutoff_EDF:
        router_output = 0
    elif router_probas[1] >= cutoff_24C:
        router_output = 1
    elif router_probas[2] >= cutoff_F:
        router_output = 2
    elif router_probas[3] >= cutoff_PCL:
        router_output = 3
    elif router_probas[4] >= cutoff_CL:
        router_output = 4
    else:
        router_output = 5  # Ensemble model C
    return router_output


def OLD_get_best_classifier(row, cov_matrix, router):
    input_tensor = np.array([cov_matrix])
    router_probas = router.predict_proba(input_tensor)[0]
    return router_probas


def generate_datasets(X, y, cv):
    datasets = []

    for train_index, test_index in cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        datasets.append((X_train, y_train, X_test, y_test))
    return datasets


if __name__ == "__main__":
    X, y = get_data(5)
    cv = get_cv()
    eigenvectors, eigenvals = compute_scalp_eigenvectors_and_values()
    datasets = generate_datasets(X, y, cv)
    sorted_channels = get_sorted_channels(X, y, cv)

    meta_clf_scores = []
    edf_scores = []
    scf_scores = []
    f_scores = []
    pcl_scores = []
    cl_scores = []
    router_scores = []

    for i, (X_train, y_train, X_test, y_test) in tqdm(enumerate(datasets)):

        # Train a EDFgMDM classifier (laplacian + fgmdm)
        print("Training EDFgMDM")
        edf = EDFgMDM(n_components=24, eigenvectors=eigenvectors)
        edf.fit(X_train, y_train)
        edf_y_pred = edf.predict(X_test)
        edf_score = accuracy_score(edf_y_pred, y_test)
        edf_scores.append(edf_score)

        # Train a 20-channel FgMDM classifier
        print("20C-FgMDM")
        channel_X_train = X_train[:, sorted_channels[:20], :]
        channel_X_train_cov = get_covariances(channel_X_train)

        channel_X_test = X_test[:, sorted_channels[:20], :]
        channel_X_test_cov = get_covariances(channel_X_test)

        scf = pyriemann.classification.FgMDM()
        scf.fit(channel_X_train_cov, y_train)
        scf_y_pred = scf.predict(channel_X_test_cov)
        scf_score = accuracy_score(scf_y_pred, y_test)
        scf_scores.append(scf_score)

        # Train FgMDM
        print("Training FgMDM")
        Xcov = get_covariances(X_train)
        Xcov_test = get_covariances(X_test)
        f = pyriemann.classification.FgMDM()
        f.fit(Xcov, y_train)
        f_y_pred = f.predict(Xcov_test)
        f_score = accuracy_score(f_y_pred, y_test)
        f_scores.append(f_score)

        # Train PCA+CSP+LDA
        print("Training PCL")
        pcl = assemble_classifer_PCACSPLDA(n_components=42)
        pcl.fit(X_train, y_train)
        pcl_y_pred = pcl.predict(X_test)
        pcl_score = accuracy_score(pcl_y_pred, y_test)
        pcl_scores.append(pcl_score)

        # Train CSP+LDA
        print("Training CL")
        cl = assemble_classifer_CSPLDA(n_components=10)
        cl.fit(X_train, y_train)
        cl_y_pred = cl.predict(X_test)
        cl_score = accuracy_score(cl_y_pred, y_test)
        cl_scores.append(cl_score)

        # Make a Venn diagram of which model is predicting what correctly
        good_edf_subjects = np.where(edf_y_pred == y_test)[0]
        good_scf_subjects = np.where(scf_y_pred == y_test)[0]
        good_f_subjects = np.where(f_y_pred == y_test)[0]
        good_pcl_subjects = np.where(pcl_y_pred == y_test)[0]
        good_cl_subjects = np.where(cl_y_pred == y_test)[0]

        edf_set = set(good_edf_subjects)
        scf_set = set(good_scf_subjects)
        f_set = set(good_f_subjects)
        pcl_set = set(good_pcl_subjects)
        cl_set = set(good_cl_subjects)

        data = {
            "Laplacian + FgMDM": edf_set,
            "24-channel FgMDM": scf_set,
            "FgMDM": f_set,
            "PCA+CSP+LDA (n=42)": pcl_set,
            "CSP+LDA (n=10)": cl_set,
        }
        venn(data)
        plt.show()

        # Make a routing labels dataset (y_router)
        X_router = X_train
        channel_X_router_cov = channel_X_train_cov

        edf_preds = edf.predict_proba(X_router)
        scf_preds = scf.predict_proba(channel_X_router_cov)
        f_preds = f.predict_proba(Xcov)
        pcl_preds = pcl.predict_proba(X_router)
        cl_preds = cl.predict_proba(X_router)

        y_router = np.zeros(len(X_router))

        for i in range(len(X_router)):
            prob_1 = edf_preds[i][1]
            prob_2 = scf_preds[i][1]
            prob_3 = f_preds[i][1]
            prob_4 = pcl_preds[i][1]
            prob_5 = cl_preds[i][1]

            diff_1 = abs(prob_1 - y_train[i])
            diff_2 = abs(prob_2 - y_train[i])
            diff_3 = abs(prob_3 - y_train[i])
            diff_4 = abs(prob_4 - y_train[i])
            diff_5 = abs(prob_5 - y_train[i])

            best_classifier = np.argmin([diff_1, diff_2, diff_3, diff_4, diff_5])
            y_router[i] = best_classifier

        # Train a router using this dataset
        router = EDFgMDM(n_components=64, eigenvectors=eigenvectors)
        router.fit(Xcov, y_router)
        router_score = results(router, Xcov, y_router, cv)
        router_scores.append(router_score)

        channel_X_test = X_test[:, sorted_channels[:20], :]
        channel_X_test_cov = get_covariances(channel_X_test)

        1 / 0
        # Get a meta-clf score
        y_pred = []
        choices = []
        for ix, row in enumerate(X_test):
            channel_cov_matrix = channel_X_test_cov[ix]
            cov_matrix = Xcov[ix]
            choice = OLD_get_best_classifier(row, cov_matrix, router)
            choices.append(choice)
            out = predict_with_router(row, cov_matrix, channel_cov_matrix, router)
            y_pred.append(out)
        score = accuracy_score(y_pred, y_test)
        meta_clf_scores.append(score)

    print(f"Mean Meta-clf score: {np.mean(meta_clf_scores)}")
    print(f"Mean router score: {np.mean(router_scores)}")
    print(f"Mean Laplacian + FgMDM (n=24) model score: {np.mean(edf_scores)}")
    print(f"Mean 24-channel FgMDM score: {np.mean(scf_scores)}")
    print(f"Mean FgMDM score: {np.mean(f_scores)}")
    print(f"Mean PCL score: {np.mean(pcl_scores)}")
    print(f"Mean CL score: {np.mean(cl_scores)}")

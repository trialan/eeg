from eeg.main import *

"""
    This script exists to reproduce fig 3(a) from Xu et. al.
    https://hal.science/hal-03477057/documen://hal.science/hal-03477057/document

    So far it is missing:
        - Laplacian + FgMDM
        - Laplacian + CSP + LDA
"""


def assemble_classifer_PCACSPLDA(n_components):
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    pca = UnsupervisedSpatialFilter(PCA(n_components), average=False)
    clf = Pipeline([("PCA", pca), ("CSP", csp), ("LDA", lda)])
    return clf


def assemble_classifer_CSPLDA(n_components):
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    clf = Pipeline([("CSP", csp), ("LDA", lda)])
    return clf


def assemble_classifer_PCAFgMDM(n_components):
    pca = UnsupervisedSpatialFilter(PCA(n_components), average=False)
    FgMDM = pyriemann.classification.FgMDM()
    clf = Pipeline([("PCA", pca), ("FgMDM", FgMDM)])
    return clf



if __name__ == '__main__':
    X, y = get_data()
    cv = ShuffleSplit(5, test_size=0.2, random_state=42)
    component_numbers = list(range(1, 50, 5))

    print("CSP+LDA")
    scores = []
    for n_components in tqdm(component_numbers):
        clf = assemble_classifer_CSPLDA(n_components)
        score = results(clf, X, y, cv)
        scores.append(score)
    plt.plot(component_numbers, scores, marker='o', linestyle='-', label='CSP+LDA')


    #'oas' because: https://github.com/pyRiemann/pyRiemann/issues/65
    print("PCA+FgMDM")
    scores = []
    for n_components in tqdm(component_numbers):
        n_epochs, n_channels, n_times = X.shape
        X_reshaped = X.reshape(n_times * n_epochs, n_channels)
        pca = PCA(n_components=n_components)
        pca.fit(X_reshaped)

        X_pca = np.array([pca.transform(epoch.T).T for epoch in X])
        Xcov = pyriemann.estimation.Covariances('oas').fit_transform(X_pca)

        FgMDM = pyriemann.classification.FgMDM()
        score = results(FgMDM, Xcov, y, cv)
        scores.append(score)
    plt.plot(component_numbers, scores, marker='o', linestyle='-', label='PCA+FgMDM')

    print("FgMDM")
    Xcov = pyriemann.estimation.Covariances('oas').fit_transform(X)
    FgMDM = pyriemann.classification.FgMDM()
    FgMDM_score = results(FgMDM, Xcov, y, cv)
    plt.axhline(y=FgMDM_score, linestyle='--', label='FgMDM')

    print("PCA+CSP+LDA")
    scores = []
    for n_components in tqdm(component_numbers):
        clf = assemble_classifer_PCACSPLDA(n_components)
        score = results(clf, X, y, cv)
        scores.append(score)
    plt.plot(component_numbers, scores, marker='o', linestyle='-', label='PCA+CSP+LDA')


    plt.xlabel("Number of components")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.show()

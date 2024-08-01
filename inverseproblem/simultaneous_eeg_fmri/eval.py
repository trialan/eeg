import numpy as np
import pyriemann
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import resample

from eeg.inverseproblem.simultaneous_eeg_fmri.cnn import FMRI_CNN, train
from eeg.inverseproblem.simultaneous_eeg_fmri.eeg_data import get_eeg_data


#root_dir = "/Users/thomasrialan/Documents/code/DS116/"
root_dir = "/root/DS116/"


def balance_and_shuffle(X, y):
    # Separate majority and minority classes
    X_majority = X[y == 0]
    y_majority = y[y == 0]
    X_minority = X[y == 1]
    y_minority = y[y == 1]
    N = len(X_minority)
    # Upsample minority class
    X_minority_upsampled, y_minority_upsampled = resample(
        X_minority, y_minority, replace=True, n_samples=N, random_state=42
    )
    # Combine majority class with upsampled minority class
    X_balanced = np.vstack((X_majority[:N], X_minority_upsampled))
    y_balanced = np.hstack((y_majority[:N], y_minority_upsampled))
    # Shuffle the balanced dataset
    shuffle_indices = np.random.permutation(len(y_balanced))
    X_balanced_shuffled = X_balanced[shuffle_indices]
    y_balanced_shuffled = y_balanced[shuffle_indices]
    print(X_balanced.shape)
    print(y_balanced.shape)
    return X_balanced_shuffled, y_balanced_shuffled


def results_and_train(clf, X, y, cv, model=None, is_nn=False):
    scores = []
    for train_index, test_index in cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if is_nn:
            # Train the neural network
            train(model, X_train, y_train)
            # Evaluate the neural network
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test)
                y_pred = model(X_test_tensor).squeeze()
                y_pred = (y_pred > 0.5).float()
                accuracy = (y_pred == torch.FloatTensor(y_test)).float().mean()
            scores.append(accuracy.item())
        else:
            # Train and evaluate the sklearn classifier
            clf.fit(X_train, y_train)
            scores.append(clf.score(X_test, y_test))
    return np.mean(scores), np.std(scores) / np.sqrt(len(scores))

if __name__ == '__main__':
    X_eeg, y_eeg = get_eeg_data(root_dir)
    from eeg.utils import read_pickle, results, get_cv

    test_ixs = read_pickle("eeg_test_indices.pkl")
    X_eeg = X_eeg[test_ixs]
    y_eeg = y_eeg[test_ixs]

    X_eeg_balanced, y_eeg_balanced = balance_and_shuffle(X_eeg, y_eeg)

    cv = get_cv()
    Xcov = pyriemann.estimation.Covariances('oas').fit_transform(X_eeg_balanced)
    FgMDM = pyriemann.classification.FgMDM()
    score, se = results_and_train(FgMDM, Xcov, y_eeg_balanced, cv)
    print(score)
    print(se)

    from eeg.plot_reproduction import assemble_classifier_CSPLDA
    clf = assemble_classifier_CSPLDA(25)
    score, se = results_and_train(clf, X_eeg_balanced, y_eeg_balanced, cv)
    print(score)
    print(se)

    y_fmri = read_pickle("fmri_y.pkl")
    x_fmri = read_pickle("fmri_x.pkl")
    from eeg.inverseproblem.simultaneous_eeg_fmri.cnn import FMRI_CNN

    X_fmri_balanced, y_fmri_balanced = balance_and_shuffle(x_fmri, y_fmri)

    model = FMRI_CNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    score, se = results_and_train(None, X_fmri_balanced, y_fmri_balanced, cv, model = model, is_nn=True)

    print(score)
    print(se)











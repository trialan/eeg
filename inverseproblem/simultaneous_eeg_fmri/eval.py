import numpy as np
import pyriemann
import torch.nn as nn
import torch.optim as optim

from eeg.inverseproblem.simultaneous_eeg_fmri.cnn import FMRI_CNN, train

#root_dir = "/Users/thomasrialan/Documents/code/DS116/"
root_dir = "/root/DS116/"


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
    from eeg.inverseproblem.simultaneous_eeg_fmri.data import get_aligned_data

    data = get_aligned_data()
    X_eeg = data['eeg']['test']['X']
    y_eeg = data['eeg']['test']['y']
    X_fmri = data['fmri']['test']['X']
    y_fmri = data['fmri']['test']['y']


    cv = get_cv()
    Xcov = pyriemann.estimation.Covariances('oas').fit_transform(X_eeg)
    FgMDM = pyriemann.classification.FgMDM()
    score, se = results_and_train(FgMDM, Xcov, y_eeg, cv)
    print(score)
    print(se)

    from eeg.plot_reproduction import assemble_classifier_CSPLDA
    clf = assemble_classifier_CSPLDA(25)
    score, se = results_and_train(clf, X_eeg, y_eeg, cv)
    print(score)
    print(se)

    from eeg.inverseproblem.simultaneous_eeg_fmri.cnn import FMRI_CNN


    model = FMRI_CNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    score, se = results_and_train(None, X_fmri, y_fmri, cv, model = model, is_nn=True)

    print(score)
    print(se)











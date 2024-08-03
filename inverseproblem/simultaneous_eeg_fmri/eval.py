import numpy as np
import pyriemann
import torch.nn as nn
import torch.optim as optim

from eeg.utils import get_cv
from eeg.inverseproblem.simultaneous_eeg_fmri.cnn import FMRI_CNN, train
from eeg.inverseproblem.simultaneous_eeg_fmri.data import balance_and_shuffle

from eeg.inverseproblem.simultaneous_eeg_fmri._eeg_data import get_raw_eeg_data
from eeg.inverseproblem.simultaneous_eeg_fmri._fmri_data import get_bv_fmri_data

#root_dir = "/root/DS116/"
root_dir = "/Users/thomasrialan/Documents/code"


if __name__ == '__main__':
    X_eeg, y_eeg = get_raw_eeg_data(root_dir)
    X_fmri, y_fmri = get_bv_fmri_data(root_dir)

    assert np.array_equal(y_eeg, y_fmri)

    Xb_eeg, yb_eeg = balance_and_shuffle(X_eeg, y_eeg)
    Xb_fmri, yb_fmri = balance_and_shuffle(X_fmri, y_fmri)

    assert np.array_equal(yb_eeg, yb_fmri)

    from eeg.utils import read_pickle
    X_og_fmri = read_pickle("fmri_X.pkl")

    1/0

    from eeg.utils import results
    cv = get_cv()
    Xcov = pyriemann.estimation.Covariances('oas').fit_transform(Xb)
    FgMDM = pyriemann.classification.FgMDM()
    score, se = results(FgMDM, Xcov, yb, cv, return_se=True)
    print(score)
    print(se)

    from eeg.plot_reproduction import assemble_classifier_CSPLDA
    clf = assemble_classifier_CSPLDA(25)
    score, se = results(clf, Xb, yb, cv, return_se=True)
    print(score)
    print(se)

    """
    from eeg.inverseproblem.simultaneous_eeg_fmri.cnn import FMRI_CNN


    model = FMRI_CNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    score, se = results_and_train(None, X_fmri, y_fmri, cv, model = model, is_nn=True, fmri_data=data['fmri'])

    print(score)
    print(se)
    """











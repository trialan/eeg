
from eeg.utils import read_pickle
from eeg.inverseproblem.simultaneous_eeg_fmri.eeg_data import get_eeg_data
from eeg.inverseproblem.simultaneous_eeg_fmri.fmri_data import get_fmri_data


root_dir = "/root/DS116/"
#root_dir = "/Users/thomasrialan/Documents/code/DS116/"


if __name__ == '__main__':
    X_eeg, y_eeg = get_eeg_data()
    #X_fmri, y_fmri = get_fmri_data()
    y_fmri = read_pickle("fmri_y.pkl")

    assert y_eeg == y_fmri


from scipy.signal import butter, lfilter
from sklearn.model_selection import train_test_split
import scipy
from tqdm import tqdm
import numpy as np
import os
import zipfile

from eeg import data_path

"""
    We arbitrarily choose 1 to represent hands (6 in the raw data), and
    0 to represent feet (7 in the raw data).

    Original events labels can be found here:
    https://github.com/xiangzhang1015/Deep-Learning-for-BCI/blob/master/tutorial/1-Data.ipynb

    Open fists: label == 6
    Open feet: label == 7
    Rest: label == 10
"""


def get_channel_data():
    """ Data with dimensions (n_epochs, n_channels, n_times) for CSP use """
    data = read_raw_data()
    X = data[:, :, :64]
    X = apply_band_pass_filter([x for x in X])
    X = X.transpose(0,2,1)

    y = data[:, :, 64]
    y = np.where(y == 6, 1, 0)
    import pdb;pdb.set_trace() 
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    return X_train, X_test, y_train, y_test


def get_nonspatial_data():
    """ Non-spatial because we drop the info about channels. """
    data = read_raw_data()
    X, y = split_according_to_event_annotations(data)
    import pdb;pdb.set_trace() 
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    return X_train, X_test, y_train, y_test


def prepare_data(X, y):
    """ We want X: training data, and y, the target variable.  """
    X_bpf = apply_band_pass_filter(X)
    y_bpf = apply_band_pass_filter(y)

    X_train, X_test, y_train, y_test = train_test_split(
                X_bpf, y_bpf, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test


def apply_band_pass_filter(data):
    """
        A band-pass filter was used to extract the α and β
        bands (8-30 Hz) since they are particular discriminant
        for motor imagery [27] -- section III (a)

        We use a Butterworth bandpass filter.
    """

    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = scipy.signal.lfilter(b, a, data)
        return np.array(y)

    def butter_bandpass(lowcut, highcut, fs, order=5):
        """ Higher order means better filter """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    #160 is the sampling rate in EEGMIDB
    filtered_data = butter_bandpass_filter(data, 8, 30, 160)
    return filtered_data


def read_raw_data(n_files=2):
    """
        Read data files provided by Xian Zhang here:
        https://github.com/xiangzhang1015/Deep-Learning-for-BCI
    """
    npy_arrays = []
    for file_name in tqdm(os.listdir(data_path)[:n_files]):
        if file_name.endswith('.zip'):
            zip_path = os.path.join(data_path, file_name)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for member in zip_ref.namelist():
                    if member.endswith('.npy'):
                        with zip_ref.open(member) as npy_file:
                            npy_array = np.load(npy_file)
                            npy_arrays.append(npy_array)
    return np.array(npy_arrays, dtype=np.float64)


def split_according_to_event_annotations(data):
    data = np.concat(data)
    data = data[np.isin(data[:,-1], [6,7])]
    y = data[:, -1]
    X = data[:, :-1]
    y = np.where(y == 6, 1, 0)
    assert_shapes_are_good(X, y, data)
    return X, y


def assert_shapes_are_good(X, y, data):
    """ 64 channels, column 65 is the annotation """
    assert X.shape == (len(data), 64)
    assert y.shape == (len(data),)
    assert data.shape == (len(data), 65)



import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from eeg.utils import read_pickle
from eeg.inverseproblem.simultaneous_eeg_fmri._eeg_data import get_raw_eeg_data
from eeg.inverseproblem.simultaneous_eeg_fmri._fmri_data import get_raw_fmri_data

#root_dir = "/Users/thomasrialan/Documents/code/DS116/"
root_dir = "/root/DS116/"
balanced_length = 1916


def get_fmri_data(pickle=True):
    if pickle:
        X_fmri = read_pickle("fmri_X.pkl")
        y_frmi = read_pickle("fmri_y.pkl")
    else:
        X_fmri, y_fmri = get_raw_fmri_data(root_dir)
    X_fmri_balanced, y_fmri_balanced = balance_and_shuffle(X_fmri, y_fmri)
    assert len(y_fmri_balanced) == balanced_length
    train_ixs, val_ixs, test_ixs = create_split_indices()
    data = {"train": {"X": X_fmri_balanced[train_ixs],
                      "y": y_fmri_balanced[train_ixs]},
            "val":   {"X": X_fmri_balanced[val_ixs],
                      "y": y_fmri_balanced[val_ixs]},
            "test":  {"X": X_fmri_balanced[test_ixs],
                      "y": y_fmri_balanced[test_ixs]}}
    return data


def get_eeg_data():
    X_eeg, y_eeg = get_raw_eeg_data(root_dir)
    X_eeg_balanced, y_eeg_balanced = balance_and_shuffle(X_eeg, y_eeg)
    assert len(y_eeg_balanced) == balanced_length
    train_ixs, val_ixs, test_ixs = create_split_indices()
    data = {"train": {"X": X_eeg_balanced[train_ixs],
                      "y": y_eeg_balanced[train_ixs]},
            "val":   {"X": X_eeg_balanced[val_ixs],
                      "y": y_eeg_balanced[val_ixs]},
            "test":  {"X": X_eeg_balanced[test_ixs],
                      "y": y_eeg_balanced[test_ixs]}}
    return data


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



def create_split_indices(
    train_size=0.7, val_size=0.15, test_size=0.15, random_state=42
):
    # Create indices for the entire dataset
    indices = np.arange(balanced_length)

    # First split: separate test indices
    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )

    # Second split: separate train and validation indices
    val_size_adjusted = val_size / (train_size + val_size)
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_size_adjusted, random_state=random_state
    )

    return train_indices, val_indices, test_indices


def create_dataloaders(X_eeg, X_fmri, batch_size):
    train_indices, val_indices, test_indices = create_split_indices()

    # Split the data using the indices
    X_eeg_train, X_eeg_val, X_eeg_test = (
        X_eeg[train_indices],
        X_eeg[val_indices],
        X_eeg[test_indices],
    )
    X_fmri_train, X_fmri_val, X_fmri_test = (
        X_fmri[train_indices],
        X_fmri[val_indices],
        X_fmri[test_indices],
    )

    # Create datasets
    train_dataset = TensorDataset(X_eeg_train, X_fmri_train)
    val_dataset = TensorDataset(X_eeg_val, X_fmri_val)
    test_dataset = TensorDataset(X_eeg_test, X_fmri_test)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    d = get_eeg_data()


import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from eeg.utils import read_pickle
from eeg.inverseproblem.simultaneous_eeg_fmri._eeg_data import get_raw_eeg_data
from eeg.inverseproblem.simultaneous_eeg_fmri._fmri_data import get_raw_fmri_data

root_dir = "/Users/thomasrialan/Documents/code/DS116/"
# root_dir = "/root/DS116/"
balanced_length = 1916


def get_aligned_data(pickle=True):
    X_eeg, y_eeg = get_raw_eeg_data(root_dir)

    if pickle:
        X_fmri = read_pickle("fmri_X.pkl")
        y_fmri = read_pickle("fmri_y.pkl")
    else:
        X_fmri, y_fmri = get_raw_fmri_data(root_dir)

    assert np.array_equal(y_eeg, y_fmri), "EEG and fMRI labels are not aligned!"

    # Balance the data once
    indices_balanced = balance_indices(y_eeg)

    X_eeg_balanced = X_eeg[indices_balanced]
    X_fmri_balanced = X_fmri[indices_balanced]
    y_balanced = y_eeg[indices_balanced]  # or y_fmri, they should be the same

    assert len(y_balanced) == balanced_length

    # Create split indices once
    train_ixs, val_ixs, test_ixs = create_split_indices()

    data = {
        "eeg": {
            "train": {"X": X_eeg_balanced[train_ixs], "y": y_balanced[train_ixs]},
            "val": {"X": X_eeg_balanced[val_ixs], "y": y_balanced[val_ixs]},
            "test": {"X": X_eeg_balanced[test_ixs], "y": y_balanced[test_ixs]},
        },
        "fmri": {
            "train": {"X": X_fmri_balanced[train_ixs], "y": y_balanced[train_ixs]},
            "val": {"X": X_fmri_balanced[val_ixs], "y": y_balanced[val_ixs]},
            "test": {"X": X_fmri_balanced[test_ixs], "y": y_balanced[test_ixs]},
        },
    }
    for subset in ["train", "val", "test"]:
        assert np.array_equal(data['eeg'][subset]['y'], data['fmri'][subset]['y'])
    return data


def balance_indices(y, random_state=42):
    majority_class = int(np.mean(y))
    minority_indices = np.where(y != majority_class)[0]
    majority_indices = np.where(y == majority_class)[0]

    N = len(minority_indices)
    np.random.seed(random_state)
    majority_indices_sampled = np.random.choice(majority_indices, N, replace=False)

    balanced_indices = np.concatenate([minority_indices, majority_indices_sampled])
    np.random.shuffle(balanced_indices)

    return balanced_indices


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
    train_dataset = TensorDataset(X_eeg_train, X_fmri_train)
    val_dataset = TensorDataset(X_eeg_val, X_fmri_val)
    test_dataset = TensorDataset(X_eeg_test, X_fmri_test)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader



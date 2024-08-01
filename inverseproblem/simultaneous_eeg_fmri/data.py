import numpy as np
from sklearn.utils import resample

balanced_length = 1916


def balance_and_shuffle(X, y):
    X_majority = X[y == 0]
    y_majority = y[y == 0]
    X_minority = X[y == 1]
    y_minority = y[y == 1]

    N = len(X_minority)
    assert N == balanced_length / 2
    X_balanced = np.vstack((X_majority[:N], X_minority))
    y_balanced = np.hstack((y_majority[:N], y_minority))
    np.random.seed(42)
    shuffle_indices = np.random.permutation(len(y_balanced))
    X_balanced_shuffled = X_balanced[shuffle_indices]
    y_balanced_shuffled = y_balanced[shuffle_indices]
    print(X_balanced.shape)
    print(y_balanced.shape)

    return X_balanced_shuffled, y_balanced_shuffled


def create_dataloaders(X_eeg, X_fmri, batch_size):
    train_dataset = TensorDataset(X_eeg_train, X_fmri_train)
    val_dataset = TensorDataset(X_eeg_val, X_fmri_val)
    test_dataset = TensorDataset(X_eeg_test, X_fmri_test)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader




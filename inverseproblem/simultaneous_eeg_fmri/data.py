import numpy as np
import torch
import random
from sklearn.utils import resample

balanced_length = 2066

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def balance_and_shuffle(X, y):
    seed_everything()
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



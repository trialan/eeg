import numpy as np
import random
import sklearn
import pyriemann
from sklearn.model_selection import ShuffleSplit, cross_val_score
from scipy.stats import boxcox
import pickle


def read_pickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def write_pickle(obj, path):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    sklearn.utils.check_random_state(seed)


def get_covariances(M):
    #'oas' because: https://github.com/pyRiemann/pyRiemann/issues/65
    cov = pyriemann.estimation.Covariances('oas').fit_transform(M)
    assert len(cov) == len(M)
    return cov


def results(clf, X, y, cv, return_se=False):
    """ clf is classifier. This function trains the model and scores it """
    scores = cross_val_score(clf, X, y, cv=cv, n_jobs=None)
    if return_se:
        std_error = np.std(scores) / len(scores)
        return np.mean(scores), std_error
    return np.mean(scores)


def get_cv():
    """ Estimate the classification accuracy using 5-fold cross-validation as
        in Xu et. al. https://hal.science/hal-03477057 """
    cv = ShuffleSplit(5, test_size=0.2, random_state=42)
    return cv


def jitter(x, sigma=0.3):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=np.mean(x), scale=sigma, size=x.shape)


def get_fraction(x, fraction):
    """ Useful to get fraction of a dataset for ablation experiments """
    return x[:int(fraction * len(x))]


def avg_power_matrix(m):
    """ Drop the time dimension on matrix by averaging power """
    assert len(m.shape) == 2
    y = np.array([avg_power_vector(row) for row in m])
    transformed_y, best_lambda = boxcox(y)
    return transformed_y


def avg_power_vector(u):
    """ Return the avg power of a vector """
    assert len(u.shape) == 1
    powers = [el**2 for el in u]
    avg_power = sum(powers) / len(u)
    return avg_power



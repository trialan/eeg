import numpy as np
from scipy.stats import boxcox


def avg_power_vector(u):
    """ Return the avg power of a vector """
    assert len(u.shape) == 1
    powers = [el**2 for el in u]
    avg_power = sum(powers) / len(u)
    return avg_power


def avg_power_matrix(m):
    """ Drop the time dimension on matrix by averaging power """
    assert len(m.shape) == 2
    y = np.array([avg_power_vector(row) for row in m])
    transformed_y, best_lambda = boxcox(y)
    return transformed_y


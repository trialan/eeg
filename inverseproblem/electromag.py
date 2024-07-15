import numpy as np


class Dipole():
    def __init__(self, position, moment):
        self.position = np.array(position)
        self.moment = np.array(moment)
        assert position.shape == (3,) #a point in 3D space
        assert moment.shape == (3,) #a vector in 3D space


def compute_vector_electric_field(dipole, observation_points, epsilon=1):
    """ E field due to a dipole P:
        E(r) = 3 * ((P . R_hat)R - P) / (4 * pi * epsilon * ||R||^3
        Source: Wikipedia, Electric dipole moment """
    electric_field = np.zeros_like(observation_points)
    for i, point in enumerate(observation_points):
        r = point - dipole.position
        r_norm = np.linalg.norm(r)
        n = r / r_norm if r_norm != 0 else np.zeros_like(r)
        if r_norm != 0:
            E = (1 / (4 * np.pi * epsilon)) * (3 * n * np.dot(n, dipole.moment) - dipole.moment) / (r_norm ** 3)
            electric_field[i] = E
    return electric_field


def convert_to_scalar_field(vector_field):
    scalar_field = np.linalg.norm(vector_field, axis=1)
    return scalar_field

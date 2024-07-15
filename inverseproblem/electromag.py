import numpy as np

from eeg.inverseproblem.spheres import Brain, Scalp


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


def generate_random_scalp_brain_efield(brain, scalp):
    """ Randomly position a random number of dipoles on the brain mesh,
        then compute each dipole E-field on the brain mesh and on the scalp mesh.
        Take vector sum of all dipole E-fields and return the scalar fields """
    brain_vec_efield = np.zeros_like(brain.vertices)
    scalp_vec_efield = np.zeros_like(scalp.vertices)

    random_number_of_dipoles = np.random.randint(len(brain.vertices))
    for i in range(random_number_of_dipoles):
        dipole = generate_random_dipole(brain.vertices)

        dipole_vec_efield_on_brain = compute_vector_electric_field(dipole, brain.vertices)
        dipole_vec_efield_on_scalp = compute_vector_electric_field(dipole, scalp.vertices)

        brain_vec_efield += dipole_vec_efield_on_brain
        scalp_vec_efield += dipole_vec_efield_on_scalp

    return brain_vec_efield, scalp_vec_efield


def generate_random_dipole(vertices):
    """ Generate a random dipole on the brain """
    position = vertices[np.random.choice(vertices.shape[1])]
    moment = np.random.rand(3)
    return Dipole(position, moment)



def convert_to_scalar_field(vector_field):
    scalar_field = np.linalg.norm(vector_field, axis=1)
    return scalar_field

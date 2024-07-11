import numpy as np
from eeg.scalp2brain.spheres import generate_sphere_mesh
from scipy.linalg import lstsq

"""
Let's start with some skeleton code / pseudo-code to flesh out this
problem a little.

ef_brain would be a (n_simplices, ) shaped vector
ef_scalp would also be a (n_simplices, ) shaped vector
(we can construct our brain and scalp meshes to have the same number
of simplices).

Simplifications:
    1. it's reasonable to start with spheres of different radius (like a
       numerical 3-sphere model)
    2. let's just place a single dipole on the brain and compute the field,
       then we'll see about more realistic models.

Phsyiological weirdness:
    1. How do i treat the cerebro-spinal fluid?
    2. How do the E field "bounce off" the skull, seems like a classic
       wave at a boundary problem. Is that it?
    3. Is it realistic to treat the E field on the brain as a collection
       of dipoles?
"""


def compute_electric_field(dipole_position, dipole_moment, observation_points, epsilon=1):
    """ E field due to a dipole P:
        E(r) = 3 * ((P . R_hat)R - P) / (4 * pi * epsilon * ||R||^3 """
    electric_field = np.zeros(len(observation_points))
    dipole_position = np.asarray(dipole_position)
    dipole_moment = np.asarray(dipole_moment)
    for i, point in enumerate(observation_points):
        r = point - dipole_position
        r_norm = np.linalg.norm(r)
        n = r / r_norm if r_norm != 0 else np.zeros_like(r)
        if r_norm != 0:
            E = (1 / (4 * np.pi * epsilon)) * (3 * n * np.dot(n, dipole_moment) - dipole_moment) / (r_norm ** 3)
            electric_field[i] = np.sqrt(np.dot(E, E))
    return electric_field.reshape(-1, 1)


def compute_transformation_matrix(source_efield, target_efield):
    """
    Compute the transformation matrix A such that target_efield ≈ A * source_efield using least-squares fitting.
    """
    A, _, _, _ = lstsq(source_efield, target_efield)
    return A


if __name__ == "__main__":
    brain_radius = 1.
    skull_radius = 2.
    scalp_radius = 3.

    brain_points, brain_faces = generate_sphere_mesh(brain_radius)
    skull_points, skull_faces = generate_sphere_mesh(skull_radius)
    scalp_points, scalp_faces = generate_sphere_mesh(scalp_radius)

    dipole_position = brain_points[0]
    dipole_moment = np.array([1.0, 0.0, 0.0])

    brain_efield = compute_electric_field(dipole_position, dipole_moment, brain_points)
    skull_efield = compute_electric_field(dipole_position, dipole_moment, skull_points)
    scalp_efield = compute_electric_field(dipole_position, dipole_moment, scalp_points)

    A = compute_transformation_matrix(brain_efield, scalp_efield)
    print("Transformation matrix A:")
    print(A)

    # Validate the transformation
    scalp_efield_estimated = brain_efield.dot(A)
    print("Scalp electric field (estimated):")
    print(scalp_efield_estimated[:5])  # Print first 5 values for comparison
    print("Scalp electric field (actual):")
    print(scalp_efield[:5]) 

import numpy as np

from eeg.scalp2brain.spheres import generate_sphere_mesh
from eeg.scalp2brain.electromag import (compute_vector_electric_field, Dipole,
                                        convert_to_scalar_field)
from eeg.scalp2brain.test_utils import (its_a_vector_field, its_a_scalar_field,
                                        we_have_field_at_all_points)

def test_computing_electric_field():
    brain_radius = 1.
    points, _ = generate_sphere_mesh(brain_radius)
    dipole = Dipole(position=points[0],
                    moment=np.array([1.0, 0.0, 0.0]))

    vec_efield = compute_vector_electric_field(dipole, points)
    assert its_a_vector_field(vec_efield)
    assert we_have_field_at_all_points(points, vec_efield)

    scalar_efield = convert_to_scalar_field(vec_efield)
    assert its_a_scalar_field(scalar_efield)



import numpy as np

from eeg.scalp2brain.spheres import Brain, Scalp
from eeg.scalp2brain.data import get_inverse_problem_dataset_np
from eeg.scalp2brain.test_utils import (its_a_scalar_field, its_a_scalar_field,
                                        we_have_field_at_all_points)


def test_generating_numpy_training_datasets():
    brain = Brain(num_points=50)
    scalp = Scalp(num_points=50)
    brain_efields, scalp_efields = get_inverse_problem_dataset_np(brain,
                                                                  scalp,
                                                                  size=3)
    assert brain_efields.shape == (3, 50)
    assert scalp_efields.shape == (3, 50)

    assert np.all([we_have_field_at_all_points(f, brain.points) for f in brain_efields])
    assert np.all([we_have_field_at_all_points(f, brain.points) for f in scalp_efields])

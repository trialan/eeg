import pytest
from eeg.inverseproblem.old.spheres import Brain, Scalp


@pytest.mark.skip()
def test_brain_and_scalp():
    brain = Brain(3)
    assert brain.radius == 3

    scalp = Scalp(7)
    assert scalp.radius == 7

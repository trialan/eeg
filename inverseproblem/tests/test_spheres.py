from eeg.inverseproblem.spheres import Brain, Scalp


def test_brain_and_scalp():
    brain = Brain(3)
    assert brain.radius == 3

    scalp = Scalp(7)
    assert scalp.radius == 7

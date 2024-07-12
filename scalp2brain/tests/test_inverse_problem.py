from eeg.scalp2brain.inverse_problem import solve_inverse_problem
from eeg.scalp2brain.data import get_inverse_problem_dataloaders
from eeg.scalp2brain.spheres import Brain, Scalp


def test_solving_inverse_problem_pipeline():
    """ Check nothing breaks """
    brain = Brain()
    scalp = Scalp()
    train_dataloader, test_dataloader = get_inverse_problem_dataloaders(brain,
                                                              scalp,
                                                              dataset_size=20)
    model = solve_inverse_problem(train_dataloader, test_dataloader)



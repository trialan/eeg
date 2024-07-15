from eeg.inverseproblem.inverse_problem import solve_inverse_problem
from eeg.inverseproblem.data import get_inverse_problem_dataloaders
from eeg.inverseproblem.spheres import Brain, Scalp


if __name__ == '__main__':
    brain = Brain()
    scalp = Scalp()
    train_dataloader, test_dataloader = get_inverse_problem_dataloaders(brain,
                                                              scalp,
                                                              dataset_size=20)
    model = solve_inverse_problem(train_dataloader, test_dataloader)

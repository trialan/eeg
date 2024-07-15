import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from eeg.inverseproblem.spheres import Brain, Scalp
from eeg.inverseproblem.electromag import (Dipole, compute_vector_electric_field,
                                        convert_to_scalar_field)


def get_inverse_problem_dataloaders(brain, scalp, dataset_size=1000, test_split_ratio=0.2, batch_size=32):
    """ Load inverse problem dataset into PyTorch DataLoader objects """
    brain_efields, scalp_efields = get_inverse_problem_dataset_np(brain, scalp,
                                                                  size=dataset_size)

    brain_efields_tensor = torch.tensor(brain_efields, dtype=torch.float32)
    scalp_efields_tensor = torch.tensor(scalp_efields, dtype=torch.float32)

    dataset = TensorDataset(scalp_efields_tensor, brain_efields_tensor)

    test_size = int(test_split_ratio * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_inverse_problem_dataset_np(brain, scalp, size):
    """ Generate scalar E-fields on brain and scalp in NumPy format """
    brain_efields = []
    scalp_efields = []
    for training_example in tqdm(range(size), desc="Computing E-fields"):
        brain_field, scalp_field = generate_random_scalp_brain_efield(brain,
                                                                      scalp)
        brain_efields.append(brain_field)
        scalp_efields.append(scalp_field)
    return np.array(brain_efields), np.array(scalp_efields)


def generate_random_scalp_brain_efield(brain, scalp):
    """ Randomly position a random number of dipoles on the brain mesh,
        then compute each dipole E-field on the brain mesh and on the scalp mesh.
        Take vector sum of all dipole E-fields and return the scalar fields """
    brain_vec_efield = np.zeros_like(brain.points)
    scalp_vec_efield = np.zeros_like(scalp.points)

    random_number_of_dipoles = np.random.randint(len(brain.points))
    for i in range(random_number_of_dipoles):
        dipole = generate_random_dipole(brain.points)

        dipole_vec_efield_on_brain = compute_vector_electric_field(dipole, brain.points)
        dipole_vec_efield_on_scalp = compute_vector_electric_field(dipole, scalp.points)

        brain_vec_efield += dipole_vec_efield_on_brain
        scalp_vec_efield += dipole_vec_efield_on_scalp

    scalar_brain_efield = convert_to_scalar_field(brain_vec_efield)
    scalar_scalp_efield = convert_to_scalar_field(scalp_vec_efield)
    return scalar_brain_efield, scalar_scalp_efield


def generate_random_dipole(brain_points):
    """ Generate a random dipole on the brain """
    position = brain_points[np.random.choice(brain_points.shape[1])]
    moment = np.random.rand(3)
    return Dipole(position, moment)


if __name__ == "__main__":
    brain = Brain()
    scalp = Scalp()
    brain_efields, scalp_efields = generate_inverse_problem_dataset(brain, scalp,
                                                                    size=5)



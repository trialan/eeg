import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from eeg.utils import read_pickle
from eeg.inverseproblem.simultaneous_eeg_fmri.eeg_data import get_eeg_data
from eeg.inverseproblem.simultaneous_eeg_fmri.fmri_data import get_fmri_data
from eeg.inverseproblem.simultaneous_eeg_fmri.cyclic_cnn import (EEGEncoder,
                                                                 EEGDecoder,
                                                                 fMRIEncoder,
                                                                 fMRIDecoder)


root_dir = "/root/DS116/"
# root_dir = "/Users/thomasrialan/Documents/code/DS116/"


if __name__ == "__main__":
    X_eeg, y_eeg = get_eeg_data(root_dir)
    # X_fmri, y_fmri = get_fmri_data()
    y_fmri = read_pickle("fmri_y.pkl")
    X_fmri = read_pickle("fmri_X.pkl")

    assert len(X_eeg) == len(X_fmri)
    assert np.array_equal(y_eeg, y_fmri)

    X_eeg = torch.tensor(X_eeg).float()
    X_fmri = torch.tensor(X_fmri).float()

    # DataLoader
    batch_size = 32
    dataset = TensorDataset(X_eeg, X_fmri)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model Initialization
    eeg_encoder = EEGEncoder()
    fmri_encoder = fMRIEncoder()
    eeg_decoder = EEGDecoder()
    fmri_decoder = fMRIDecoder()
    eeg_to_fmri_decoder = fMRIDecoder()  # Decoder for mapping EEG to fMRI
    fmri_to_eeg_decoder = EEGDecoder()  # Decoder for mapping fMRI to EEG

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        list(eeg_encoder.parameters()) + list(fmri_encoder.parameters()) +
        list(eeg_decoder.parameters()) + list(fmri_decoder.parameters()) +
        list(eeg_to_fmri_decoder.parameters()) + list(fmri_to_eeg_decoder.parameters()), 
        lr=0.001
    )

    # Training Loop
    num_epochs = 100
    for epoch in range(num_epochs):
        eeg_encoder.train()
        fmri_encoder.train()
        eeg_decoder.train()
        fmri_decoder.train()
        eeg_to_fmri_decoder.train()
        fmri_to_eeg_decoder.train()

        total_loss = 0
        for batch_idx, (eeg_batch, fmri_batch) in enumerate(dataloader):
            # EEG to EEG
            eeg_encoded = eeg_encoder(eeg_batch.unsqueeze(1))
            eeg_decoded = eeg_decoder(eeg_encoded)
            loss_eeg = criterion(eeg_decoded, eeg_batch.unsqueeze(1))

            # fMRI to fMRI
            fmri_encoded = fmri_encoder(fmri_batch.unsqueeze(1))
            fmri_decoded = fmri_decoder(fmri_encoded)
            loss_fmri = criterion(fmri_decoded, fmri_batch.unsqueeze(1))

            # EEG to fMRI
            eeg_to_fmri = eeg_to_fmri_decoder(eeg_encoded)
            loss_eeg_to_fmri = criterion(eeg_to_fmri, fmri_batch.unsqueeze(1))

            # fMRI to EEG
            fmri_to_eeg = fmri_to_eeg_decoder(fmri_encoded)
            loss_fmri_to_eeg = criterion(fmri_to_eeg, eeg_batch.unsqueeze(1))

            # Total loss
            loss = loss_eeg + loss_fmri + loss_eeg_to_fmri + loss_fmri_to_eeg

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()


        torch.save({
            'eeg_encoder': eeg_encoder.state_dict(),
            'fmri_encoder': fmri_encoder.state_dict(),
            'eeg_decoder': eeg_decoder.state_dict(),
            'fmri_decoder': fmri_decoder.state_dict(),
            'eeg_to_fmri_decoder': eeg_to_fmri_decoder.state_dict(),
            'fmri_to_eeg_decoder': fmri_to_eeg_decoder.state_dict(),
        }, 'cyclic_cnn.pth')
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}')


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import optuna

from eeg.utils import read_pickle
from eeg.inverseproblem.simultaneous_eeg_fmri.eeg_data import get_eeg_data
from eeg.inverseproblem.simultaneous_eeg_fmri.fmri_data import get_fmri_data
from eeg.inverseproblem.simultaneous_eeg_fmri.cyclic_cnn import (
    EEGEncoder,
    EEGDecoder,
    fMRIEncoder,
    fMRIDecoder,
)


root_dir = "/root/DS116/"


def downsample_eeg(eeg_data, original_rate=500, target_rate=2.86):
    downsample_factor = int(original_rate / target_rate)
    return eeg_data[:, :, ::downsample_factor]


def train_epoch(
    dataloader,
    optimizer,
    criterion,
    eeg_encoder,
    fmri_encoder,
    eeg_decoder,
    fmri_decoder,
    eeg_to_fmri_decoder,
    fmri_to_eeg_decoder,
):
    """Sort of bad practice to have so many args. Fix later"""
    eeg_encoder.train()
    fmri_encoder.train()
    eeg_decoder.train()
    fmri_decoder.train()
    eeg_to_fmri_decoder.train()
    fmri_to_eeg_decoder.train()

    total_loss = 0
    for batch_idx, (eeg_batch, fmri_batch) in tqdm(enumerate(dataloader)):
        # EEG to EEG
        eeg_encoded = eeg_encoder(eeg_batch)
        eeg_decoded = eeg_decoder(eeg_encoded)
        loss_eeg = criterion(eeg_decoded, eeg_batch)

        # fMRI to fMRI
        fmri_encoded = fmri_encoder(fmri_batch)
        fmri_decoded = fmri_decoder(fmri_encoded)
        loss_fmri = criterion(fmri_decoded, fmri_batch)

        # EEG to fMRI
        eeg_to_fmri = eeg_to_fmri_decoder(eeg_encoded.view(eeg_encoded.size(0), -1))
        loss_eeg_to_fmri = criterion(eeg_to_fmri.squeeze(1), fmri_batch)

        # fMRI to EEG
        fmri_to_eeg = fmri_to_eeg_decoder(fmri_encoded.view(fmri_encoded.size(0), -1))
        loss_fmri_to_eeg = criterion(fmri_to_eeg, eeg_batch)

        # Total loss
        loss = loss_eeg + loss_fmri + loss_eeg_to_fmri + loss_fmri_to_eeg

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(
    dataloader,
    criterion,
    eeg_encoder,
    fmri_encoder,
    eeg_decoder,
    fmri_decoder,
    eeg_to_fmri_decoder,
    fmri_to_eeg_decoder,
):
    eeg_encoder.eval()
    fmri_encoder.eval()
    eeg_decoder.eval()
    fmri_decoder.eval()
    eeg_to_fmri_decoder.eval()
    fmri_to_eeg_decoder.eval()

    total_loss = 0
    for batch_idx, (eeg_batch, fmri_batch) in tqdm(enumerate(dataloader)):
        # EEG to EEG
        eeg_encoded = eeg_encoder(eeg_batch)
        eeg_decoded = eeg_decoder(eeg_encoded)
        loss_eeg = criterion(eeg_decoded, eeg_batch)

        # fMRI to fMRI
        fmri_encoded = fmri_encoder(fmri_batch)
        fmri_decoded = fmri_decoder(fmri_encoded)
        loss_fmri = criterion(fmri_decoded, fmri_batch)

        # EEG to fMRI
        eeg_to_fmri = eeg_to_fmri_decoder(eeg_encoded.view(eeg_encoded.size(0), -1))
        loss_eeg_to_fmri = criterion(eeg_to_fmri.squeeze(1), fmri_batch)

        # fMRI to EEG
        fmri_to_eeg = fmri_to_eeg_decoder(fmri_encoded.view(fmri_encoded.size(0), -1))
        loss_fmri_to_eeg = criterion(fmri_to_eeg, eeg_batch)

        # Total loss
        loss = loss_eeg + loss_fmri + loss_eeg_to_fmri + loss_fmri_to_eeg

        total_loss += loss.item()
    return total_loss / len(dataloader)




if __name__ == "__main__":
    X_eeg, y_eeg = get_eeg_data(root_dir)
    X_fmri, y_fmri = get_fmri_data()
    from eeg.utils import write_pickle
    y_fmri = write_pickle(y_fmri, "fmri_y.pkl")
    x_fmri = write_pickle(X_fmri, "fmri_X.pkl")
    #y_fmri = read_pickle("fmri_y.pkl")
    #x_fmri = read_pickle("fmri_x.pkl")

    assert len(X_eeg) == len(X_fmri)
    assert np.array_equal(y_eeg, y_fmri)

    X_eeg = torch.tensor(X_eeg).float()
    X_fmri = torch.tensor(X_fmri).float()

    X_eeg_downsampled = downsample_eeg(X_eeg)
    eeg_time_dim = X_eeg_downsampled.shape[2]


    batch_size = 32
    lr = 0.001
    weight_decay = 0.0
    dropout_rate = 0.0

    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        X_eeg_downsampled, X_fmri, batch_size
    )

    # Model Initialization
    eeg_encoder = EEGEncoder(dropout_rate=dropout_rate)
    fmri_encoder = fMRIEncoder(dropout_rate=dropout_rate)
    eeg_decoder = EEGDecoder(dropout_rate=dropout_rate)
    fmri_decoder = fMRIDecoder(dropout_rate=dropout_rate)

    fmri_to_eeg_decoder = nn.Sequential(
        nn.Flatten(),
        nn.Linear(32 * 32, 34 * eeg_time_dim),
        nn.Dropout(dropout_rate),
        nn.Unflatten(1, (34, eeg_time_dim)),
    )

    eeg_to_fmri_decoder = nn.Sequential(
        nn.Flatten(),
        nn.Linear(32 * 32, 64 * 64 * 32),
        nn.Dropout(dropout_rate),
        nn.Unflatten(1, (1, 64, 64, 32)),
    )

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        list(eeg_encoder.parameters())
        + list(fmri_encoder.parameters())
        + list(eeg_decoder.parameters())
        + list(fmri_decoder.parameters())
        + list(eeg_to_fmri_decoder.parameters())
        + list(fmri_to_eeg_decoder.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Training loop
    num_epochs = 100  # Maximum number of epochs
    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss = train_epoch(
            train_dataloader,
            optimizer,
            criterion,
            eeg_encoder,
            fmri_encoder,
            eeg_decoder,
            fmri_decoder,
            eeg_to_fmri_decoder,
            fmri_to_eeg_decoder,
        )

        val_loss = validate(
            val_dataloader,
            criterion,
            eeg_encoder,
            fmri_encoder,
            eeg_decoder,
            fmri_decoder,
            eeg_to_fmri_decoder,
            fmri_to_eeg_decoder,
        )

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save(
                {
                    "eeg_encoder": eeg_encoder.state_dict(),
                    "fmri_encoder": fmri_encoder.state_dict(),
                    "eeg_decoder": eeg_decoder.state_dict(),
                    "fmri_decoder": fmri_decoder.state_dict(),
                    "eeg_to_fmri_decoder": eeg_to_fmri_decoder.state_dict(),
                    "fmri_to_eeg_decoder": fmri_to_eeg_decoder.state_dict(),
                },
                "cyclic_cnn.pth",
            )

        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

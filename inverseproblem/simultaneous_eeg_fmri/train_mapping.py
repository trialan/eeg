import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
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
# root_dir = "/Users/thomasrialan/Documents/code/DS116/"


def downsample_eeg(eeg_data, original_rate=500, target_rate=2.86):
    downsample_factor = int(original_rate / target_rate)
    return eeg_data[:, :, ::downsample_factor]


def train_epoch(dataloader, optimizer, criterion):
    """Sort of bad practice to "assume" the models are in memory, even though
    they certainly are, come back and fix this eventually"""
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


def validate(dataloader, criterion):
    """Sort of bad practice to "assume" the models are in memory, even though
    they certainly are, come back and fix this eventually"""
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


def create_dataloaders(
    X_eeg,
    X_fmri,
    batch_size=96,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    random_state=42,
):
    # First split: separate test set
    X_eeg_trainval, X_eeg_test, X_fmri_trainval, X_fmri_test = train_test_split(
        X_eeg, X_fmri, test_size=test_size, random_state=random_state
    )

    # Second split: separate train and validation sets
    val_size_adjusted = val_size / (train_size + val_size)
    X_eeg_train, X_eeg_val, X_fmri_train, X_fmri_val = train_test_split(
        X_eeg_trainval,
        X_fmri_trainval,
        test_size=val_size_adjusted,
        random_state=random_state,
    )

    # Create datasets
    train_dataset = TensorDataset(X_eeg_train, X_fmri_train)
    val_dataset = TensorDataset(X_eeg_val, X_fmri_val)
    test_dataset = TensorDataset(X_eeg_test, X_fmri_test)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

def objective(trial):
    # Define hyperparameters to optimize
    batch_size = trial.suggest_int('batch_size', 32, 256)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)

    # Create dataloaders
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(X_eeg, X_fmri, batch_size=batch_size)

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
        nn.Unflatten(1, (1, 64, 64, 32))
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
        weight_decay=weight_decay
    )

    # Training loop
    num_epochs = 100  # Maximum number of epochs
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss = train_epoch(train_dataloader, optimizer, criterion)
        val_loss = validate(val_dataloader, criterion)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # Report intermediate objective value
        trial.report(val_loss, epoch)

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_loss


if __name__ == "__main__":
    X_eeg, y_eeg = get_eeg_data(root_dir)
    # X_fmri, y_fmri = get_fmri_data()
    y_fmri = read_pickle("fmri_y.pkl")
    X_fmri = read_pickle("fmri_X.pkl")

    assert len(X_eeg) == len(X_fmri)
    assert np.array_equal(y_eeg, y_fmri)

    X_eeg = torch.tensor(X_eeg).float()
    X_fmri = torch.tensor(X_fmri).float()

    X_eeg = downsample_eeg(X_eeg)
    eeg_time_dim = X_eeg.shape[2]

    # Create an Optuna study object
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=300, timeout=3600*8)  # Run for 8 hours

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save the best model
    best_params = study.best_params

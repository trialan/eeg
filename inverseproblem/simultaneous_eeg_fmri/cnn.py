import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from eeg.inverseproblem.simultaneous_eeg_fmri.data import (
    balance_and_shuffle,
    seed_everything,
)
from eeg.utils import get_cv
from torch.utils.data import TensorDataset, DataLoader


seed_everything()


class FMRI_CNN(nn.Module):
    def __init__(self):
        super(FMRI_CNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8 * 4, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 8 * 8 * 4)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


def train(model, X, y, epochs=50, batch_size=32, seed=42, val_size=0.2):
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Convert X to the correct shape if it's not already
    if X.shape[1] != 32:
        X = np.transpose(X, (0, 3, 1, 2))  # (4875, 64, 64, 32) -> (4875, 32, 64, 64)

    # Perform train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=seed
    )

    # Convert X and y to PyTorch tensors
    X_train = torch.from_numpy(X_train).float().unsqueeze(1)  # Add channel dimension
    y_train = torch.from_numpy(y_train).float().view(-1, 1)
    X_val = torch.from_numpy(X_val).float().unsqueeze(1)
    y_val = torch.from_numpy(y_val).float().view(-1, 1)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        for i in tqdm(range(0, len(X_train), batch_size)):
            batch_X = X_train[i : i + batch_size]
            batch_y = y_train[i : i + batch_size]
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct_predictions += (predicted == batch_y).sum().item()
            total_predictions += batch_y.size(0)

        train_loss = total_loss / (len(X_train) // batch_size)
        train_accuracy = correct_predictions / total_predictions

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            val_predicted = (val_outputs > 0.5).float()
            val_accuracy = (val_predicted == y_val).float().mean().item()

        print(
            f"Epoch {epoch+1}, "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}"
        )

    return model


def train_cv(model_class, X, y, cv, epochs=15, batch_size=32, seed=42, val_size=0.2):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if X.shape[1] != 32:
        X = np.transpose(X, (0, 3, 1, 2))

    scores = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        print(f"Fold {fold}")
        if fold == 1:
            print("Top score was 88%")
            continue
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train = torch.from_numpy(X_train).float().unsqueeze(1)
        y_train = torch.from_numpy(y_train).float().view(-1, 1)
        X_val = torch.from_numpy(X_val).float().unsqueeze(1)
        y_val = torch.from_numpy(y_val).float().view(-1, 1)

        model = model_class()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        best_val_loss = float("inf")
        best_val_acc = 0.0
        no_improve_count = 0

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0

            for i in tqdm(range(0, len(X_train), batch_size)):
                batch_X = X_train[i : i + batch_size]
                batch_y = y_train[i : i + batch_size]
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct_predictions += (predicted == batch_y).sum().item()
                total_predictions += batch_y.size(0)

            train_loss = total_loss / (len(X_train) // batch_size)
            train_accuracy = correct_predictions / total_predictions

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
                val_predicted = (val_outputs > 0.5).float()
                val_accuracy = (val_predicted == y_val).float().mean().item()

            print(
                f"Epoch {epoch+1}, "
                f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}, "
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_accuracy
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= 2:
                print(f"Early stopping at epoch {epoch+1}")
                break

        scores.append(best_val_acc)

    mean_score = np.mean(scores)
    se = np.std(scores) / len(scores)
    return mean_score, se


if __name__ == "__main__":
    from eeg.utils import read_pickle, write_pickle
    from eeg.inverseproblem.simultaneous_eeg_fmri._fmri_data import get_raw_fmri_data

    #X, y = get_raw_fmri_data("/root/DS116/")
    #write_pickle(X, "fmri_X.pkl")
    #write_pickle(y, "fmri_y.pkl")
    model = FMRI_CNN()
    criterion = nn.BCELoss()
    X = read_pickle("fmri_X.pkl")
    y = read_pickle("fmri_y.pkl")
    Xb, yb = balance_and_shuffle(X, y)
    cv = get_cv()
    train_cv(FMRI_CNN, Xb, yb, cv)
    # train(model, Xb, yb)

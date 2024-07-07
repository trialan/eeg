import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import ShuffleSplit, cross_val_score, train_test_split
import pyriemann
from eeg.laplacian import (
    get_electrode_coordinates,
    get_256D_eigenvectors,
    create_triangular_dmesh,
    ED,
)
from eeg.ml import results
from eeg.data import get_formatted_data, get_data
from eeg.utils import avg_power_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TimeSeriesCNN(nn.Module):
    def __init__(self):
        super(TimeSeriesCNN, self).__init__()
        self.layer1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=(3, 3), padding=(1, 1)
        )
        self.layer2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=(3, 3), padding=(1, 1)
        )
        self.layer3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1)
        )
        self.layer4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1)
        )
        self.layer5 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1)
        )
        self.layer6 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1)
        )
        self.layer7 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 2)  # Binary classification

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = F.relu(self.layer6(x))
        x = F.relu(self.layer7(x))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=3, eval_interval=100):
    losses = []
    val_scores = []
    train_scores = []
    total_steps = 0

    for epoch in tqdm(range(num_epochs), desc=f"training {num_epochs} epochs"):
        model.train()  # Ensure model is in training mode
        running_loss = 0.0
        for step, (inputs, labels) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total_steps += 1

            if (step + 1) % eval_interval == 0:
                step_loss = running_loss / total_steps
                val_score = evaluate_model_CE(model, val_loader)
                #train_score = evaluate_model(model, train_loader)

                losses.append(step_loss)
                #train_scores.append(train_score)
                val_scores.append(val_score)

                running_loss = 0.0

    return model, np.array(losses), np.array(val_scores)


def makeplot(val_scores, losses):
    plt.figure(figsize=(12, 5))
    plt.title("TimeSeriesCNN loss curves on raw dataset")

    plt.subplot(1, 2, 1)
    plt.plot(range(len(val_scores)), losses,
             color='g', label="train loss",
             linestyle="--", marker="o")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(len(val_scores)), val_scores,
                color='g', label="validation loss",
                linestyle="--", marker="o")
    plt.legend()

    plt.show()


def evaluate_model_CE(model, val_loader):
    """ A good metric for binary classification, should be less noisy than
        accuracy I think """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    model.train()
    return total_loss / total_samples


def evaluate_model_accuracy(model, val_loader):
    """ Interpretable model score  """
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # No need to track gradients
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    model.train()
    return total_correct / total_samples


def get_fraction(x, fraction):
    return x[:int(fraction * len(x))]


def get_dataloaders(X_train, X_val, y_train, y_val):
        fraction = 1.0
        frac_X_train = get_fraction(X_train, fraction)
        frac_y_train = get_fraction(y_train, fraction)

        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(frac_X_train, dtype=torch.float32),
            torch.tensor(frac_y_train, dtype=torch.long),
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=3, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=3, shuffle=False
        )
        return train_loader, val_loader


if __name__ == "__main__":
    X, y = get_data()
    X = X[:, np.newaxis, :, :]  # Add channel dimension
    cv = ShuffleSplit(5, test_size=0.2, random_state=42)

    models = []
    split_train_losses = []
    split_val_losses = []
    for train_index, val_index in tqdm(cv.split(X), desc="Split learning"):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        train_loader, val_loader = get_dataloaders(X_train, X_val,
                                                   y_train, y_val)

        device = torch.device("mps")
        model = TimeSeriesCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005)

        model, train_losses, val_losses = train_model(model, train_loader,
                                         val_loader, criterion,
                                         optimizer, num_epochs=1)

        split_train_losses.append(train_losses)
        split_val_losses.append(val_losses)
        models.append(model)

    #0.2 because there are 5 splits, we want the averages
    makeplot(0.2*sum(split_val_losses),
             0.2*sum(split_train_losses))







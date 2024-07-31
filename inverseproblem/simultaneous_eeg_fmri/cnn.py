import torch
import torch.nn as nn
import torch.optim as optim


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


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def train(model, X, y, epochs=50, batch_size=32):
    # Convert X to the correct shape if it's not already
    if X.shape[1] != 32:
        X = np.transpose(X, (0, 3, 1, 2))  # (4875, 64, 64, 32) -> (4875, 32, 64, 64)

    # Convert X and y to PyTorch tensors
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float().view(-1, 1)

    # Add channel dimension: (4875, 32, 64, 64) -> (4875, 1, 32, 64, 64)
    X = X.unsqueeze(1)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        for i in range(0, len(X), batch_size):
            batch_X = X[i : i + batch_size]
            batch_y = y[i : i + batch_size]
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # Calculate accuracy
            predicted = (outputs > 0.5).float()
            correct_predictions += (predicted == batch_y).sum().item()
            total_predictions += batch_y.size(0)
        epoch_loss = total_loss / (len(X) // batch_size)
        epoch_accuracy = correct_predictions / total_predictions
        print(
            f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}"
        )


model = FMRI_CNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())
train(model, X_resampled, y_resampled)

import torch
import torch.nn as nn
import torch.optim as optim

from eeg.scalp2brain.data import get_inverse_problem_dataloaders
from eeg.scalp2brain.spheres import Brain, Scalp


def solve_inverse_problem(train_dataloader, test_dataloader):
    model = MLP()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

        test_loss /= len(test_dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128, 100)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


if __name__ == '__main__':
    brain = Brain()
    scalp = Scalp()
    train_dataloader, test_dataloader = get_inverse_problem_dataloaders(brain,
                                                              scalp,
                                                              dataset_size=100)
    model = solve_inverse_problem(train_dataloader, test_dataloader)



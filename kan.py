import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from efficient_kan import KAN
from eeg.data import get_data
from eeg.utils import get_cv, avg_power_matrix
import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label


def get_dataloaders(dataset, train_idx, test_idx):
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, test_idx)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    test_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    return train_loader, test_loader


if __name__ == '__main__':
    X, y = get_data()
    X_ap = np.array([avg_power_matrix(m) for m in X])
    dataset = CustomDataset(X_ap, y)
    cv = get_cv()

    accuracies = []
    for train_idx, test_idx in cv.split(X):
        train_loader, test_loader = get_dataloaders(dataset, train_idx, test_idx)

        #N.B i made up these constants so that init predictions
        #are roughly 0.5, default base/noise/spline, are like 0.1
        #i have no idea if this is a dumb idea, could be!

        model = KAN([64, 256, 512, 512, 256, 128, 1],
                    scale_base=10,
                    scale_noise=1e3/(1.68*2),
                    scale_spline=10)
        device = torch.device("mps")
        model.to(device)

        optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

        # Define loss
        criterion = nn.BCEWithLogitsLoss()

        in_fold_batch_test_accuracies = []
        for epoch in tqdm(list(range(5))):
            train_loss = 0
            batch_test_accuracies = []

            for batch, (inputs, labels) in tqdm(enumerate(train_loader), desc=f"Training"):
                model.train()
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                output = model(inputs)
                loss = criterion(output.reshape(-1), labels.float().to(device))
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Validation
                model.eval()
                test_loss = 0
                test_accuracy = 0
                with torch.no_grad():
                    for samples, labels in test_loader:
                        samples = samples.to(device)
                        labels = labels.to(device)
                        output = model(samples)
                        test_loss += criterion(output.reshape(-1), labels.float()).item()
                        predicted = (output > 0.5).float()
                        test_accuracy += (predicted == labels).float().mean().item()
                test_accuracy /= len(test_loader)
                batch_test_accuracies.append(test_accuracy)

            best_test_accuracy = np.max(batch_test_accuracies)
            print(f"Epoch best acc.: {best_test_accuracy}")
            in_fold_batch_test_accuracies.append(best_test_accuracy)
        accuracies.append(np.mean(in_fold_batch_test_accuracies))

    print(f"Mean score: {np.mean(accuracies)}")



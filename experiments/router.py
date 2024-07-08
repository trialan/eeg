import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F

from eeg.data import get_data
from eeg.utils import results, get_cv
from eeg.ml import (assemble_classifer_PCACSPLDA,
                    assemble_classifer_CSPLDA,
                    PCA)
from eeg.experiments.ensemble import EDFgMDM
from eeg.laplacian import compute_scalp_eigenvectors_and_values


def predict_with_router(row):
    chosen_classifier = get_best_classifier(row)
    if chosen_classifier == 0:
        return edf.predict(np.array([row]))[0]
    elif chosen_classifier == 1:
        return pcl.predict(np.array([row]))[0]
    else:
        return cl.predict(np.array([row]))[0]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class RouterNetwork(nn.Module):
    def __init__(self):
        super(RouterNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 128, 2, stride=2)
        self.layer2 = self._make_layer(128, 256, 2, stride=2)
        self.fc1 = nn.Linear(10240, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 3)
        self.dropout = nn.Dropout(0.5)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = F.avg_pool2d(x, kernel_size=4)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def get_best_classifier(row):
    input_tensor = torch.tensor(row, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        router_output = router_net(input_tensor)
        chosen_classifier = torch.argmax(router_output).item()
    return chosen_classifier


if __name__ == '__main__':
    X, y = get_data()
    cv = get_cv()
    eigenvectors, eigenvals = compute_scalp_eigenvectors_and_values()

    X_temp, X_test, y_temp, y_test = train_test_split(X, y,
                                                      test_size=0.15,
                                                      random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp,
                                                      test_size=0.1765,
                                                      random_state=42)

    print(f"Train X size: {X_train.shape}")
    print(f"Train y size: {y_train.shape}")

    print(f"Val X size: {X_val.shape}")
    print(f"Val y size: {y_val.shape}")

    print(f"Test X size: {X_test.shape}")
    print(f"Test y size: {y_test.shape}")

    edf = EDFgMDM(n_components=24, eigenvectors=eigenvectors)
    edf.fit(X_train, y_train)
    edf_y_pred = edf.predict(X_test)
    score = accuracy_score(edf_y_pred, y_test)
    print(f"Laplacian + FgMDM score: {score}") #0.6425

    pcl = assemble_classifer_PCACSPLDA(n_components=30)
    pcl.fit(X_train, y_train)
    pcl_y_pred = pcl.predict(X_test)
    score = accuracy_score(pcl_y_pred, y_test)
    print(f"PCA+CSP+LDA score: {score}") #0.6285

    cl = assemble_classifer_CSPLDA(n_components=10)
    cl.fit(X_train, y_train)
    cl_y_pred = cl.predict(X_test)
    score = accuracy_score(cl_y_pred, y_test)
    print(f"CSP+LDA score: {score}") #0.6327

    #Make the Venn diagram
    good_edf_subjects = np.where(edf_y_pred == y_test)[0]
    good_pcl_subjects = np.where(pcl_y_pred == y_test)[0]
    good_cl_subjects = np.where(cl_y_pred == y_test)[0]

    edf_set = set(good_edf_subjects)
    pcl_set = set(good_pcl_subjects)
    cl_set = set(good_cl_subjects)

    plt.figure(figsize=(8, 8))
    venn = venn3([edf_set, pcl_set, cl_set],
                 ('Laplacian + FgMDM (24 components)',
                  'PCA+CSP+LDA (30 components)',
                  'CSP+LDA (10 components)'))

    #Make the routing dataset using the validation dataset
    edf_preds = edf.predict_proba(X_val)
    pcl_preds = pcl.predict_proba(X_val)
    cl_preds = cl.predict_proba(X_val)

    X_router = X_val
    y_router = np.zeros(len(X_val))

    for i in range(len(X_val)):
        prob_1 = edf_preds[i][1]
        prob_2 = pcl_preds[i][1]
        prob_3 = cl_preds[i][1]

        diff_1 = abs(prob_1 - y_val[i])
        diff_2 = abs(prob_2 - y_val[i])
        diff_3 = abs(prob_3 - y_val[i])

        best_classifier = np.argmin([diff_1, diff_2, diff_3])
        y_router[i] = best_classifier

    #Put it in PyTorch-ready format
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cpu")
    generator = torch.Generator().manual_seed(42)

    X_router_tensor = torch.tensor(X_router, dtype=torch.float32,
                                   device=device)
    y_router_tensor = torch.tensor(y_router, dtype=torch.long,
                                   device=device)

    # Split the router dataset into training and validation sets
    dataset = TensorDataset(X_router_tensor, y_router_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset,
                                              [train_size, val_size])


    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    #Train the Router Network
    router_net = RouterNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(router_net.parameters(), lr=0.05)

    num_epochs = 15
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training
        router_net.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.unsqueeze(1).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = router_net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        router_net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.unsqueeze(1).to(device)
                labels = labels.to(device)
                outputs = router_net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}')

    # Plotting the learning curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.show()

    router_net.to("cpu")
    y_pred = []
    for row in X_test:
        out = predict_with_router(row)
        y_pred.append(out)
    score = accuracy_score(y_pred, y_test)
    print(f"Router score: {score}")

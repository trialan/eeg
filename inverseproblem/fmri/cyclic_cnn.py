import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class EEGfMRIDataset(Dataset):
    def __init__(self, eeg_data, fmri_data):
        self.eeg_data = eeg_data
        self.fmri_data = fmri_data

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        eeg = self.eeg_data[idx]
        fmri = self.fmri_data[idx]
        return eeg, fmri


class CyclicCNN(nn.Module):
    def __init__(self):
        super(CyclicCNN, self).__init__()
        # Encoder: EEG to latent space
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=49, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        # Decoder: Latent space to fMRI
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=64, out_channels=49, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=49, out_channels=1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# Initialize data, model, loss function, and optimizer
eeg_data = ...  # Load your EEG data here
fmri_data = ...  # Load your fMRI data here

dataset = EEGfMRIDataset(eeg_data, fmri_data)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = CyclicCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    for i, (eeg, fmri) in enumerate(dataloader):
        eeg, fmri = eeg.float(), fmri.float()

        # Forward pass
        outputs = model(eeg)
        loss = criterion(outputs, fmri)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model checkpoint
torch.save(model.state_dict(), 'cyclic_cnn.pth')

# Evaluation
# You can define evaluation metrics and further analyze the model's predictions

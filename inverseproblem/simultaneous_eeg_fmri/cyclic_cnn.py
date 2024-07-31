import torch
import torch.nn as nn

class CyclicCNN(nn.Module):
    def __init__(self):
        super(CyclicCNN, self).__init__()
        
        # EEG decoder (spatial and temporal)
        self.eeg_decoder = nn.Sequential(
            nn.Conv2d(in_channels=34, out_channels=64, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU()
        )
        
        # EEG encoder (spatial and temporal)
        self.eeg_encoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=34, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU()
        )
        
        # fMRI decoder (spatial)
        self.fmri_decoder = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # fMRI encoder (spatial)
        self.fmri_encoder = nn.Sequential(
            nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=16, out_channels=1, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def eeg_to_fmri(self, eeg):
        # eeg shape: (batch, 34, 1, 1001)
        latent = self.eeg_decoder(eeg)
        # Reshape latent to match fMRI dimensions
        latent = latent.view(latent.size(0), 32, 4, 4, 4)
        return self.fmri_encoder(latent)

    def fmri_to_eeg(self, fmri):
        # fmri shape: (batch, 64, 64, 32)
        latent = self.fmri_decoder(fmri.unsqueeze(1))
        # Reshape latent to match EEG dimensions
        latent = latent.view(latent.size(0), 128, 1, -1)
        return self.eeg_encoder(latent)

    def forward(self, eeg, fmri):
        eeg_to_fmri = self.eeg_to_fmri(eeg)
        fmri_to_eeg = self.fmri_to_eeg(fmri)
        eeg_reconstructed = self.fmri_to_eeg(eeg_to_fmri)
        fmri_reconstructed = self.eeg_to_fmri(fmri_to_eeg)
        return eeg_to_fmri, fmri_to_eeg, eeg_reconstructed, fmri_reconstructed

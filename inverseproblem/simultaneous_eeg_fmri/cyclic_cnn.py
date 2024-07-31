import torch
import torch.nn as nn


class CyclicCNN(nn.Module):
    def __init__(self):
        super(CyclicCNN, self).__init__()

        # EEG decoder (spatial only)
        self.eeg_decoder = nn.Sequential(
            nn.Conv3d(in_channels=34, out_channels=64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
        )

        # EEG encoder (spatial only)
        self.eeg_encoder = nn.Sequential(
            nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=64, out_channels=34, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
        )

        # fMRI decoder (temporal only)
        self.fmri_decoder = nn.Sequential(
            nn.Conv1d(in_channels=64 * 64 * 32, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # fMRI encoder (temporal only)
        self.fmri_encoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=256, out_channels=64 * 64 * 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def eeg_to_fmri(self, eeg):
        latent = self.eeg_decoder(eeg)
        return self.fmri_encoder(latent.view(latent.size(0), -1, 1))

    def fmri_to_eeg(self, fmri):
        latent = self.fmri_decoder(fmri.view(fmri.size(0), -1, 1))
        return self.eeg_encoder(latent.view(latent.size(0), 128, 1, 1, 1))

    def forward(self, eeg, fmri):
        eeg_to_fmri = self.eeg_to_fmri(eeg)
        fmri_to_eeg = self.fmri_to_eeg(fmri)
        eeg_reconstructed = self.fmri_to_eeg(eeg_to_fmri)
        fmri_reconstructed = self.eeg_to_fmri(fmri_to_eeg)
        return eeg_to_fmri, fmri_to_eeg, eeg_reconstructed, fmri_reconstructed



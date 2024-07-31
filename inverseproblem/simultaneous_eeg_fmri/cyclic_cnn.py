import torch
import torch.nn as nn


class CyclicCNN(nn.Module):
    def __init__(self):
        super(CyclicCNN, self).__init__()

        # EEG decoder (spatial only)
        self.eeg_decoder = nn.Sequential(
            nn.Conv1d(in_channels=34, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # EEG encoder (spatial only)
        self.eeg_encoder = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=128, out_channels=64, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                in_channels=64, out_channels=34, kernel_size=3, padding=1
            ),
            nn.ReLU(),
        )

        # fMRI decoder (temporal only)
        self.fmri_decoder = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # fMRI encoder (temporal only)
        self.fmri_encoder = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=32, out_channels=16, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose3d(
                in_channels=16, out_channels=1, kernel_size=3, padding=1
            ),
            nn.ReLU(),
        )

    def eeg_to_fmri(self, eeg):
        latent = self.eeg_decoder(eeg.squeeze(2).squeeze(2))
        return self.fmri_encoder(latent.unsqueeze(2).unsqueeze(3).unsqueeze(4))

    def fmri_to_eeg(self, fmri):
        latent = self.fmri_decoder(fmri.unsqueeze(1))
        return self.eeg_encoder(latent.squeeze(2).squeeze(2).squeeze(2))

    def forward(self, eeg, fmri):
        eeg_to_fmri = self.eeg_to_fmri(eeg)
        fmri_to_eeg = self.fmri_to_eeg(fmri)
        eeg_reconstructed = self.fmri_to_eeg(eeg_to_fmri)
        fmri_reconstructed = self.eeg_to_fmri(fmri_to_eeg)
        return eeg_to_fmri, fmri_to_eeg, eeg_reconstructed, fmri_reconstructed

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv3DBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class EEGEncoder(nn.Module):
    def __init__(self):
        super(EEGEncoder, self).__init__()
        self.conv1 = Conv3DBlock(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv3DBlock(16, 32, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        import pdb;pdb.set_trace() 
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class fMRIEncoder(nn.Module):
    def __init__(self):
        super(fMRIEncoder, self).__init__()
        self.conv1 = Conv1DBlock(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv1DBlock(16, 32, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class EEGDecoder(nn.Module):
    def __init__(self):
        super(EEGDecoder, self).__init__()
        self.deconv1 = nn.ConvTranspose3d(32, 16, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose3d(16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        return x


class fMRIDecoder(nn.Module):
    def __init__(self):
        super(fMRIDecoder, self).__init__()
        self.deconv1 = nn.ConvTranspose1d(32, 16, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose1d(16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        return x




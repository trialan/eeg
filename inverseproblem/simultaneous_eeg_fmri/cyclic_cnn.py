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


class fMRIEncoder(nn.Module):
    def __init__(self, dropout_rate=0.):
        super(fMRIEncoder, self).__init__()
        self.conv1 = Conv3DBlock(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = Conv3DBlock(16, 32, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 16 * 16 * 8, 1024)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x.view(-1, 32, 32)

class EEGEncoder(nn.Module):
    def __init__(self, dropout_rate=0.):
        super(EEGEncoder, self).__init__()
        self.conv1 = Conv3DBlock(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv3DBlock(16, 32, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 6 * 1 * 34, 1024)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(1)
        x = x.permute(0, 1, 4, 2, 3)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x.view(-1, 32, 32)

class fMRIDecoder(nn.Module):
    def __init__(self, dropout_rate=0.):
        super(fMRIDecoder, self).__init__()
        self.fc = nn.Linear(1024, 32 * 16 * 16 * 8)
        self.deconv1 = nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose3d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.view(-1, 1024)
        x = self.dropout(x)
        x = self.fc(x)
        x = x.view(-1, 32, 16, 16, 8)
        x = self.deconv1(x)
        x = self.deconv2(x)
        return x.squeeze(1)

class EEGDecoder(nn.Module):
    def __init__(self, dropout_rate=0.):
        super(EEGDecoder, self).__init__()
        self.fc = nn.Linear(1024, 32 * 6 * 1 * 34)
        self.deconv1 = nn.ConvTranspose3d(32, 16, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose3d(16, 1, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.view(-1, 1024)
        x = self.dropout(x)
        x = self.fc(x)
        x = x.view(-1, 32, 6, 1, 34)
        x = self.deconv1(x)
        x = self.deconv2(x)
        return x.squeeze(1).squeeze(2).permute(0, 2, 1)

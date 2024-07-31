import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from eeg.utils import read_pickle
from eeg.inverseproblem.simultaneous_eeg_fmri.eeg_data import get_eeg_data
from eeg.inverseproblem.simultaneous_eeg_fmri.fmri_data import get_fmri_data
from eeg.inverseproblem.simultaneous_eeg_fmri.cyclic_cnn import CyclicCNN


root_dir = "/root/DS116/"
# root_dir = "/Users/thomasrialan/Documents/code/DS116/"


if __name__ == "__main__":
    X_eeg, y_eeg = get_eeg_data(root_dir)
    # X_fmri, y_fmri = get_fmri_data()
    y_fmri = read_pickle("fmri_y.pkl")
    X_fmri = read_pickle("fmri_X.pkl")

    assert len(X_eeg) == len(X_fmri)
    assert np.array_equal(y_eeg, y_fmri)

    eeg_data = torch.Tensor(X_eeg)  # shape: (4875, 34, 1001)
    fmri_data = torch.Tensor(X_fmri)  # shape: (4875, 64, 64, 32)
    dataset = TensorDataset(eeg_data, fmri_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = CyclicCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    mse_loss = nn.MSELoss()

    num_epochs = 100
    for epoch in range(num_epochs):
        for batch_eeg, batch_fmri in dataloader:
            optimizer.zero_grad()

            eeg_to_fmri, fmri_to_eeg, eeg_reconstructed, fmri_reconstructed = model(
                batch_eeg, batch_fmri
            )

            loss_eeg_to_fmri = mse_loss(
                eeg_to_fmri, batch_fmri.view(batch_fmri.size(0), -1, 1)
            )
            loss_fmri_to_eeg = mse_loss(fmri_to_eeg, batch_eeg)
            loss_eeg_cycle = mse_loss(eeg_reconstructed, batch_eeg)
            loss_fmri_cycle = mse_loss(
                fmri_reconstructed, batch_fmri.view(batch_fmri.size(0), -1, 1)
            )

            total_loss = (
                loss_eeg_to_fmri + loss_fmri_to_eeg + loss_eeg_cycle + loss_fmri_cycle
            )

            total_loss.backward()
            optimizer.step()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
        }, 'cyclic_cnn_checkpoint.pth')

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.4f}")



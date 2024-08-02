import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
Roughly correct, based on Fig.10, Appendix E.
"""

channel_mapping = {
    1:  {"name": "Fp1", "coords": (2, 9, 2)},
    2:  {"name": "Fp2", "coords": (10, 9, 2)},
    3:  {"name": "AF3", "coords": (4, 8, 3)},
    4:  {"name": "AF4", "coords": (8, 8, 3)},
    5:  {"name": "F7",  "coords": (2, 7, 2)},
    6:  {"name": "F3",  "coords": (4, 7, 4)},
    7:  {"name": "Fz",  "coords": (6, 7, 4)},
    8:  {"name": "F4",  "coords": (8, 7, 4)},
    9:  {"name": "F8",  "coords": (10, 7, 2)},
    10: {"name": "FC5", "coords": (3, 6, 3)},
    11: {"name": "FC1", "coords": (5, 6, 5)},
    12: {"name": "FC2", "coords": (7, 6, 5)},
    13: {"name": "FC6", "coords": (9, 6, 3)},
    14: {"name": "T7",  "coords": (2, 5, 2)},
    15: {"name": "C4",  "coords": (7, 5, 4)},
    16: {"name": "Cz",  "coords": (6, 5, 5)},
    17: {"name": "C3",  "coords": (5, 5, 4)},
    18: {"name": "T8",  "coords": (10, 5, 2)},
    19: {"name": "CP5", "coords": (3, 4, 3)},
    20: {"name": "CP1", "coords": (5, 4, 5)},
    21: {"name": "CP2", "coords": (7, 4, 5)},
    22: {"name": "CP6", "coords": (9, 4, 3)},
    23: {"name": "P7",  "coords": (2, 3, 2)},
    24: {"name": "P3",  "coords": (4, 3, 4)},
    25: {"name": "Pz",  "coords": (6, 3, 4)},
    26: {"name": "P4",  "coords": (8, 3, 4)},
    27: {"name": "P8",  "coords": (10, 3, 2)},
    28: {"name": "P07", "coords": (4, 2, 2)},
    29: {"name": "P03", "coords": (4, 2, 3)},
    30: {"name": "P04", "coords": (8, 2, 3)},
    31: {"name": "P08", "coords": (8, 2, 2)},
    32: {"name": "O1",  "coords": (5, 1, 2)},
    33: {"name": "Oz",  "coords": (6, 1, 2)},
    34: {"name": "O2",  "coords": (7, 1, 2)},
}


def eeg_to_volume(eeg_data):
    num_channels, num_timepoints = eeg_data.shape
    volume = np.zeros((11, 9, 5, num_timepoints))

    # Assign EEG data to volume
    for channel_ix, channel_data in channel_mapping.items():
        x, y, z = channel_data["coords"]
        volume[x-1, y-1, z-1, :] = eeg_data[channel_ix-1, :]

    return volume


def visualize_eeg_volume(channel_mapping):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the volume boundaries
    ax.plot([0, 11, 11, 0, 0], [0, 0, 9, 9, 0], [0, 0, 0, 0, 0], "k-")
    ax.plot([0, 11, 11, 0, 0], [0, 0, 9, 9, 0], [5, 5, 5, 5, 5], "k-")
    ax.plot([0, 0], [0, 0], [0, 5], "k-")
    ax.plot([11, 11], [0, 0], [0, 5], "k-")
    ax.plot([11, 11], [9, 9], [0, 5], "k-")
    ax.plot([0, 0], [9, 9], [0, 5], "k-")

    # Plot electrode positions
    for channel_ix, channel_data in channel_mapping.items():
        x, y, z = channel_data["coords"]
        ax.scatter(x, y, z-1, c="r", s=100)
        ax.text(x, y, z-1, channel_data["name"], fontsize=8)

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("EEG Electrode Positions in 3D Volume (12x10x6)")

    # Set axis limits
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 9)
    ax.set_zlim(0, 5)

    # Adjust the view angle
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    num_channels = 34
    num_timepoints = 1000
    dummy_eeg_data = np.random.rand(num_channels, num_timepoints)
    channel_names = list(channel_mapping.keys())  # Assuming all channels are present
    volume_data = eeg_to_volume(dummy_eeg_data)
    print(volume_data.shape)  # Should print (11, 9, 5, 1000)

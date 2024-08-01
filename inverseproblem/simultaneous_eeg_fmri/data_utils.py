import os
import glob
import numpy as np


def get_paths(root):
    def _get_paths(path_type, root):
        assert path_type in ["BOLD", "EEG", "behav"]
        paths = []
        file_name = {
            "BOLD": "bold_mcf_brain.nii.gz",
            "EEG": "EEG_rereferenced.mat",
            "behav": "behavdata.txt",
        }[path_type]

        subject_pattern = os.path.join(root, "sub*")

        for subject_dir in glob.glob(subject_pattern):
            pattern = os.path.join(subject_dir, path_type, "task002_run*", file_name)
            paths.extend(glob.glob(pattern))

        return np.array(sorted(paths))

    bold_paths = _get_paths("BOLD", root)
    eeg_paths = _get_paths("EEG", root)
    event_paths = _get_paths("behav", root)
    return bold_paths, eeg_paths, event_paths


def load_events(file_path):
    """load in mne Raw format: [time, 0., label]"""
    event_data = []
    with open(file_path, "r") as file:
        next(file)
        for line in file:
            components = line.strip().split()
            if len(components) == 4:
                time = np.int64(float(components[0]) * 1000)  # time in seconds
                event_type = np.int64(components[1])
                event_data.append(np.array([time, event_type]))
    event_data = np.array(event_data)
    mne_fmt = np.concatenate(
        (
            event_data[:, :1],
            np.int64(np.zeros((len(event_data), 1))),
            event_data[:, 1:],
        ),
        axis=1,
    )
    return mne_fmt



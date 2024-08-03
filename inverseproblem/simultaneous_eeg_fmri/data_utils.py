import os
import glob
import numpy as np


def get_bv_paths(root_dir):
    vtc_files = glob.glob(os.path.join(root_dir, "clean_fmri_ds116/*.vtc"))
    vtc_files = sorted(vtc_files)
    event_files = [get_matching_event_file(f, root_dir) for f in vtc_files]
    eeg_files = [get_matching_eeg_file(f, root_dir) for f in vtc_files]
    return np.array(vtc_files), np.array(eeg_files), np.array(event_files)


def get_matching_eeg_file(filename, root_dir):
    """ Match EEG files in DS116 to BrainVoyager .vtc files in
        the dataset hand cleaned by Nyx """
    filename = filename.split("/")[-1]
    subject = filename.split("_")[0]
    run = filename.split("_")[2]
    matching_file = os.path.join(root_dir, "DS116", subject, "EEG", f"task002_{run}" , "EEG_rereferenced.mat")
    return matching_file



def get_matching_event_file(filename, root_dir):
    """ Match events (event files in DS116) to BrainVoyager .vtc files in
        the dataset hand cleaned by Nyx """
    filename = filename.split("/")[-1]
    subject = filename.split("_")[0]
    run = filename.split("_")[2]
    matching_file = os.path.join(root_dir, "DS116", subject, "behav", f"task002_{run}" , "behavdata.txt")
    return matching_file


def get_DS116_paths(root):
    """ 'DS116' is the processed dataset downloaded her:
        https://legacy.openfmri.org/dataset/ds000116/
    """
    def _get_paths(path_type, root):
        """ important to sort for cross machine reproducibility """
        assert path_type in ["BOLD", "EEG", "behav"]
        paths = []
        file_name = {
            "BOLD": "bold_mcf_brain.nii.gz",
            "EEG": "EEG_rereferenced.mat",
            "behav": "behavdata.txt",
        }[path_type]

        subject_pattern = os.path.join(root, "DS116/sub*")

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



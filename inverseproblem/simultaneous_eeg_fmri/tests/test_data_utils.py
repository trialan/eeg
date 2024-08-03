import os
from eeg.inverseproblem.simultaneous_eeg_fmri.data_utils import (get_matching_event_file,
                                                                 get_bv_paths,
                                                                 get_matching_eeg_file)


root_dir = "/Users/thomasrialan/Documents/code/"


def test_get_event_paths():
    """ This should return events paths correctly lined up with the
    corresponding .vtc file """
    test_file = "/some/path/to/sub005_task002_run003_SCCAI_3DMCTS_THPGLMF3c_256_sinc3_3x0.9_MNI.vtc"
    correct_events_file = os.path.join(root_dir, "DS116/sub005/behav/task002_run003/behavdata.txt")
    correct_eeg_file = os.path.join(root_dir, "DS116/sub005/EEG/task002_run003/EEG_rereferenced.mat")

    matching_file = get_matching_event_file(test_file, root_dir)
    assert matching_file == correct_events_file

    matching_eeg_file = get_matching_eeg_file(test_file, root_dir)
    assert matching_eeg_file == correct_eeg_file


def test_get_bv_paths():
    bold_paths, eeg_paths, event_paths = get_bv_paths(root_dir)
    assert len(bold_paths) == len(eeg_paths) == len(event_paths)

    for i in range(len(bold_paths)):
        assert _get_run(event_paths[i]) == _get_run(eeg_paths[i])
        assert _get_bold_run(bold_paths[i]) == _get_run(eeg_paths[i])


def _get_run(path):
    """ works for EEG and Events """
    return path.split("/")[-2].split("_")[1]


def _get_bold_run(path):
    return path.split("/")[-1].split("_")[2]


def _get_events_run(path):
    pass


def _get_subject(path):
    pass


def _get_task(path):
    pass

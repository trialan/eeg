import mne
from mne import Epochs, pick_types
from mne.minimum_norm import apply_inverse_epochs, make_inverse_operator

import matplotlib.pyplot as plt
import numpy as np

from eeg import physionet_runs
from eeg.data import get_raw_data, tmax, tmin

"""
https://mne.tools/stable/auto_tutorials/inverse/30_mne_dspm_loreta.html
"""

raw = get_raw_data(1, physionet_runs)

raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")
picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
event_id = ["hands", "feet"]
epochs = Epochs(
    raw,
    event_id=event_id,
    tmin=tmin,
    tmax=tmax,
    proj=True,
    picks=picks,
    baseline=None,
    preload=True,
)

# in the tutorial they do the 'avg' localistion for the avg evoked response
evoked = epochs.average().pick(picks)
#evoked.plot(time_unit="s")

from eeg.inverseproblem.leadfield import compute_forward_solution

fwd = compute_forward_solution()
noise_cov = mne.compute_covariance(
    epochs, tmax=tmax, method=["shrunk", "empirical"], rank=None, verbose=True
)

inverse_operator = make_inverse_operator(
    evoked.info, fwd, noise_cov, loose=0.2, depth=0.8
)
del fwd

method = "eLORETA"
snr = 3.0 #why??
lambda2 = 1.0 / snr**2
stcs = apply_inverse_epochs(
    epochs,
    inverse_operator,
    lambda2,
    method=method,
)



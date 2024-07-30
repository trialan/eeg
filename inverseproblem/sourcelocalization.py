import matplotlib.pyplot as plt
import numpy as np

import mne
from mne.datasets import sample
from mne.minimum_norm import apply_inverse, make_inverse_operator
data_path = sample.data_path()
raw_fname = data_path / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"

raw = mne.io.read_raw_fif(raw_fname)  # already has an average reference
events = mne.find_events(raw, stim_channel="STI 014")

event_id = dict(aud_l=1)  # event trigger and conditions
tmin = -0.2  # start of each epoch (200ms before the trigger)
tmax = 0.5  # end of each epoch (500ms after the trigger)
raw.info["bads"] = ["MEG 2443", "EEG 053"]
baseline = (None, 0)  # means from the first instant to t = 0
reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)

epochs = mne.Epochs(
            raw,
                events,
                    event_id,
                        tmin,
                            tmax,
                                proj=True,
                                    picks=("meg", "eog"),
                                        baseline=baseline,
                                            reject=reject,
                                            )

noise_cov = mne.compute_covariance(
            epochs, tmax=0.0, method=["shrunk", "empirical"], rank=None, verbose=True
            )

fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, raw.info)

evoked = epochs.average().pick("meg")
evoked.plot(time_unit="s")
evoked.plot_topomap(times=np.linspace(0.05, 0.15, 5), ch_type="mag")

evoked.plot_white(noise_cov, time_unit="s")
del epochs, raw  # to save memory

fname_fwd = data_path / "MEG" / "sample" / "sample_audvis-meg-oct-6-fwd.fif"
fwd = mne.read_forward_solution(fname_fwd)

inverse_operator = make_inverse_operator(
            evoked.info, fwd, noise_cov, loose=0.2, depth=0.8
            )
del fwd

method = "dSPM"
snr = 3.0
lambda2 = 1.0 / snr**2
stc, residual = apply_inverse(
            evoked,
                inverse_operator,
                    lambda2,
                        method=method,
                            pick_ori=None,
                                return_residual=True,
                                    verbose=True,
                                    )

fig, ax = plt.subplots()
ax.plot(1e3 * stc.times, stc.data[::100, :].T)
ax.set(xlabel="time (ms)", ylabel="%s value" % method)

fig, axes = plt.subplots(2, 1)
evoked.plot(axes=axes)
for ax in axes:
        for text in list(ax.texts):
                text.remove()
        for line in ax.lines:
                line.set_color("#98df81")
residual.plot(axes=axes)

vertno_max, time_max = stc.get_peak(hemi="rh")

subjects_dir = data_path / "subjects"
surfer_kwargs = dict(
            hemi="rh",
                subjects_dir=subjects_dir,
                    clim=dict(kind="value", lims=[8, 12, 15]),
                        views="lateral",
                            initial_time=time_max,
                                time_unit="s",
                                    size=(800, 800),
                                        smoothing_steps=10,
                                        )
brain = stc.plot(**surfer_kwargs)
brain.add_foci(
            vertno_max,
                coords_as_verts=True,
                    hemi="rh",
                        color="blue",
                            scale_factor=0.6,
                                alpha=0.5,
                                )
brain.add_text(
            0.1, 0.9, "dSPM (plus location of maximal activation)", "title", font_size=14
            )

# The documentation website's movie is generated with:
# brain.save_movie(..., tmin=0.05, tmax=0.15, interpolation='linear',
#                  time_dilation=20, framerate=10, time_viewer=True)




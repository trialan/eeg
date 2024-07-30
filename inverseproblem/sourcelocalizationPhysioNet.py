import matplotlib.pyplot as plt
import numpy as np
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.minimum_norm import apply_inverse, make_inverse_operator
from mne.datasets import sample
from leadfield import compute_forward_solution
from data import get_raw_data
from scipy.stats import boxcox

def avg_power_vector(u):
    """ Return the avg power of a vector """
    assert len(u.shape) == 1
    powers = [el**2 for el in u]
    avg_power = sum(powers) / len(u)
    return avg_power

def avg_power_matrix(m):
    """ Drop the time dimension on matrix by averaging power """
    assert len(m.shape) == 2
    y = np.array([avg_power_vector(row) for row in m])
    transformed_y, best_lambda = boxcox(y)
    return transformed_y

if __name__ == '__main__': 
    # Load the EEG data from PhysioNet for subject
    subject = 5
    runs = [10, 14]  # motor imagery: hands vs feet
    raw = get_raw_data(subject, runs)
    
    # Preprocess the data
    events, event_id = mne.events_from_annotations(raw)
    tmin, tmax = 1.0, 2.0
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)

    # Compute noise covariance
    noise_cov = mne.compute_covariance(epochs, tmin=1.0, tmax=2.0, method=['shrunk', 'empirical'])
    
    # Compute the average evoked response
    print(epochs['feet'])
    evoked = epochs['feet'].average()
    
    # Compute the forward solution (once, for subject 1 with surface from MNE sample dataset)
    fwd = compute_forward_solution()

    # Make an inverse operator
    inverse_operator = make_inverse_operator(evoked.info, fwd, noise_cov, loose=0.2, depth=0.8)

    # Compute the inverse solution for subject
    method = "dSPM"
    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    stc, residual = apply_inverse(evoked, inverse_operator, lambda2, method=method, pick_ori=None, return_residual=True)
    
    print(stc.data.shape)
    vec = avg_power_matrix(stc.data)
    print(vec.shape)
    print(vec)
    
    # Plot the results
    fig, ax = plt.subplots()
    for i, data in enumerate(stc.data[::10, :]):
        ax.plot(1e3 * stc.times, data, label=f'Source {i}')
    ax.set(xlabel="time (ms)", ylabel="%s value" % method)
    ax.legend(loc='upper right')
    plt.show()

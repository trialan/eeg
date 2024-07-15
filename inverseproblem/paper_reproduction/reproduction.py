import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from eeg.data import get_data

"""
Paper: Real-time automated EEG tracking of brain states using neural field theory
Authors: R.G. Abeysuriya P.A. Robinson
Lefebvre Lab code: https://github.com/Lefebvrelab/SpectralNeuralModels/blob/master/code/robinson.py
"""


def model_error(params):
    model = Model(params)
    simulated_power_spectrum = np.array([model.power(w) for w in omegas])
    return np.sum((simulated_power_spectrum - real_power_spectrum)**2)


class Model():
    def __init__(self, params):
        #Constants
        L = 0.5  # Line 194 in paper justifies this
        self.L_x = L  # Length in x direction
        self.L_y = L  # Length in y direction
        self.gamma_e = 116
        self.r_e = 86 / 1000. #86mm
        self.G_sn = 1.0
        self.G_rs = 0.1
        self.G_re = 0.2
        self.modulus_phi_n = 1e-5 #Lines 182, 247 in paper.
        self.k_0 = 1/10. #Line 199 in paper

        #Parameters we want to optimize, initialized to their Table (1) values.
        #I also used values from LefebreLab's __init__ function.
        self.G_ee = params[0]
        self.G_ei = params[1]
        self.G_ese = params[2]
        self.G_esre = params[3]
        self.G_srs = params[4]
        self.alpha = params[5]
        self.beta = params[6]
        self.t0 = params[7]
        self.f_EMG = 40
        self.A_EMG = 0.5e-12

    def power(self, omega):
        eeg_power = self.eeg_power(omega)
        emg_power = self.emg_power(omega)
        return eeg_power + emg_power

    def G(self):
        G_sr = self.G_srs / self.G_rs
        G_dict = {"sn":  self.G_sn,
                  "ee":  self.G_ee,
                  "rs":  self.G_rs,
                  "srs": self.G_srs,
                  "ei":  self.G_ei,
                  "ese": self.G_ese,
                  "esre": self.G_esre,
                  "es": self.G_esre / (G_sr * self.G_re), #line 178 in paper
                  "ee": self.G_ee}
        return G_dict

    def eeg_power(self, omega):
        """ Eqn. (13) for P(omega) """
        m_range = np.arange(-10, 11) #ranges should be -inf, +inf
        n_range = np.arange(-10, 11)
        delta_kx = 3 * np.pi / self.L_x
        delta_ky = 2 * np.pi / self.L_y

        eeg_power = 0.0
        for m in m_range:
            for n in n_range:
                kx = 2 * np.pi * m / self.L_x
                ky = 2 * np.pi * n / self.L_y
                k = np.sqrt(kx**2 + ky**2)
                modulus_phi_e_k = self.modulus_phi_e(kx, ky, omega)
                eeg_power += modulus_phi_e_k**2 * self.F(k) * delta_kx * delta_ky
        return eeg_power

    def emg_power(self, omega):
        numerator = self.A_EMG * omega / (2 * np.pi * self.f_EMG)
        denominator = 1 + (omega / (2 * np.pi * self.f_EMG)**2)**2
        emg_power = numerator / denominator
        return emg_power


    def modulus_phi_e(self, kx, ky, omega):
        """ Eqn. (10) """
        G = self.G()
        L = self.L(omega)
        k2 = kx**2 + ky**2
        oscillating_term = np.exp(1j * omega * self.t0 / 2)
        q2r2 = self.q2r2(omega, oscillating_term)
        numerator = G["es"] * G["sn"] * L**2 * oscillating_term
        denominator = (1 - G["srs"] * L**2) * (1 - G["ei"] * L) * (k2 * self.r_e**2 + q2r2)
        modulus_phi_e = self.modulus_phi_n * numerator / denominator
        return modulus_phi_e

    def L(self, omega):
        """ Eqn. (12) for L(omega) """
        alpha, beta = self.alpha, self.beta
        L = (1 - 1j*omega/alpha)**-1 * (1 - 1j*omega/beta)**-1
        return L

    def F(self, k):
        """ Eqn. (15) Cerebrospinal fluid filter. Line 199 of paper for k_0. """
        k_0 = 1/10. #unit of k: inverse meters
        return np.exp(-(k/self.k_0)**2)

    def q2r2(self, omega, oscillating_term):
        """ Eqn. (11) """
        G = self.G()
        L = self.L(omega)
        term_1 = (1 - 1j * omega / self.gamma_e)**2
        term_2 = 1 / (1 - G["ei"] * L)
        term_3_numerator = (L**2 * G["ese"] + L**3 * G["esre"]) * oscillating_term
        term_3_denominator = 1 - L**2 * G["srs"]
        term_3 = L * G["ee"] + term_3_numerator / term_3_denominator
        q2r2 = term_1 - term_2 * term_3
        return q2r2


def get_experimental_power_spectrum():
    """ Power spectrum for each channel, only keep half of frequencies
        for readability since it's symmetric """
    eeg_data, _ = get_data(2)
    eeg_data = eeg_data[0]
    fft_eeg = np.fft.fft(eeg_data, axis=1)
    power_spectrum = np.abs(fft_eeg) ** 2
    sampling_rate = 160
    frequencies = np.fft.fftfreq(eeg_data.shape[1], d=1/sampling_rate)
    positive_freqs = frequencies[:eeg_data.shape[1] // 2]
    positive_power_spectrum = power_spectrum[:, :eeg_data.shape[1] // 2]
    return positive_power_spectrum, positive_freqs


if __name__ == '__main__':
    ac_real_power_spectrum, frequencies = get_experimental_power_spectrum()
    omegas = 2 * np.pi * frequencies

    real_power_spectrum = np.sum(ac_real_power_spectrum, axis=0)

    #Written in this un-intuitive way to fit in with minimize
    initial_params = [("G_ee", 5.4),
                      ("G_ei", -7),
                      ("G_ese", 100.),
                      ("G_esre", -100),
                      ("G_srs", -1.0),
                      ("alpha", 75),
                      ("beta", 75*3.8),
                      ("t0", 84/1000.),
                      ("f_EMG", 40),
                      ("A_EMG", 0.5e-12)]
    initial_params = [p[1] for p in initial_params]

    result = minimize(model_error, initial_params)
    print(result)

    good_model = Model(params = result.x)
    simulated_power_spectrum = np.array([good_model.power(w) for w in omegas])
    plt.plot(omegas, real_power_spectrum, color='b')
    plt.plot(omegas, simulated_power_spectrum, color='r')
    plt.show()


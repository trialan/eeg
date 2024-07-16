import cvxpy as cp
import numpy as np

from eeg.data import get_data
from eeg.inverseproblem.mne_approach import compute_lead_field_matrix


"""
SOCP: Second Order Cone Programming. This file is about solving equation (8)
in Ou et.al. 2009, "A distributed spatio-temporal EEG/MEG inverse solver".
"""


A = compute_lead_field_matrix()
lambda_val = 1e9 #See "Sensitivity to regularization strength" section, p.939


def compute_brain_signal(y):
    """ y is EEG recording, shape (M=64,). S is brain Returns S_opt, (N,). """
    S = cp.Variable((N, K))
    prob = setup_optimisation_problem(S, y)
    result = prob.solve()
    S_opt = S.value
    return S_opt


def setup_optimisation_problem(S, y):
    M, N = A.shape
    K = y.shape[1]  # Number of temporal basis functions

    # Variables
    q = cp.Variable()
    z = cp.Variable()
    w = cp.Variable(K)
    r = cp.Variable(N)

    # Objective function
    objective = cp.Minimize(q + lambda_val * z)

    # Constraints
    constraints = []
    for k in range(K):
        constraints.append(cp.norm(y[:, k] - A @ S[:, k], 2) <= w[k])

    constraints.append(cp.sum(w) <= q)

    for n in range(N):
        constraints.append(cp.norm(S[n, :], 2) <= r[n])

    constraints.append(cp.sum(r) <= z)
    prob = cp.Problem(objective, constraints)
    return prob


if __name__ == '__main__':



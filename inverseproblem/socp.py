import time
from tqdm import tqdm
import cvxpy as cp
import numpy as np
import multiprocessing as mp
import pickle

from eeg.data import get_data
from eeg.inverseproblem.leadfield import compute_lead_field_matrix

"""
SOCP: Second Order Cone Programming. This file is about solving equation (8)
in Ou et.al. 2009, "A distributed spatio-temporal EEG/MEG inverse solver".

Using the MOSEK solver is very important, 5x speedup over default solver,
it requires a license, free trial available at: mosek.com/license/request/

This solver is important because it solves the dual conic problem, which is
much easier to solve. This is discussed in appendix B of the paper and in
these lecture notes: people.eecs.berkeley.edu/~elghaoui/Teaching/EE227A/lecture10.pdf
"""


lambda_val = 1e9 #See "Sensitivity to regularization strength" section, p.939


def compute_brain_signal(y, A, solver=cp.MOSEK):
    """ y is EEG recording, shape (n_channels, n_times). S is brain Returns S_opt, (n_sources, n_times). """
    S = cp.Variable((A.shape[1], y.shape[1]))
    prob = setup_optimisation_problem(S, y, A)
    result = prob.solve(solver=solver)
    S_opt = S.value
    return S_opt


def setup_optimisation_problem(S, y, A):
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
    A = compute_lead_field_matrix()
    X, _ = get_data()

    print("\n #### Beginning the SOCP solving #### \n")
    brain_signals = []
    for Y in tqdm(X):
        U, S, VT = np.linalg.svd(Y, full_matrices=False)
        Psi_Y = VT.T

        K = 3
        Psi_Y_reduced = Psi_Y[:, :K]
        Y_transformed_reduced = np.dot(Y, Psi_Y_reduced)

        s = compute_brain_signal(Y_transformed_reduced,
                                             A, solver=cp.MOSEK)
        brain_signals.append(s)
    S = np.array(brain_signals)

    with open('array_data.pkl', 'wb') as file:
        pickle.dump(S, file)



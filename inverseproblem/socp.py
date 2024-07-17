import cvxpy as cp
import numpy as np
import multiprocessing as mp

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


def compute_brain_signal(y, A):
    """ y is EEG recording, shape (n_channels, n_times). S is brain Returns S_opt, (n_sources, n_times). """
    S = cp.Variable((A.shape[1], y.shape[1]))
    prob = setup_optimisation_problem(S, y, A)
    result = prob.solve(solver=cp.MOSEK)
    S_opt = S.value
    return S_opt


def run_compute_brain_signal_multithread(X, A):
    num_sub_matrices = X.shape[0]
    with mp.Pool(4) as pool:
        results = pool.starmap(process_sub_matrix, [(X, i, A) for i in range(num_sub_matrices)])
    return results


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


def process_sub_matrix(X, index, A):
    Y = X[index]
    S_opt = compute_brain_signal(Y, A)
    return S_opt

if __name__ == '__main__':
    A = compute_lead_field_matrix()
    import time
    X, _ = get_data(1)
    print("\n #### Beginning the SOCP solving #### \n")
    t0 = time.time()
    S = run_compute_brain_signal_multithread(X, A)
    t1 = time.time()
    print(t1-t0)




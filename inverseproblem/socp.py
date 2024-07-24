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


lambda_val = 1e9 # 1e9See "Sensitivity to regularization strength" section, p.939


def l1_l2_solver(y, A, solver=cp.MOSEK):
    """ y is EEG recording, shape (n_channels, n_times). S is brain Returns S_opt, (n_sources, n_times). """
    S = cp.Variable((A.shape[1], y.shape[1]))
    prob = setup_cone_problem(S, y, A)
    result = prob.solve(solver=solver)
    print(f'Problem status: {prob.status}')
    if prob.status != cp.OPTIMAL:
        print('Solver did not converge!')
    S_opt = S.value
    return S_opt


def eLORETA_solver(y, A, alpha):
    n_channels, n_times = y.shape
    n_channels, n_voxels = A.shape

    S = cp.Variable((n_voxels, n_times))
    W = np.eye(n_voxels)
    objective = cp.Minimize(cp.norm(y - A @ S, 'fro')**2 + alpha * cp.norm(W @ S, 'fro')**2)
    prob = cp.Problem(objective)
    prob.solve()
    S_opt = S.value
    return S_opt


def setup_cone_problem(S, y, A):
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
    #X, _ = get_data()

    from sklearn.metrics import mean_absolute_error

    n_sources = 1422  # Number of sources in the brain model
    n_times = 161    # Number of time points

    sparsity_level = 0.1  # Proportion of non-zero entries
    J = np.zeros((n_sources, n_times))
    source_dipole = np.random.randint(n_sources)

    source_time_series = np.sin(2.0 * np.pi * 18.0 * np.arange(161) * 1./160) * 10e-9

    J[source_dipole] = source_time_series
    Y = A @ J

    print(f"Mean abs. value of J: {np.mean(np.abs(J))}")
    for lambda_val in [1e10, 1e11, 1e15]:
        # Evaluate the accuracy
        J_recovered = eLORETA_solver(Y, A, lambda_val)
        #J_loreta = eLORETA_solver(Y, A, 1e-2)
        mae = mean_absolute_error(J, J_recovered)
        print(f'Lambda: {lambda_val}, Mean Abs. Error: {mae}')
        print(f"Mean abs. val. J_recovered: {np.mean(np.abs(J_recovered))}")


    """
    print("\n #### Beginning the SOCP solving #### \n")
    brain_signals = []
    for Y in tqdm(X):
        U, S, VT = np.linalg.svd(Y, full_matrices=False)
        Psi_Y = VT.T
        Y_transformed = np.dot(Y, Psi_Y)

        K = 3
        Psi_Y_reduced = Psi_Y[:, :K]
        Y_transformed_reduced = np.dot(Y, Psi_Y_reduced)

        s = compute_brain_signal(Y_transformed_reduced,
                                             A, solver=cp.MOSEK)
        brain_signals.append(s)
    S = np.array(brain_signals)

    with open('array_data.pkl', 'wb') as file:
        pickle.dump(S, file)
    """



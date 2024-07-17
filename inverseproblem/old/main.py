import numpy as np

from eeg.inverseproblem.electromag import generate_random_scalp_brain_efield
from eeg.inverseproblem.spheres import Brain, Scalp
from eeg.laplacian import compute_mesh_eigenvectors_and_values


def assert_matrix_is_invertible(M):
    """ Square, full-rank, non-zero determinant """
    assert len(M.shape) == 2
    assert M.shape[1] == M.shape[0]
    assert np.linalg.matrix_rank(M) == 100.
    assert np.linalg.det(M) != 0.


if __name__ == '__main__':
    brain = Brain()
    scalp = Scalp()

    B, _ = compute_mesh_eigenvectors_and_values(brain.mesh)
    S, _ = compute_mesh_eigenvectors_and_values(scalp.mesh)

    assert_matrix_is_invertible(B)
    assert_matrix_is_invertible(S)

    """
        An E-field vector x living on the scalp can be re-written in the basis
        of the brain by doing y = B^(-1) x.
    """

    #Use our simulator to get E-field on the scalp from activity on the brain
    brain_field, scalp_field = generate_random_scalp_brain_efield(brain, scalp)

    # I want the vectors to be columns of my matrices for visually intuitive
    # Linear Algebra operations
    brain_field = brain_field.T
    scalp_field = scalp_field.T

    """
        The problem isn't so much finding the eigenvectors of the brain mesh and
        re-writing our EEG in that basis, this much is trivial. What we actually
        want is to know: what signal on the brain caused this signal on the scalp
        measured by the EEG.

        Perhaps it is worth it though to consider how LSP(Brain)+FgMDM compares to
        LSP(Scalp)+FgMDM. --> we can experiment with this tomorrow.
    """

    scalp_field_in_brain_basis = np.linalg.inverse(B) @ scalp_field



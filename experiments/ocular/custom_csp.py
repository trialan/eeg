import numpy as np
from scipy.linalg import eigh
from mne._fiff.meas_info import create_info
from mne.cov import _compute_rank_raw_array, _regularized_covariance, _smart_eigh
from mne.fixes import pinv
from mne.utils import (
    _check_option,
    _validate_type,
    _verbose_safe_false,
    copy_doc,
    fill_doc,
)
from sklearn.base import BaseEstimator, TransformerMixin
from pyentrp import entropy as ent
from scipy.signal import coherence, hilbert
import pywt
from mne.time_frequency import psd_array_welch

"""
This file implements a custom version of mne.decoding.CSP which,
instead of transforming into average_power, transforms into features
"""


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """ (Separate) Can be used in a sklearn Pipeline """
    def __init__(self, feats_extractor):
        self.sf = 160 #sampling frequency in Hz
        self.feats_extractor = feats_extractor

    def fit(self, X, y=None):
        # No fitting necessary, so we just return self
        return self

    def transform(self, X):
        return self.feats_extractor(X, self.sf)


@fill_doc
class CustomCSP(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        n_components=4,
        reg=None,
        log=None,
        cov_est="concat",
        feats_extractor=None,
        transform_into="feats",
        norm_trace=False,
        cov_method_params=None,
        rank=None,
        component_order="mutual_info",
    ):
        self.feats_extractor = feats_extractor
        # Init default CSP
        if not isinstance(n_components, int):
            raise ValueError("n_components must be an integer.")
        self.n_components = n_components
        self.rank = rank
        self.reg = reg

        # Init default cov_est
        if not (cov_est == "concat" or cov_est == "epoch"):
            raise ValueError("unknown covariance estimation method")
        self.cov_est = cov_est

        # Init default transform_into
        self.transform_into = _check_option(
            "transform_into", transform_into, ["feats"]
        )

        # Init default log
        if transform_into == "feats":
            if log is not None and not isinstance(log, bool):
                raise ValueError(
                    "log must be a boolean if transform_into == " '"feats".'
                )
        else:
            raise ValueError(
                    "transform_into should be == feats"
                )
        self.log = log

        _validate_type(norm_trace, bool, "norm_trace")
        self.norm_trace = norm_trace
        self.cov_method_params = cov_method_params
        self.component_order = _check_option(
            "component_order", component_order, ("mutual_info", "alternate")
        )

    def _check_Xy(self, X, y=None):
        """Check input data."""
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be of type ndarray (got %s)." % type(X))
        if y is not None:
            if len(X) != len(y) or len(y) < 1:
                raise ValueError("X and y must have the same length.")
        if X.ndim < 3:
            raise ValueError("X must have at least 3 dimensions.")

    def fit(self, X, y):
        self._check_Xy(X, y)

        self._classes = np.unique(y)
        n_classes = len(self._classes)
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2.")
        if n_classes > 2 and self.component_order == "alternate":
            raise ValueError(
                "component_order='alternate' requires two classes, but data contains "
                f"{n_classes} classes; use component_order='mutual_info' instead."
            )

        # Convert rank to one that will run
        _validate_type(self.rank, (dict, None), "rank")

        covs, sample_weights = self._compute_covariance_matrices(X, y)
        eigen_vectors, eigen_values = self._decompose_covs(covs, sample_weights)
        ix = self._order_components(
            covs, sample_weights, eigen_vectors, eigen_values, self.component_order
        )

        eigen_vectors = eigen_vectors[:, ix]

        self.filters_ = eigen_vectors.T
        self.patterns_ = pinv(eigen_vectors)

        pick_filters = self.filters_[: self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])

        # compute features (mean power)
        X = (X**2).mean(axis=2)

        # To standardize features
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

        return self

    def transform(self, X):
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be of type ndarray (got %s)." % type(X))
        if self.filters_ is None:
            raise RuntimeError(
                "No filters available. Please first fit CSP " "decomposition."
            )

        pick_filters = self.filters_[: self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])

        # compute features (mean band power)
        if self.transform_into == "average_power":
            X = (X**2).mean(axis=2)
            log = True if self.log is None else self.log
            if log:
                X = np.log(X)
            else:
                X -= self.mean_
                X /= self.std_
        if self.transform_into == "feats":
            X = self.feats_extractor(X)
        return X

    @copy_doc(TransformerMixin.fit_transform)
    def fit_transform(self, X, y, **fit_params):  # noqa: D102
        return super().fit_transform(X, y=y, **fit_params)



    def _compute_covariance_matrices(self, X, y):
        _, n_channels, _ = X.shape

        if self.cov_est == "concat":
            cov_estimator = self._concat_cov
        elif self.cov_est == "epoch":
            cov_estimator = self._epoch_cov

        # Someday we could allow the user to pass this, then we wouldn't need to convert
        # but in the meantime they can use a pipeline with a scaler
        self._info = create_info(n_channels, 1000.0, "mag")
        if self.rank is None:
            self._rank = _compute_rank_raw_array(
                X.transpose(1, 0, 2).reshape(X.shape[1], -1),
                self._info,
                rank=None,
                scalings=None,
                log_ch_type="data",
            )
        else:
            self._rank = {"mag": sum(self.rank.values())}

        covs = []
        sample_weights = []
        for ci, this_class in enumerate(self._classes):
            cov, weight = cov_estimator(
                X[y == this_class],
                cov_kind=f"class={this_class}",
                log_rank=ci == 0,
            )

            if self.norm_trace:
                cov /= np.trace(cov)

            covs.append(cov)
            sample_weights.append(weight)

        return np.stack(covs), np.array(sample_weights)

    def _concat_cov(self, x_class, *, cov_kind, log_rank):
        """Concatenate epochs before computing the covariance."""
        _, n_channels, _ = x_class.shape

        x_class = x_class.transpose(1, 0, 2).reshape(n_channels, -1)
        cov = _regularized_covariance(
            x_class,
            reg=self.reg,
            method_params=self.cov_method_params,
            rank=self._rank,
            info=self._info,
            cov_kind=cov_kind,
            log_rank=log_rank,
            log_ch_type="data",
        )
        weight = x_class.shape[0]

        return cov, weight

    def _epoch_cov(self, x_class, *, cov_kind, log_rank):
        """Mean of per-epoch covariances."""
        cov = sum(
            _regularized_covariance(
                this_X,
                reg=self.reg,
                method_params=self.cov_method_params,
                rank=self._rank,
                info=self._info,
                cov_kind=cov_kind,
                log_rank=log_rank and ii == 0,
                log_ch_type="data",
            )
            for ii, this_X in enumerate(x_class)
        )
        cov /= len(x_class)
        weight = len(x_class)

        return cov, weight

    def _decompose_covs(self, covs, sample_weights):
        n_classes = len(covs)
        n_channels = covs[0].shape[0]
        assert self._rank is not None  # should happen in _compute_covariance_matrices
        _, sub_vec, mask = _smart_eigh(
            covs.mean(0),
            self._info,
            self._rank,
            proj_subspace=True,
            do_compute_rank=False,
            log_ch_type="data",
            verbose=_verbose_safe_false(),
        )
        sub_vec = sub_vec[mask]
        covs = np.array([sub_vec @ cov @ sub_vec.T for cov in covs], float)
        assert covs[0].shape == (mask.sum(),) * 2
        if n_classes == 2:
            eigen_values, eigen_vectors = eigh(covs[0], covs.sum(0))
        else:
            # The multiclass case is adapted from
            # http://github.com/alexandrebarachant/pyRiemann
            eigen_vectors, D = _ajd_pham(covs)
            eigen_vectors = self._normalize_eigenvectors(
                eigen_vectors.T, covs, sample_weights
            )
            eigen_values = None
        # project back
        eigen_vectors = sub_vec.T @ eigen_vectors
        assert eigen_vectors.shape == (n_channels, mask.sum())
        return eigen_vectors, eigen_values

    def _compute_mutual_info(self, covs, sample_weights, eigen_vectors):
        class_probas = sample_weights / sample_weights.sum()

        mutual_info = []
        for jj in range(eigen_vectors.shape[1]):
            aa, bb = 0, 0
            for cov, prob in zip(covs, class_probas):
                tmp = np.dot(np.dot(eigen_vectors[:, jj].T, cov), eigen_vectors[:, jj])
                aa += prob * np.log(np.sqrt(tmp))
                bb += prob * (tmp**2 - 1)
            mi = -(aa + (3.0 / 16) * (bb**2))
            mutual_info.append(mi)

        return mutual_info

    def _normalize_eigenvectors(self, eigen_vectors, covs, sample_weights):
        # Here we apply an euclidean mean. See pyRiemann for other metrics
        mean_cov = np.average(covs, axis=0, weights=sample_weights)

        for ii in range(eigen_vectors.shape[1]):
            tmp = np.dot(np.dot(eigen_vectors[:, ii].T, mean_cov), eigen_vectors[:, ii])
            eigen_vectors[:, ii] /= np.sqrt(tmp)
        return eigen_vectors

    def _order_components(
        self, covs, sample_weights, eigen_vectors, eigen_values, component_order
    ):
        n_classes = len(self._classes)
        if component_order == "mutual_info" and n_classes > 2:
            mutual_info = self._compute_mutual_info(covs, sample_weights, eigen_vectors)
            ix = np.argsort(mutual_info)[::-1]
        elif component_order == "mutual_info" and n_classes == 2:
            ix = np.argsort(np.abs(eigen_values - 0.5))[::-1]
        elif component_order == "alternate" and n_classes == 2:
            i = np.argsort(eigen_values)
            ix = np.empty_like(i)
            ix[1::2] = i[: len(i) // 2]
            ix[0::2] = i[len(i) // 2 :][::-1]
        return ix


def _ajd_pham(X, eps=1e-6, max_iter=15):
    """Approximate joint diagonalization based on Pham's algorithm.

    This is a direct implementation of the PHAM's AJD algorithm [1].

    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_channels, n_channels)
        A set of covariance matrices to diagonalize.
    eps : float, default 1e-6
        The tolerance for stopping criterion.
    max_iter : int, default 1000
        The maximum number of iteration to reach convergence.

    Returns
    -------
    V : ndarray, shape (n_channels, n_channels)
        The diagonalizer.
    D : ndarray, shape (n_epochs, n_channels, n_channels)
        The set of quasi diagonal matrices.

    References
    ----------
    .. [1] Pham, Dinh Tuan. "Joint approximate diagonalization of positive
           definite Hermitian matrices." SIAM Journal on Matrix Analysis and
           Applications 22, no. 4 (2001): 1136-1152.

    """
    # Adapted from http://github.com/alexandrebarachant/pyRiemann
    n_epochs = X.shape[0]

    # Reshape input matrix
    A = np.concatenate(X, axis=0).T

    # Init variables
    n_times, n_m = A.shape
    V = np.eye(n_times)
    epsilon = n_times * (n_times - 1) * eps

    for it in range(max_iter):
        decr = 0
        for ii in range(1, n_times):
            for jj in range(ii):
                Ii = np.arange(ii, n_m, n_times)
                Ij = np.arange(jj, n_m, n_times)

                c1 = A[ii, Ii]
                c2 = A[jj, Ij]

                g12 = np.mean(A[ii, Ij] / c1)
                g21 = np.mean(A[ii, Ij] / c2)

                omega21 = np.mean(c1 / c2)
                omega12 = np.mean(c2 / c1)
                omega = np.sqrt(omega12 * omega21)

                tmp = np.sqrt(omega21 / omega12)
                tmp1 = (tmp * g12 + g21) / (omega + 1)
                tmp2 = (tmp * g12 - g21) / max(omega - 1, 1e-9)

                h12 = tmp1 + tmp2
                h21 = np.conj((tmp1 - tmp2) / tmp)

                decr += n_epochs * (g12 * np.conj(h12) + g21 * h21) / 2.0

                tmp = 1 + 1.0j * 0.5 * np.imag(h12 * h21)
                tmp = np.real(tmp + np.sqrt(tmp**2 - h12 * h21))
                tau = np.array([[1, -h12 / tmp], [-h21 / tmp, 1]])

                A[[ii, jj], :] = np.dot(tau, A[[ii, jj], :])
                tmp = np.c_[A[:, Ii], A[:, Ij]]
                tmp = np.reshape(tmp, (n_times * n_epochs, 2), order="F")
                tmp = np.dot(tmp, tau.T)

                tmp = np.reshape(tmp, (n_times, n_epochs * 2), order="F")
                A[:, Ii] = tmp[:, :n_epochs]
                A[:, Ij] = tmp[:, n_epochs:]
                V[[ii, jj], :] = np.dot(tau, V[[ii, jj], :])
        if decr < epsilon:
            break
    D = np.reshape(A, (n_times, -1, n_times)).transpose(1, 0, 2)
    return V, D



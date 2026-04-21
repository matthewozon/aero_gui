"""
chambolle_pock_alg_2.py -- Chambolle-Pock Algorithm 2 for Gaussian noise.

Solves::

    n_hat = argmin_n  f(A n) + g(n)

with

* ``f(z) = ||z - y_tilde||^2``, where ``y_tilde = E y`` is the
  preconditioned data, and
* ``g = indicator of the positive quadrant (R^+)^N``.

The stacked operator is ``A = [E K; J]`` with ``K`` the measurement
model, ``E`` a diagonal preconditioner, and ``J`` a smoothness prior.

Reference
---------
Chambolle, A. and Pock, T. (2011).  "A First-Order Primal-Dual Algorithm
for Convex Problems with Applications to Imaging."  *Journal of
Mathematical Imaging and Vision* 40, 120-145.  (Algorithm 2.)

Transliterated from chambolle_pock_alg_2.jl (AeroInv.jl) by Matthew Ozon.
Copyright (C) 2018-2019  Matthew Ozon  (MIT "Expat" Licence)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Proximal operators
# ---------------------------------------------------------------------------

def _prox_F_conj(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float,
    M: int,
    Ndim: int,
) -> np.ndarray:
    """Proximal of the conjugate of the quadratic cost f.

    For the first M entries (data space)::

        u[i] = (x[i] - sigma * y[i]) / (1 + 0.5 * sigma)

    For the remaining entries (regularisation space)::

        u[i] = x[i] / (1 + 0.5 * sigma)

    Parameters
    ----------
    x : ndarray, shape (Ndim,)
        Current dual variable.
    y : ndarray, shape (M,)
        Preconditioned measurement data.
    sigma : float
        Dual step size.
    M : int
        Number of data channels.
    Ndim : int
        Total dual dimension (data + regularisation).

    Returns
    -------
    ndarray, shape (Ndim,)
    """
    u = np.empty(Ndim)
    u[:M] = (x[:M] - sigma * y) / (1.0 + 0.5 * sigma)
    u[M:] = x[M:] / (1.0 + 0.5 * sigma)
    return u


def _prox_G(x: np.ndarray) -> np.ndarray:
    """Proximal of the positivity-constraint indicator: clips negative values."""
    return x * (x >= 0.0)


# ---------------------------------------------------------------------------
# Core Chambolle-Pock iteration (Algorithm 2)
# ---------------------------------------------------------------------------

def _alg2_chpo(
    x0: np.ndarray,
    y: np.ndarray,
    A: np.ndarray,
    tau0: float = 1.0e15,
    Niter: int = 1000,
):
    """Run Chambolle-Pock Algorithm 2 for the Gaussian noise model.

    Parameters
    ----------
    x0 : ndarray, shape (N,)
        Initial primal point.
    y : ndarray, shape (M,)
        Preconditioned measurement vector.
    A : ndarray, shape (Ndim, N)
        Stacked operator [E K; J].
    tau0 : float, optional
        Initial primal step size (default 1e15).
    Niter : int, optional
        Number of iterations (default 1000).

    Returns
    -------
    xn : ndarray, shape (N,)
        Final primal state (estimated density).
    sn : ndarray, shape (Ndim,)
        Final dual state.
    """
    M = len(y)
    Ndim, N = A.shape

    L = np.linalg.norm(A, ord=2)          # spectral norm (Lipschitz constant)
    sigma0 = 1.0 / (tau0 * L ** 2)
    dxhx0 = np.sqrt(len(x0)) * 1.0e4     # rough upper bound on ||x* - x0||
    gamma = 100.0 * dxhx0 / tau0

    sigman = sigma0
    taun = tau0
    xn = x0.copy()
    sn = np.zeros(Ndim)
    xn_pre = x0.copy()
    xn_bar = x0.copy()

    for _ in range(Niter):
        # Dual step
        sn = _prox_F_conj(sn + sigman * A @ xn_bar, y, sigman, M, Ndim)
        # Primal step
        xn_pre = xn.copy()
        xn = _prox_G(xn - taun * A.T @ sn)
        # Adaptive step-size update
        thetan = 1.0 / np.sqrt(1.0 + 2.0 * gamma * taun)
        taun *= thetan
        sigman /= thetan
        # Extrapolation
        xn_bar = xn + thetan * (xn - xn_pre)

    return xn, sn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def aeroinv_reg(
    y: np.ndarray,
    K: np.ndarray,
    tau00: float = 1.0e15,
    Niter: int = 1000,
):
    """Regularised inversion with a positivity constraint (Gaussian noise).

    Builds the stacked operator ``A = [diag(W) K; J]`` and the
    preconditioned data ``E y``, then runs :func:`_alg2_chpo` for each
    time step, using the previous estimate as a warm start.

    Parameters
    ----------
    y : ndarray, shape (n_data, Nt)
        Time series of measurements.
    K : ndarray, shape (n_data, n_model)
        Measurement model matrix.
    tau00 : float, optional
        Initial primal step size (default 1e15).
    Niter : int, optional
        Number of Chambolle-Pock iterations per time step (default 1000).

    Returns
    -------
    F_ALL : ndarray, shape (Nt, n_model)
        Estimated density for every time step.
    S_ALL : ndarray, shape (Nt, n_data + n_model)
        Dual variable for every time step.

    Raises
    ------
    ValueError
        If the model and data dimensions do not match.
    """
    n_data_dim, Nt = y.shape
    nK, n_model_dim = K.shape
    if n_data_dim != nK:
        raise ValueError(
            "aeroinv_reg: measurement model and data dimension do not match"
        )

    # Second-order finite-difference regularisation operator
    J = 1.0e-3 * (
        np.diag(-2.0 * np.ones(n_model_dim))
        + np.diag(np.ones(n_model_dim - 1), k=1)
        + np.diag(np.ones(n_model_dim - 1), k=-1)
    )
    J[0, 0] = 0.0
    J[0, 1] = 0.0
    J[-1, -1] = 0.0
    J[-1, -2] = 0.0

    # Data-driven diagonal preconditioner (column sums of K, normalised)
    W_prob = np.sum(K, axis=1)             # (n_data_dim,)
    W_prob = 1.0 / W_prob
    W_prob = n_data_dim * W_prob / np.sum(W_prob)

    # Stacked operator and preconditioned data
    A = np.vstack([np.diag(W_prob) @ K, J])   # (n_data_dim + n_model_dim, n_model_dim)
    En = W_prob[:, np.newaxis] * y             # (n_data_dim, Nt)

    F_ALL = np.zeros((Nt, n_model_dim))
    S_ALL = np.zeros((Nt, n_data_dim + n_model_dim))

    for idx in range(Nt):
        x0 = np.ones(n_model_dim) if idx == 0 else F_ALL[idx - 1, :]
        F_ALL[idx, :], S_ALL[idx, :] = _alg2_chpo(
            x0, En[:, idx], A, tau0=tau00, Niter=Niter
        )

    return F_ALL, S_ALL

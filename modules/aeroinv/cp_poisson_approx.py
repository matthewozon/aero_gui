"""
cp_poisson_approx.py -- Poisson inversion with approximate (diagonal) Hessian.

Implements the Newton-like outer loop from Ozon et al. (2020) that uses a
diagonal Hessian approximation to form a quadratic sub-problem which is then
solved by a Chambolle-Pock inner solver.

The outer iteration refines the density estimate ``f`` by:

1. Computing the gradient ``∇phi`` and a diagonal Hessian approximation
   ``D_k`` of the Poisson log-likelihood.
2. Forming the proximal / Newton step target ``q = f - D_k^{-1} ∇phi``.
3. Solving the positivity-constrained sub-problem with Chambolle-Pock.
4. Performing a simple backtracking line-search that scales ``D_k`` upward
   if the objective did not decrease.

Transliterated from cp_poisson_approx.jl (AeroInv.jl) by Matthew Ozon.
Copyright (C) 2018-2019  Matthew Ozon  (MIT "Expat" Licence)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Proximal operators for the approximate (linearised-Poisson) sub-problem
# ---------------------------------------------------------------------------

def _prox_F_conj_pois_ap(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float,
    alpha_k: float,
    Dk: np.ndarray,
    M: int,
) -> np.ndarray:
    """Proximal of the approximate conjugate cost (diagonal Hessian version).

    Data space::

        u[i] = (x[i] - sigma * y[i]) / (1 + sigma * alpha_k / (2 * Dk[i]))

    Regularisation space::

        u[i] = x[i] / (1 + 0.5 * sigma * alpha_k)

    Parameters
    ----------
    x : ndarray, shape (M + Nr,)
        Current dual iterate.
    y : ndarray, shape (M,)
        Data (target for the data-space entries).
    sigma : float
        Dual step size.
    alpha_k : float
        Weighted mean Hessian (scalar curvature estimate).
    Dk : ndarray, shape (M,)
        Diagonal of the Hessian approximation (data-space part only).
    M : int
        Number of data channels.

    Returns
    -------
    ndarray, shape (M + Nr,)
    """
    u = np.empty_like(x)
    # Data space
    u[:M] = (x[:M] - sigma * y) / (1.0 + sigma * alpha_k / (2.0 * Dk))
    # Regularisation space
    u[M:] = x[M:] / (1.0 + 0.5 * sigma * alpha_k)
    return u


def _prox_G_pois_ap(x: np.ndarray) -> np.ndarray:
    """Proximal of the positivity indicator: clips negative entries to zero."""
    return x * (x >= 0.0)


def _alg2_chpo_pois_ap(
    x0: np.ndarray,
    s0: np.ndarray,
    y: np.ndarray,
    alpha_k: float,
    Dk: np.ndarray,
    A: np.ndarray,
    tau0: float = 1.0,
    Niter: int = 100,
):
    """Chambolle-Pock inner solver for the approximate Poisson sub-problem.

    Parameters
    ----------
    x0 : ndarray, shape (N,)
        Initial primal point.
    s0 : ndarray, shape (Ndim,)
        Initial dual point.
    y : ndarray, shape (M,)
        Sub-problem right-hand side (Newton step target ``q``).
    alpha_k : float
        Scalar curvature estimate.
    Dk : ndarray, shape (M,)
        Diagonal Hessian (data-space entries only).
    A : ndarray, shape (Ndim, N)
        Stacked operator [I; J].
    tau0 : float, optional
        Initial primal step size (default 1.0).
    Niter : int, optional
        Number of iterations (default 100).

    Returns
    -------
    xn : ndarray, shape (N,)
    sn : ndarray, shape (Ndim,)
    """
    M = len(y)
    N = len(x0)

    L = np.linalg.norm(A, ord=2)
    sigma0 = 1.0 / (tau0 * L ** 2)
    dxhx0 = np.sqrt(N) * 1.0e4
    gamma = 100.0 * dxhx0 / tau0

    sigman = sigma0
    taun = tau0
    xn = x0.copy()
    sn = s0.copy()
    xn_pre = x0.copy()
    xn_bar = x0.copy()

    for _ in range(Niter):
        sn = _prox_F_conj_pois_ap(sn + sigman * A @ xn_bar, y, sigman, alpha_k, Dk, M)
        xn_pre = xn.copy()
        xn = _prox_G_pois_ap(xn - taun * A.T @ sn)
        thetan = 1.0 / np.sqrt(1.0 + 2.0 * gamma * taun)
        taun *= thetan
        sigman /= thetan
        xn_bar = xn + thetan * (xn - xn_pre)

    return xn, sn


# ---------------------------------------------------------------------------
# Poisson log-likelihood, gradient, and diagonal Hessian
# ---------------------------------------------------------------------------

def phi_pois_ap(
    f: np.ndarray,
    y: np.ndarray,
    H: np.ndarray,
    W: np.ndarray,
    beta_f: float = 0.1,
) -> float:
    """Weighted Poisson negative log-likelihood.

    ::

        phi(f) = sum_i  W[i] * (H f[i] - y[i] * ln(H f[i]))

    A small ``beta_f`` floor prevents ``ln(0)``.

    Parameters
    ----------
    f : ndarray, shape (N,)
        Current density estimate.
    y : ndarray, shape (M,)
        Observed counts.
    H : ndarray, shape (M, N)
        Measurement operator.
    W : ndarray, shape (M,)
        Channel-wise weights.
    beta_f : float, optional
        Background regularisation floor (default 0.1).

    Returns
    -------
    float
    """
    f_reg = np.maximum(f, 1.0e-300)
    Hf = H @ f_reg
    return np.sum(W * (Hf - y * np.log(Hf)))


def grad_phi_pois_ap(
    f: np.ndarray,
    y: np.ndarray,
    H: np.ndarray,
    W: np.ndarray,
    beta_f: float = 0.1,
) -> np.ndarray:
    """Gradient of :func:`phi_pois_ap` with respect to ``f``.

    ::

        ∇phi(f) = H^T W  -  H^T [ W[i] * (y[i]+beta_f) / (H f[i]+beta_f) ]

    Parameters
    ----------
    f : ndarray, shape (N,)
    y : ndarray, shape (M,)
    H : ndarray, shape (M, N)
    W : ndarray, shape (M,)
    beta_f : float, optional

    Returns
    -------
    ndarray, shape (N,)
    """
    y_model = H @ f
    scale = W * (y + beta_f) / (y_model + beta_f)   # (M,)
    return H.T @ W - H.T @ scale


def hess_phi_diag_pois_ap(
    f: np.ndarray,
    y: np.ndarray,
    H: np.ndarray,
    W: np.ndarray,
    beta_f: float = 0.1,
) -> np.ndarray:
    """Diagonal of the Hessian of :func:`phi_pois_ap`.

    ::

        diag(H^2)^T  [ W * (y + beta_f) / (H f + beta_f)^2 ]

    where ``H^2`` denotes element-wise squaring.

    Parameters
    ----------
    f : ndarray, shape (N,)
    y : ndarray, shape (M,)
    H : ndarray, shape (M, N)
    W : ndarray, shape (M,)
    beta_f : float, optional

    Returns
    -------
    ndarray, shape (N,)
    """
    y_model = H @ f
    v = W * (y + beta_f) / (y_model + beta_f) ** 2   # (M,)
    return v @ (H ** 2)                                # (N,)


# ---------------------------------------------------------------------------
# Newton outer iteration
# ---------------------------------------------------------------------------

def ozon2020_iteration(
    f: np.ndarray,
    delta_f: np.ndarray,
    y: np.ndarray,
    H: np.ndarray,
    J: np.ndarray,
    W: np.ndarray,
    beta_f: float = 0.1,
    max_iter: int = 5000,
    verbose: bool = False,
):
    """One Newton-like outer iteration with a Chambolle-Pock inner solver.

    Builds the quadratic model ``q = f - D_k^{-1} ∇phi(f)``, solves the
    positivity-constrained sub-problem, and performs a simple backtracking
    line search that scales the curvature ``D_k`` upward if the objective
    did not decrease.

    Parameters
    ----------
    f : ndarray, shape (N,)
        Current density estimate.
    delta_f : ndarray, shape (N,)
        Previous density update (used to estimate ``alpha_k``).
    y : ndarray, shape (M,)
        Observed Poisson counts.
    H : ndarray, shape (M, N)
        Measurement operator.
    J : ndarray, shape (Nr, N)
        Regularisation operator.
    W : ndarray, shape (M,)
        Channel-wise weights.
    beta_f : float, optional
        Log-floor regularisation (default 0.1).
    max_iter : int, optional
        Maximum inner iterations (default 5000).
    verbose : bool, optional
        Unused (kept for API compatibility).

    Returns
    -------
    ff : ndarray, shape (N,)
        Updated density estimate.
    delta_F : ndarray, shape (N,)
        Update step ``ff - f``.
    phi_val : float
        Objective value ``phi(ff)``.
    dPhi : ndarray, shape (N,)
        Gradient at ``f``.
    hPhi : ndarray, shape (N, N)
        Diagonal Hessian as a full diagonal matrix (for API compatibility).
    """
    N_data, N_model = H.shape

    # Gradient and diagonal Hessian at current estimate
    dPhi = grad_phi_pois_ap(f, y, H, W, beta_f=beta_f)
    hPhi_diag = hess_phi_diag_pois_ap(f, y, H, W, beta_f=beta_f)

    sum_df2 = np.sum(delta_f ** 2)
    if sum_df2 > 0.0:
        alpha_k = np.dot(hPhi_diag, delta_f ** 2) / sum_df2
    else:
        alpha_k = np.mean(hPhi_diag)

    # Newton step target
    q = f - dPhi / hPhi_diag    # element-wise (diagonal inverse)

    # Stacked operator: [I; J]
    A = np.vstack([np.eye(N_model), J])
    s0 = np.zeros(A.shape[0])

    # Inner Chambolle-Pock solve
    ff, ss = _alg2_chpo_pois_ap(f, s0, q, alpha_k, hPhi_diag, A, tau0=1.0e3, Niter=max_iter)
    delta_F = ff - f

    # Backtracking line search: scale D_k if objective did not decrease
    eta = 2.0
    sigma = 0.5
    itit = 0
    phi_f = phi_pois_ap(f, y, H, W, beta_f=beta_f)
    while (
        phi_pois_ap(ff, y, H, W, beta_f=beta_f)
        > phi_f - 0.5 * sigma * alpha_k * np.sum(delta_F ** 2)
        and eta ** itit <= 10.0
    ):
        hPhi_diag = eta * hPhi_diag
        sum_df2 = np.sum(delta_f ** 2)
        if sum_df2 > 0.0:
            alpha_k = np.dot(hPhi_diag, delta_f ** 2) / sum_df2
        else:
            alpha_k = np.mean(hPhi_diag)

        q = f - dPhi / hPhi_diag
        ff, ss = _alg2_chpo_pois_ap(ff, s0, q, alpha_k, hPhi_diag, A, tau0=1.0e3, Niter=max_iter)
        delta_F = ff - f
        itit += 1

    phi_val = phi_pois_ap(ff, y, H, W, beta_f=beta_f)
    return ff, delta_F, phi_val, dPhi, np.diag(hPhi_diag)


# ---------------------------------------------------------------------------
# Outer Newton loop
# ---------------------------------------------------------------------------

def ozon2020(
    f: np.ndarray,
    y: np.ndarray,
    H: np.ndarray,
    J: np.ndarray,
    W: np.ndarray,
    beta_f: float = 1.0e-5,
    max_iter: int = 10000,
    N_max: int = 100,
    verbose: bool = False,
    rtol: float = 1.0e-5,
) -> np.ndarray:
    """Newton-like outer loop with Chambolle-Pock inner solver (Ozon 2020).

    Runs up to ``N_max + 1`` calls to :func:`ozon2020_iteration` and
    returns the last finite density estimate.

    Parameters
    ----------
    f : ndarray, shape (N,)
        Initial density estimate (must be positive).
    y : ndarray, shape (M,)
        Observed Poisson counts.
    H : ndarray, shape (M, N)
        Measurement operator.
    J : ndarray, shape (Nr, N)
        Regularisation operator.
    W : ndarray, shape (M,)
        Channel-wise weights.
    beta_f : float, optional
        Log-floor regularisation (default 1e-5).
    max_iter : int, optional
        Maximum inner iterations per outer step (default 10000).
    N_max : int, optional
        Maximum outer Newton iterations (default 100).
    verbose : bool, optional
        Passed to :func:`ozon2020_iteration`.
    rtol : float, optional
        Relative tolerance (currently unused -- kept for API compatibility).

    Returns
    -------
    ndarray, shape (N,)
        Final estimated density.
    """
    n_model = H.shape[1]
    phi_val = np.zeros(N_max + 3)
    F_EST = np.zeros((N_max + 3, n_model))

    f_est = f.copy()
    delta_f = f.copy()

    F_EST[0, :] = f_est
    phi_val[0] = phi_pois_ap(f_est, y, H, W, beta_f=beta_f)

    # First outer iteration (outside the main loop)
    f_est, delta_f, phi_val[1], dPhi, hPhi = ozon2020_iteration(
        f_est, delta_f, y, H, J, W, beta_f=beta_f, max_iter=max_iter
    )
    F_EST[1, :] = f_est

    # Main outer loop (N_max iterations)
    i = 0   # points to the last valid F_EST row (0-indexed)
    for k in range(N_max):
        f_est, delta_f, phi_val_cur, dPhi, hPhi = ozon2020_iteration(
            f_est, delta_f, y, H, J, W,
            beta_f=beta_f, max_iter=max_iter, verbose=verbose
        )
        F_EST[i + 2, :] = f_est
        phi_val[i + 2] = phi_val_cur
        i += 1

        if np.isnan(phi_val_cur):
            # Mark current slot invalid and revert to previous estimate
            F_EST[i + 1, :] = -1.0
            f_est = F_EST[i, :].copy()   # revert to the last good estimate
            i -= 1

    return F_EST[i + 1, :]

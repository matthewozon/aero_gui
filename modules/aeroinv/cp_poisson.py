"""
cp_poisson.py -- Chambolle-Pock for Poisson noise with exact proximal operators.

Solves::

    x_hat = argmin_x  F(A x) + G(x)

with the Poisson-regularised cost::

    F(z) = sum_{i=1}^{M} W[i] * (z[i] - y[i] * ln z[i])
           + sum_{i=M+1}^{end} z[i]^2

and ``G`` the indicator of the positive quadrant (R^+)^N.

Reference
---------
Chambolle, A. and Pock, T. (2011).  "A First-Order Primal-Dual Algorithm
for Convex Problems with Applications to Imaging."  *Journal of
Mathematical Imaging and Vision* 40, 120-145.

Transliterated from cp_poisson.jl (AeroInv.jl) by Matthew Ozon.
Copyright (C) 2018-2019  Matthew Ozon  (MIT "Expat" Licence)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Cost function and its conjugate
# ---------------------------------------------------------------------------

def F_poisson(
    z: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
) -> float:
    """Weighted Poisson negative log-likelihood with quadratic regularisation.

    ::

        F(z) = sum_{i: z[i]>0}  W[i] * (z[i] - y[i] * ln z[i])
               + sum_{i=M+1}^{end}  z[i]^2

    Parameters
    ----------
    z : ndarray, shape (M + N,)
        Argument: first M entries are modelled Poisson means, remaining N
        are regularisation entries.
    y : ndarray, shape (M,)
        Observed Poisson counts.
    W : ndarray, shape (M,)
        Channel-wise weights (preconditioner diagonal).

    Returns
    -------
    float
    """
    M = len(y)
    idx = np.where(z[:M] > 0.0)[0]
    return (
        np.sum(W[idx] * (z[idx] - y[idx] * np.log(z[idx])))
        + np.sum(z[M:] ** 2)
    )


def G_poisson(x: np.ndarray) -> float:
    """Indicator of the positive quadrant: 0 if x >= 0, +inf otherwise."""
    return 0.0 if np.all(x >= 0.0) else np.inf


def F_convex_conjugate_poisson(
    u: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
) -> float:
    """Legendre-Fenchel conjugate of :func:`F_poisson`.

    ::

        F*(u) = sum_{i: y[i]>0}  y[i] W[i] (1 + ln(y[i] / (1 - u[i]/W[i])))
                + (1/4) sum_{i=M+1}^{end}  u[i]^2

    Parameters
    ----------
    u : ndarray, shape (M + N,)
        Dual argument.
    y : ndarray, shape (M,)
        Observed Poisson counts.
    W : ndarray, shape (M,)
        Channel-wise weights.

    Returns
    -------
    float
    """
    M = len(y)
    idx = np.where(y > 0.0)[0]
    return (
        np.sum(y[idx] * W[idx] * (1.0 + np.log(y[idx] / (1.0 - u[idx] / W[idx]))))
        + 0.25 * np.sum(u[M:] ** 2)
    )


# ---------------------------------------------------------------------------
# Proximal operators
# ---------------------------------------------------------------------------

def prox_F_conj_poisson(
    x: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Proximal operator of the conjugate of :func:`F_poisson`.

    Solves ``argmin_u { sigma * F*(u) + (1/2) ||u - x||^2 }``.

    For channels where ``y[i] > 0`` the minimiser is found via the
    quadratic formula.  The result is clipped to the domain ``u[i] <= W[i]``.
    For channels where ``y[i] == 0`` the minimiser is simply ``x[i]``.
    For the regularisation entries the closed-form solution is
    ``u[i] = x[i] / (1 + 0.5 * sigma)``.

    Parameters
    ----------
    x : ndarray, shape (M + N,)
        Current dual iterate.
    y : ndarray, shape (M,)
        Observed Poisson counts.
    W : ndarray, shape (M,)
        Channel-wise weights.
    sigma : float
        Dual step size.

    Returns
    -------
    ndarray, shape (M + N,)
    """
    M = len(y)
    N = len(x) - M
    u = np.empty(M + N)

    for i in range(M):
        if y[i] == 0.0:
            u[i] = x[i]
        else:
            disc = (x[i] + W[i]) ** 2 - 4.0 * W[i] * (x[i] - sigma * y[i])
            if disc > 0.0:
                # Prefer the larger root; fall back to the smaller root if
                # the larger one violates the domain u[i] < W[i]
                u[i] = 0.5 * (x[i] + W[i] + np.sqrt(disc))
                if u[i] >= W[i]:
                    u[i] = 0.5 * (x[i] + W[i] - np.sqrt(disc))
            else:
                u[i] = 0.5 * (x[i] + W[i])
        # Hard clip to domain boundary
        u[i] = min(u[i], W[i])

    # Regularisation space: proximal of quadratic
    u[M:] = x[M:] / (1.0 + 0.5 * sigma)
    return u


def prox_G_poisson(x: np.ndarray) -> np.ndarray:
    """Proximal of the positivity indicator: clips negative entries to zero."""
    return x * (x >= 0.0)


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

def alg2_cp_poisson(
    x0: np.ndarray,
    y: np.ndarray,
    A: np.ndarray,
    W: np.ndarray,
    W_stop: np.ndarray,
    tau0: float = 1.0,
    Niter: int = 100,
    r_n_tol: float = 1.0e-6,
    r_y_tol: float = 0.5,
):
    """Chambolle-Pock Algorithm 2 for the Poisson inverse problem.

    Minimises ``F(A x) + G(x)`` using primal-dual iterations with adaptive
    step-size scheduling.  Iteration stops early when both a data-fidelity
    criterion and a relative step-size criterion are satisfied.

    Parameters
    ----------
    x0 : ndarray, shape (N,)
        Initial density estimate.
    y : ndarray, shape (M,)
        Observed Poisson counts.
    A : ndarray, shape (Ndim, N)
        Stacked measurement + regularisation operator.
    W : ndarray, shape (M,)
        Channel-wise preconditioner weights.
    W_stop : ndarray, shape (N,)
        Weights used in the relative-step stopping criterion.
    tau0 : float, optional
        Initial primal step size (default 1.0).
    Niter : int, optional
        Maximum number of iterations (default 100).
    r_n_tol : float, optional
        Tolerance on weighted relative step ``||W_stop (x_n - x_{n-1})||
        / ||W_stop x_n||`` (default 1e-6).
    r_y_tol : float, optional
        Tolerance on median relative data residual (default 0.5).

    Returns
    -------
    xn : ndarray, shape (N,)
        Final density estimate.
    sn : ndarray, shape (Ndim,)
        Final dual variable.
    taun : float
        Final primal step size.
    X_ALL : ndarray, shape (N, Niter+1)
        Primal trajectory (columns are iterates 0 .. N_last).
    S_ALL : ndarray, shape (Ndim, Niter+1)
        Dual trajectory.
    T_ALL : ndarray, shape (Niter+1,)
        Primal step-size trajectory.
    N_last : int
        Iteration index at which the stopping criterion was met.

    Raises
    ------
    ValueError
        If dimensions are inconsistent.
    """
    N = len(x0)
    if N != A.shape[1]:
        raise ValueError(
            "alg2_cp_poisson: operator A and state x0 have incompatible dimensions"
        )
    if len(W) != len(y):
        raise ValueError(
            "alg2_cp_poisson: data y and weight vector W have incompatible dimensions"
        )

    L = np.linalg.norm(A, ord=2)           # spectral norm
    sigma0 = 1.0 / (tau0 * L ** 2)
    dxhx0 = np.sqrt(N) * 1.4e4             # rough upper bound on ||x* - x0||
    gamma = 100.0 * dxhx0 / tau0

    Ndim = A.shape[0]
    X_ALL = np.zeros((N, Niter + 1))
    S_ALL = np.zeros((Ndim, Niter + 1))
    T_ALL = np.zeros(Niter + 1)

    sigman = sigma0
    taun = tau0
    xn = x0.copy()
    sn = np.zeros(Ndim)
    xn_pre = x0.copy()
    xn_bar = x0.copy()
    X_ALL[:, 0] = xn
    S_ALL[:, 0] = sn
    T_ALL[0] = taun

    idx_pos = np.where(y > 0.0)[0]
    idx_zero = np.where(y == 0.0)[0]
    Y_rel = np.zeros(len(y))
    N_last = Niter

    for i in range(Niter):
        # Dual step
        sn = prox_F_conj_poisson(sn + sigman * A @ xn_bar, y, W, sigman)
        # Primal step
        xn_pre = xn.copy()
        xn = prox_G_poisson(xn - taun * A.T @ sn)
        # Adaptive step-size update
        thetan = 1.0 / np.sqrt(1.0 + 2.0 * gamma * taun)
        taun *= thetan
        sigman /= thetan
        # Extrapolation
        xn_bar = xn + thetan * (xn - xn_pre)

        # --- Stopping criteria ---
        # Relative data residual
        Axn = A @ xn
        Y_rel[idx_pos] = np.abs(y[idx_pos] - Axn[idx_pos]) / y[idx_pos]
        Y_rel[idx_zero] = np.abs(Axn[idx_zero])

        # Weighted relative step size
        dX_W = W_stop * (xn - xn_pre)
        normdX_W = np.sum(dX_W ** 2)
        normX_W = np.sum((W_stop * xn) ** 2)

        if (
            np.median(Y_rel) <= r_y_tol
            and normX_W > 0.0
            and np.sqrt(normdX_W / normX_W) <= r_n_tol
        ):
            N_last = i + 1   # 1-based iteration count
            break

        X_ALL[:, i + 1] = xn
        S_ALL[:, i + 1] = sn
        T_ALL[i + 1] = taun

    return xn, sn, taun, X_ALL, S_ALL, T_ALL, N_last

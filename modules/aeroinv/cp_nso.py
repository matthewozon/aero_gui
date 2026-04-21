"""
cp_nso.py -- Chambolle-Pock with Gaussian noise and null-space optimisation.

Instead of solving directly for the density ``n``, the algorithm solves for
the null-space coordinates ``x`` of the system, using the decomposition::

    n = xr + V0 x

where ``xr`` is a particular solution of the unconstrained problem and
``V0`` is a basis for the null space of the measurement operator.  The
positivity constraint ``n >= 0`` translates to ``V0 x + xr >= 0``.

Solves::

    x_hat = argmin_x  F(A x) + G(x)

with the Gaussian cost::

    F(z) = sum_i (z[i] - y[i])^2

and ``G`` the indicator of the feasible set ``{x : V0 x + xr >= 0}``.

Reference
---------
Chambolle, A. and Pock, T. (2011).  "A First-Order Primal-Dual Algorithm
for Convex Problems with Applications to Imaging."  *Journal of
Mathematical Imaging and Vision* 40, 120-145.

Transliterated from cp_nso.jl (AeroInv.jl) by Matthew Ozon.
Copyright (C) 2018-2019  Matthew Ozon  (MIT "Expat" Licence)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Cost function and its conjugate
# ---------------------------------------------------------------------------

def F_nso(z: np.ndarray, y: np.ndarray) -> float:
    """Quadratic data fidelity: ``F(z) = sum_i (z[i] - y[i])^2``."""
    return np.sum((z - y) ** 2)


def F_convex_conjugate_nso(u: np.ndarray, y: np.ndarray) -> float:
    """Legendre-Fenchel conjugate of :func:`F_nso`.

    ::

        F*(u) = (1/4) sum_i u[i]^2 + sum_i u[i] y[i]
    """
    return 0.25 * np.sum(u ** 2) + np.sum(u * y)


# ---------------------------------------------------------------------------
# Proximal operators
# ---------------------------------------------------------------------------

def prox_F_conj_nso(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Proximal of the conjugate of :func:`F_nso`.

    Closed-form solution for the quadratic case::

        u[i] = (x[i] - sigma * y[i]) / (1 + 0.5 * sigma)

    Parameters
    ----------
    x : ndarray, shape (M,)
        Current dual variable.
    y : ndarray, shape (M,)
        Observed data.
    sigma : float
        Dual step size.

    Returns
    -------
    ndarray, shape (M,)
    """
    return (x - sigma * y) / (1.0 + 0.5 * sigma)


def G_nso(
    x: np.ndarray,
    V0: np.ndarray,
    xr: np.ndarray,
) -> float:
    """Indicator of the feasible set ``{x : V0 x + xr >= 0}``.

    Returns 0 if the constraint is satisfied, +inf otherwise.

    Parameters
    ----------
    x : ndarray, shape (K,)
        Null-space parameter vector.
    V0 : ndarray, shape (N, K)
        Null-space basis matrix.
    xr : ndarray, shape (N,)
        Particular solution to the unconstrained problem.

    Returns
    -------
    float
    """
    return 0.0 if np.all(V0 @ x + xr >= 0.0) else np.inf


def prox_G_nso(
    x: np.ndarray,
    V0: np.ndarray,
    xr: np.ndarray,
) -> np.ndarray:
    """Proximal of :func:`G_nso` via projection onto the feasible set.

    When no positivity constraints are violated the input ``x`` is returned
    unchanged.  Otherwise the function:

    1. Finds the subset of violated constraints.
    2. Computes the least-squares point ``x_ls`` at their intersection
       (closest to the origin).
    3. Finds the null space of that constraint subset via a full SVD.
    4. Projects ``x - x_ls`` onto the null space and adds ``x_ls``.

    Parameters
    ----------
    x : ndarray, shape (K,)
        Current null-space iterate.
    V0 : ndarray, shape (N, K)
        Null-space basis.
    xr : ndarray, shape (N,)
        Particular solution.

    Returns
    -------
    ndarray, shape (K,)

    Raises
    ------
    ValueError
        If ``V0.shape[0] != len(xr)``.
    """
    N, K = V0.shape
    if N != len(xr):
        raise ValueError("prox_G_nso: dimension mismatch (V0 rows != len(xr))")

    # Identify violated positivity constraints
    idx_nn = np.where(V0 @ x + xr < 0.0)[0]

    if len(idx_nn) == 0:
        # All constraints satisfied: no correction needed
        return x.copy()

    # Least-squares point at the intersection of violated hyperplanes
    V0_sub = V0[idx_nn, :]                              # (p, K)
    V0V0T = V0_sub @ V0_sub.T                           # (p, p)
    x_ls = -V0_sub.T @ np.linalg.solve(V0V0T, xr[idx_nn])   # (K,)

    # Null space basis of V0_sub via full SVD
    # np.linalg.svd returns Vh (= V^T) with shape (K, K)
    p = len(idx_nn)
    _, _, Vh = np.linalg.svd(V0_sub, full_matrices=True)
    W_k = Vh[p:, :].T    # (K, K-p) -- right singular vectors for zero singular values

    # Project (x - x_ls) onto the null space
    # W_k columns are orthonormal, so inv(W_k^T W_k) = I
    eta_k = W_k.T @ (x - x_ls)               # (K-p,)
    return x_ls + W_k @ eta_k                 # (K,)


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

def alg2_chpo_nso(
    x0: np.ndarray,
    y: np.ndarray,
    A: np.ndarray,
    V0: np.ndarray,
    xr: np.ndarray,
    W_stop: np.ndarray,
    tau0: float = 1.0e15,
    Niter: int = 1000,
    r_n_tol: float = 1.0e-6,
):
    """Chambolle-Pock Algorithm 2 with null-space optimisation (NSO).

    Minimises ``F(A x) + G(x)`` where ``G`` enforces ``V0 x + xr >= 0``.

    Parameters
    ----------
    x0 : ndarray, shape (K,)
        Initial null-space parameter.
    y : ndarray, shape (M,)
        Preconditioned measurement vector.
    A : ndarray, shape (Ndim, K)
        Stacked measurement + regularisation operator (in the null space).
    V0 : ndarray, shape (N, K)
        Null-space basis matrix.
    xr : ndarray, shape (N,)
        Particular solution (physical-space offset).
    W_stop : ndarray, shape (K,)
        Weights for the weighted relative-step stopping criterion.
    tau0 : float, optional
        Initial primal step size (default 1e15).
    Niter : int, optional
        Maximum number of iterations (default 1000).
    r_n_tol : float, optional
        Stopping tolerance on ``||W_stop (x_n - x_{n-1})|| / ||W_stop x_n||``
        (default 1e-6).

    Returns
    -------
    xn : ndarray, shape (K,)
        Final null-space parameter.
    sn : ndarray, shape (Ndim,)
        Final dual variable.
    taun : float
        Final primal step size.
    X_ALL : ndarray, shape (K, Niter+1)
        Primal trajectory.
    S_ALL : ndarray, shape (Ndim, Niter+1)
        Dual trajectory.
    T_ALL : ndarray, shape (Niter+1,)
        Step-size trajectory.
    N_last : int
        Iteration index at which the stopping criterion was met.
    """
    Ndim, K = A.shape

    L = np.linalg.norm(A, ord=2)
    sigma0 = 1.0 / (tau0 * L ** 2)
    dxhx0 = np.sqrt(len(x0)) * 1.0e4
    gamma = 100.0 * dxhx0 / tau0

    X_ALL = np.zeros((len(x0), Niter + 1))
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
    N_last = Niter

    for i in range(Niter):
        # Dual step (quadratic proximal)
        sn = prox_F_conj_nso(sn + sigman * A @ xn_bar, y, sigman)
        # Primal step (null-space projection)
        xn_pre = xn.copy()
        xn = prox_G_nso(xn - taun * A.T @ sn, V0, xr)
        # Adaptive step-size update
        thetan = 1.0 / np.sqrt(1.0 + 2.0 * gamma * taun)
        taun *= thetan
        sigman /= thetan
        # Extrapolation
        xn_bar = xn + thetan * (xn - xn_pre)

        # Store iterate
        X_ALL[:, i + 1] = xn
        S_ALL[:, i + 1] = sn
        T_ALL[i + 1] = taun

        # Stopping criterion: weighted relative step size
        normX_W = np.sum((W_stop * xn) ** 2)
        if normX_W > 0.0:
            rel_step = np.sqrt(np.sum((W_stop * (xn - xn_pre)) ** 2) / normX_W)
            if rel_step <= r_n_tol:
                N_last = i + 1   # 1-based iteration count
                break

    return xn, sn, taun, X_ALL, S_ALL, T_ALL, N_last

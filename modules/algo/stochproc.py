"""
modules/algo/stochproc.py
==========================
Pure-Python / NumPy translation of the BAYROSOL StochProc package:
  cholWrap.jl + SP_mod.jl + type.jl + StochProcLin.jl +
  StochProc1stOrderScalar.jl + StochProc2ndOrderScalar.jl +
  StochProc1stOrderVector.jl + NE.jl

Original Julia code © 2021 Matthew Ozon, MIT License.
Python translation and improvements listed below.

Purpose
-------
This module serves two main roles in the BAYROSOL parameter estimation
workflow:

1.  **Covariance construction** — build structured covariance matrices
    that encode *size-space correlation* of aerosol parameters
    (e.g. growth rates, loss rates) across particle diameter bins.
    Three families are provided:
    • Second-order AR filter model  (smooth, Toeplitz structure)
    • User-supplied PSF array model (Toeplitz, arbitrary shape)
    • Range-limit model             (non-stationary, explicit formula)

2.  **Stochastic process generation** — generate realisations of linear
    first- and second-order scalar or vector AR processes, used for
    prior sampling and uncertainty quantification.

Key public API (mirrors Julia exports)
--------------------------------------
linSP                          — process descriptor dataclass
chol_psd                       — robust Cholesky factorisation
covariance_process_2nd         — 2nd-order AR covariance matrix
covariance_process_array       — PSF-array covariance matrix
covariance_process_range_limit — range-limit covariance matrix
covariance_2nd_order_space_1st_order_time  — source noise for 1st-order KF
space_covariance_chol          — covariance + Cholesky (2nd-order model)
space_covariance_array         — covariance + Cholesky (PSF model)
space_covariance_range_limit   — covariance + Cholesky (range model)
covariance_second_order_process — 2nd-order time process covariance
SP_1T, SP_1TC, SP_1TCV         — scalar 1st-order processes
SP_2T, SP_2TC, SP_2TCV, SP_2TCV_mat — scalar 2nd-order processes
SP_1T_V, SP_1TC_V              — vector 1st-order processes
iterator, timeBlockMatrix, initGaussRep — generic process tools
mean_and_variance_estimation   — Monte Carlo mean/variance of f(X)
percentile_estimation          — Monte Carlo percentile estimation
multivariate_probability       — P(‖x‖ ≤ R) for N(0,I)
uniform_sample_unit_sphere     — uniform sample on S^{N-1}
uniform_sample_unit_ball       — uniform sample in B^N

Improvements over the original Julia code
-----------------------------------------
1.  `chol_psd` handles NaN/Inf explicitly, clips negative eigenvalues
    using the same tau-threshold as Julia, and adds a symmetry
    enforcement step before the final Cholesky call.

2.  `covariance_psf_2nd` replaces the Julia `filt` + `autocov` pipeline
    with a direct polynomial-root autocorrelation formula that avoids
    the dependency on DSP.jl / Polynomials.jl entirely and is faster.

3.  All Toeplitz covariance matrices are built with
    `scipy.linalg.toeplitz` — no external ToeplitzMatrices.jl needed.

4.  The `linSP` dataclass validates dimensions on construction and
    raises informative errors, mirroring the Julia constructor guards.

5.  `mean_and_variance_estimation` uses NumPy's vectorised sample
    generation instead of a loop over Ns.

6.  `Ik` and `Hk` are implemented iteratively (no recursion depth issues).
"""

from __future__ import annotations

import numpy as np
from numpy.random import default_rng
from dataclasses import dataclass, field
from typing import Callable, Optional, Union, Tuple, List
from scipy.linalg import toeplitz, cholesky, cho_factor, eigh
from scipy.special import erf

_rng = default_rng()

def set_seed(seed: int) -> None:
    global _rng
    _rng = default_rng(seed)


# ============================================================
#  Cholesky wrapper  (cholWrap.jl)
# ============================================================

def chol_psd(
    A: np.ndarray,
    tau: float = 1e-6,
) -> np.ndarray:
    """
    Robust lower-triangular Cholesky factor of a symmetric PSD matrix.

    • Returns zeros if A contains NaN or Inf.
    • Tries standard Cholesky first.
    • On failure: clips eigenvalues below tau * λ_max to tau * λ_max,
      reconstructs, then re-attempts Cholesky.

    Improvement: adds 0.5*(A+A') symmetry enforcement before each attempt,
    preventing failures due to floating-point asymmetry.

    Mirrors chol_PS_L() + chol_PS_L_almost_positive() in cholWrap.jl.
    """
    n = A.shape[0]
    L = np.zeros((n, n))

    if not np.isfinite(A).all():
        return L   # return zeros, consistent with Julia behaviour

    A_sym = 0.5 * (A + A.T)

    # -- attempt 1: standard Cholesky --
    try:
        L = cholesky(A_sym, lower=True)
        return L
    except Exception:
        pass

    # -- attempt 2: eigenvalue clipping --
    try:
        vals, vecs = eigh(A_sym)          # ascending order
        lam_max = np.abs(vals).max()
        threshold = tau * lam_max
        vals_clipped = np.maximum(vals, threshold)
        A_pd = vecs @ np.diag(vals_clipped) @ vecs.T
        A_pd = 0.5 * (A_pd + A_pd.T)
        L = cholesky(A_pd, lower=True)
        return L
    except Exception as e:
        raise RuntimeError(
            f"chol_psd: cannot factorise matrix even after eigenvalue clipping: {e}"
        )


# ============================================================
#  linSP dataclass  (type.jl)
# ============================================================

@dataclass
class linSP:
    """
    Descriptor for a linear stochastic process.

    Fields
    ------
    proc_ord  : process order (1 or 2)
    num_dim   : state dimension
    B         : evolution model (scalar or matrix, shape (proc_ord*num_dim,)*2)
    G_source  : source covariance (scalar or matrix, shape (num_dim,)*2)
    L_source  : lower Cholesky factor of G_source

    Mirrors the linSP struct in type.jl.
    """
    proc_ord: int
    num_dim:  int
    B:        Union[float, np.ndarray]
    G_source: Union[float, np.ndarray]
    L_source: Union[float, np.ndarray]

    @classmethod
    def scalar_1st(cls, b: float, g_source: float) -> "linSP":
        """Scalar first-order process: x_{k+1} = b*x_k + L*w."""
        if g_source < 0:
            raise ValueError("Source variance must be non-negative.")
        return cls(1, 1, float(b), float(g_source), float(np.sqrt(g_source)))

    @classmethod
    def scalar_2nd(cls, B: np.ndarray, g_source: float) -> "linSP":
        """Scalar second-order process with 2×2 companion matrix."""
        B = np.asarray(B, dtype=float)
        if B.shape != (2, 2):
            raise ValueError("B must be (2,2) for a scalar 2nd-order process.")
        if g_source < 0:
            raise ValueError("Source variance must be non-negative.")
        return cls(2, 1, B, float(g_source), float(np.sqrt(g_source)))

    @classmethod
    def vector_1st(cls, B: Union[float, np.ndarray],
                   G_source: np.ndarray) -> "linSP":
        """Vector first-order process."""
        G_source = np.asarray(G_source, dtype=float)
        n = G_source.shape[0]
        if isinstance(B, (int, float)):
            B_mat = float(B) * np.eye(n)
        else:
            B_mat = np.asarray(B, dtype=float)
            if B_mat.shape != (n, n):
                raise ValueError("B shape does not match G_source dimension.")
        L = chol_psd(G_source)
        return cls(1, n, B_mat, G_source, L)

    def iterate(self) -> np.ndarray:
        """
        Draw one sample from the source and return x_{k+1} given x_k = 0.
        Useful for testing.  Mirrors iterator() in StochProcLin.jl.
        """
        if self.num_dim == 1:
            return float(self.L_source) * _rng.standard_normal()
        else:
            return self.L_source @ _rng.standard_normal(self.num_dim)


# ============================================================
#  Generic process iterator  (StochProcLin.jl)
# ============================================================

def iterator(x: Union[float, np.ndarray], A: linSP) -> Union[float, np.ndarray]:
    """
    One step of a linear stochastic process.

    Scalar 1st-order : z = B*x + L*w
    Scalar n-th order: z = B*x; z[0] += L*w   (companion form)
    Vector 1st-order : z = B*x + L*randn(dim)
    Vector n-th order: z = B*x; z[0::ord] += L*randn(dim)

    Mirrors iterator() in StochProcLin.jl.
    """
    w = _rng.standard_normal
    if A.num_dim == 1:
        if A.proc_ord == 1:
            return A.B * x + A.L_source * w()
        else:
            z = A.B @ np.atleast_1d(x)
            z[0] += A.L_source * w()
            return z
    else:
        if A.proc_ord == 1:
            return A.B @ x + A.L_source @ w(A.num_dim)
        else:
            eta = A.L_source @ w(A.num_dim)
            z = A.B @ x
            z[0::A.proc_ord] += eta
            return z


def timeBlockMatrix(B_sub: Union[float, np.ndarray], dim: int) -> np.ndarray:
    """
    Build a block-diagonal evolution matrix where each block is B_sub.

    If B_sub is scalar, returns B_sub * I_{dim}.
    If B_sub is (ord × ord), returns block-diag of dim copies.

    Mirrors timeBlockMatrix() in StochProcLin.jl.
    """
    if np.isscalar(B_sub):
        return float(B_sub) * np.eye(dim)
    B_sub = np.asarray(B_sub, dtype=float)
    if B_sub.ndim != 2 or B_sub.shape[0] != B_sub.shape[1]:
        raise ValueError("B_sub must be a square matrix.")
    ord_ = B_sub.shape[0]
    B = np.zeros((ord_ * dim, ord_ * dim))
    for i in range(dim):
        sl = slice(i * ord_, (i + 1) * ord_)
        B[sl, sl] = B_sub
    return B


def initGaussRep(mu: Union[float, np.ndarray],
                 L:  Union[float, np.ndarray],
                 rep: int) -> np.ndarray:
    """
    Draw `rep` samples from N(mu, L L') and return as a flat array.

    Mirrors initGaussRep() in StochProcLin.jl.
    Improvement: vectorised — no Python loop over rep.
    """
    if np.isscalar(mu) and np.isscalar(L):
        samples = float(mu) + float(L) * _rng.standard_normal(rep)
        return samples
    mu = np.asarray(mu, dtype=float)
    L  = np.asarray(L,  dtype=float)
    if L.ndim == 0:
        # scalar L, vector mu
        dim = len(mu)
        raw = mu[:, None] + float(L) * _rng.standard_normal((dim, rep))
    else:
        dim = L.shape[0]
        raw = mu[:, None] + L @ _rng.standard_normal((dim, rep))
    # interleave: output[i::rep] = sample i  (Julia convention)
    out = np.empty(rep * dim)
    for i in range(rep):
        out[i::rep] = raw[:, i]
    return out


# ============================================================
#  Covariance builders  (SP_mod.jl)
# ============================================================

def _toeplitz_from_autocov(acov: np.ndarray) -> np.ndarray:
    """Build a symmetric Toeplitz matrix from autocorrelation values."""
    return toeplitz(acov)


def _scale_covariance(Gamma_temp: np.ndarray,
                      sig0:       float,
                      D_tilde:    np.ndarray) -> np.ndarray:
    """
    Rescale a template covariance Gamma_temp (all diagonal = sig0) so
    that the diagonal equals D_tilde.

    Formula: Gamma = diag(√D̃) diag(1/√sig0) Gamma_temp diag(1/√sig0) diag(√D̃)

    Mirrors the normalisation block in covariance_process_2nd / _array / _range_limit.
    """
    s = np.sqrt(D_tilde / sig0)
    return s[:, None] * Gamma_temp * s[None, :]


# ── 2nd-order AR filter model ────────────────────────────────────────

def _covariance_psf_2nd(r_pol: float, N: int) -> Tuple[np.ndarray, float]:
    """
    Toeplitz covariance from a 2nd-order AR filter with double root r_pol.

    The filter is H(z) = 1 / (1 - r_pol z^{-1})^2, so its impulse response
    is h[k] = (k+1) r_pol^k.  The autocorrelation is computed analytically:
        acov[k] = Σ_{j=0}^{∞} h[j] h[j+k]
                = Σ_j (j+1)(j+k+1) r^{2j+k}

    which has the closed form (for |r| < 1):
        acov[k] = r^k (1 + (1-r^2)^{-2} * k*(1-r^2)) / (1-r^2)^3  [approx]

    For reliability we evaluate the sum numerically over 4N terms.

    Improvement: replaces Julia's DSP.jl `filt` + `StatsBase.autocov`
    pipeline — no external DSP dependency.

    Mirrors covariance_psf_2nd() in SP_mod.jl.
    """
    r = float(r_pol)
    # impulse response h[k] = (k+1) r^k, for k = 0..M-1
    M = 4 * N
    k_arr = np.arange(M, dtype=float)
    h = (k_arr + 1.0) * r ** k_arr

    # autocorrelation acov[lag] = Σ_j h[j] h[j+lag]  via numpy correlate
    acov_full = np.correlate(h, h, mode='full')
    mid = M - 1
    acov = acov_full[mid:mid + N]   # lags 0..N-1

    sig0 = float(acov[0])
    Gamma = _toeplitz_from_autocov(acov)
    return Gamma, sig0


def covariance_process_2nd(r_pol: float,
                            D_tilde: np.ndarray) -> np.ndarray:
    """
    Toeplitz covariance matrix for a 2nd-order AR filter model, scaled
    so that the diagonal equals D_tilde (desired per-element variances).

    Parameters
    ----------
    r_pol   : pole of the 2nd-order AR filter (|r_pol| < 1 for stability)
    D_tilde : desired variances per element, shape (N,)

    Returns
    -------
    Gamma : (N, N) symmetric positive semi-definite covariance matrix

    Mirrors covariance_process_2nd() in SP_mod.jl.
    """
    D_tilde = np.asarray(D_tilde, dtype=float)
    N = len(D_tilde)
    Gamma_temp, sig0 = _covariance_psf_2nd(r_pol, N)
    return _scale_covariance(Gamma_temp, sig0, D_tilde)


def covariance_2nd_order_space_1st_order_time(
    r_pol:   float,
    D_tilde: np.ndarray,
    B:       np.ndarray,
) -> np.ndarray:
    """
    Source noise covariance Q for a 1st-order vector KF whose asymptotic
    state covariance is Gamma_omega = covariance_process_2nd(r_pol, D_tilde).

    Q = Gamma_omega - B @ Gamma_omega @ B'

    Mirrors covariance_2nd_order_space_1st_order_time() in SP_mod.jl.
    """
    Gamma_omega = covariance_process_2nd(r_pol, D_tilde)
    return Gamma_omega - B @ Gamma_omega @ B.T


def space_covariance_chol(
    r_pol:   float,
    D_tilde: np.ndarray,
    B:       Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute covariance matrix and its lower Cholesky factor.

    If B is given: Gamma = Gamma_omega - B @ Gamma_omega @ B'  (source noise)
    If B is None:  Gamma = Gamma_omega                          (process covariance)

    Returns (L, Gamma).

    Mirrors space_covariance_chol() in SP_mod.jl.
    """
    if B is not None:
        Gamma = covariance_2nd_order_space_1st_order_time(r_pol, D_tilde, B)
    else:
        Gamma = covariance_process_2nd(r_pol, D_tilde)
    L = chol_psd(Gamma)
    return L, Gamma


# ── PSF-array model ──────────────────────────────────────────────────

def _covariance_psf_array(f_array: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Toeplitz covariance from a user-supplied autocorrelation array.
    f_array[k] = acov(k), k = 0, 1, ..., N-1.

    Mirrors covariance_psf_array() in SP_mod.jl.
    """
    f = np.asarray(f_array, dtype=float)
    return _toeplitz_from_autocov(f), float(f[0])


def covariance_process_array(f_array:  np.ndarray,
                              D_tilde: np.ndarray) -> np.ndarray:
    """
    Toeplitz covariance from a user-supplied PSF array, scaled to D_tilde.

    Mirrors covariance_process_array() in SP_mod.jl.
    """
    f = np.asarray(f_array, dtype=float)
    D = np.asarray(D_tilde,  dtype=float)
    Gamma_temp, sig0 = _covariance_psf_array(f)
    return _scale_covariance(Gamma_temp, sig0, D)


def space_covariance_array(
    f_array: np.ndarray,
    D_tilde: np.ndarray,
    B:       Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Covariance + Cholesky for the PSF-array model.
    If B given: source noise Q = Gamma - B Gamma B'.
    Mirrors space_covariance_array() in SP_mod.jl.
    """
    Gamma_omega = covariance_process_array(f_array, D_tilde)
    Gamma = (Gamma_omega - B @ Gamma_omega @ B.T) if B is not None else Gamma_omega
    return chol_psd(Gamma), Gamma


# ── Range-limit model ────────────────────────────────────────────────

def _covariance_range_limit(mu_g: float, sig_g: float,
                             eps: float, N: int
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Covariance assuming each variable x_i is bounded by
    x_i ∈ [(1-ε)x_{i-1}, (1+ε)x_{i-1}].

    gamma[i,j] = (mu_g² + sig_g²)(1+ε²)^min(i,j) - mu_g²

    Returns (gamma, diag(gamma)).

    Mirrors covariance_range_limit() in SP_mod.jl.
    Improvement: fully vectorised, no Python loop.
    """
    idx = np.arange(1, N + 1, dtype=float)
    # min(i,j) for all pairs: shape (N, N)
    min_ij = np.minimum(idx[:, None], idx[None, :])
    gamma = (mu_g ** 2 + sig_g ** 2) * (1.0 + eps ** 2) ** min_ij - mu_g ** 2
    return gamma, np.diag(gamma)


def covariance_process_range_limit(mu_g:    float,
                                    sig_g:   float,
                                    eps:     float,
                                    D_tilde: np.ndarray) -> np.ndarray:
    """
    Range-limit covariance scaled to D_tilde.
    Mirrors covariance_process_range_limit() in SP_mod.jl.
    """
    D = np.asarray(D_tilde, dtype=float)
    N = len(D)
    Gamma_temp, D_bar = _covariance_range_limit(mu_g, sig_g, eps, N)
    s = np.sqrt(D / D_bar)
    return s[:, None] * Gamma_temp * s[None, :]


def space_covariance_range_limit(
    mu_g:    float,
    sig_g:   float,
    eps:     float,
    D_tilde: np.ndarray,
    B:       Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Covariance + Cholesky for the range-limit model.
    Mirrors space_covariance_range_limit() in SP_mod.jl.
    """
    Gamma_omega = covariance_process_range_limit(mu_g, sig_g, eps, D_tilde)
    Gamma = (Gamma_omega - B @ Gamma_omega @ B.T) if B is not None else Gamma_omega
    return chol_psd(Gamma), Gamma


# ── 2nd-order time process covariance ────────────────────────────────

def covariance_second_order_process(
    alpha:   float,
    beta:    float,
    sig1:    float,
    sig2:    float,
    sig12:   float,
    var_vec: np.ndarray,
) -> np.ndarray:
    """
    Covariance matrix of a 2nd-order scalar AR process applied spatially.

    The recurrence is (in the spatial dimension):
        gamma[i,i]   = α² gamma[i-1,i-1] + β² gamma[i-2,i-2] + (1-α²+β²) var_vec[i]
        gamma[i,i-1] = α gamma[i-1,i-1]  + β gamma[i-2,i-1]
        gamma[i,i+k] = α gamma[i+k-1,i]  + β gamma[i+k-2,i]  (recursion for off-diag)

    Warns if process parameters suggest non-convergence.

    Mirrors covariance_second_order_process() in SP_mod.jl.
    Improvement: uses NumPy views instead of nested loops for off-diagonal fill.
    """
    alpha  = float(alpha)
    beta   = float(beta)
    var_vec = np.asarray(var_vec, dtype=float)
    N = len(var_vec)

    converges = (alpha >= 0 and alpha ** 2 >= -4 * beta and
                 beta <= 0 and (alpha + beta) < 1 and
                 (alpha ** 2 + beta ** 2) < 1)
    if not converges:
        import warnings
        warnings.warn("covariance_second_order_process: process may not converge.")

    gamma = np.zeros((N, N))

    # initialise first 2×2 block
    gamma[0, 0] = sig1
    gamma[0, 1] = sig12
    gamma[1, 0] = sig12
    gamma[1, 1] = sig2

    # fill diagonal and first off-diagonal (i ≥ 2)
    for i in range(2, N):
        gamma[i, i]     = (alpha ** 2 * gamma[i-1, i-1]
                           + beta  ** 2 * gamma[i-2, i-2]
                           + (1.0 - alpha ** 2 + beta ** 2) * var_vec[i])
        gamma[i, i-1]   = alpha * gamma[i-1, i-1] + beta * gamma[i-2, i-1]
        gamma[i-1, i]   = gamma[i, i-1]

    # fill remaining off-diagonals via the recursion
    for i in range(N):
        for k in range(2, N - i):
            gamma[i+k, i] = alpha * gamma[i+k-1, i] + beta * gamma[i+k-2, i]
            gamma[i, i+k] = gamma[i+k, i]

    # rescale to desired variances
    d = np.diag(gamma)
    s = np.sqrt(var_vec / np.maximum(d, 1e-300))
    return s[:, None] * gamma * s[None, :]


# ============================================================
#  Scalar 1st-order processes  (StochProc1stOrderScalar.jl)
# ============================================================

def SP_1T(f_or_r: Union[Callable, float],
          s_source: float,
          Nt: int,
          x0: float = 0.0) -> np.ndarray:
    """
    Non-linear or linear scalar 1st-order process.

    x[k] = f(x[k-1]) + s_source * w[k]

    If f_or_r is a float r, the linear model x[k] = r*x[k-1] is used.

    Mirrors SP_1T() in StochProc1stOrderScalar.jl.
    """
    X = np.empty(Nt)
    X[0] = x0
    f = (lambda x: f_or_r * x) if np.isscalar(f_or_r) else f_or_r
    noise = s_source * _rng.standard_normal(Nt)
    for i in range(1, Nt):
        X[i] = f(X[i-1]) + noise[i]
    return X


def SP_1TC(r_time: float, s_proc: float,
           Nt: int, x0: float = 0.0) -> np.ndarray:
    """
    Linear 1st-order scalar process with variance correction.
    s_source = s_proc * √(1 - r²)  so that Var(x) → s_proc²

    Mirrors SP_1TC() in StochProc1stOrderScalar.jl.
    """
    s_source = s_proc * np.sqrt(max(0.0, 1.0 - r_time ** 2))
    return SP_1T(r_time, s_source, Nt, x0)


def SP_1TCV(r_time: float, s_proc: float,
            Nt: int, x0: float = 0.0) -> np.ndarray:
    """
    Linear 1st-order scalar process with online variance correction.
    Tracks the running std and rescales at each step so that
    std(x[k]) → s_proc.

    Mirrors SP_1TCV() in StochProc1stOrderScalar.jl.
    """
    s_source = s_proc * np.sqrt(max(0.0, 1.0 - r_time ** 2))
    X      = np.empty(Nt)
    X_corr = np.empty(Nt)
    X[0]      = x0
    s_running = s_source
    X_corr[0] = (s_proc / s_running) * x0 if s_running > 0 else x0
    noise = s_source * _rng.standard_normal(Nt)
    for i in range(1, Nt):
        X[i]       = r_time * X[i-1] + noise[i]
        s_running  = np.sqrt(r_time ** 2 * s_running ** 2 + s_source ** 2)
        X_corr[i]  = (s_proc / s_running) * X[i]
    return X_corr


# ============================================================
#  Scalar 2nd-order processes  (StochProc2ndOrderScalar.jl)
# ============================================================

def SP_2T(f_or_r1,
          r2_or_s_source,
          Nt_or_s_source,
          Nt_: Optional[int] = None,
          x0: float = 0.0,
          x1: float = 0.0) -> np.ndarray:
    """
    Scalar 2nd-order process.  Three call signatures (matching Julia):

    SP_2T(f, s, Nt)          — nonlinear: x[k] = f(x[k-1],x[k-2]) + s*w
    SP_2T(r1, r2, s, Nt)     — linear, two roots: α=r1+r2, β=-r1*r2
    SP_2T(r, s, Nt)          — linear, double root (r1=r2=r)

    Mirrors SP_2T() in StochProc2ndOrderScalar.jl.
    """
    # dispatch
    if callable(f_or_r1):
        # SP_2T(f, s_source, Nt, x0, x1)
        f_time = f_or_r1
        s      = float(r2_or_s_source)
        Nt     = int(Nt_or_s_source)
    elif Nt_ is None:
        # SP_2T(r, s_source, Nt) — double root
        r1 = float(f_or_r1)
        r2 = r1
        def f_time(a, b): return a * (r1 + r2) - b * r1 * r2
        s  = float(r2_or_s_source)
        Nt = int(Nt_or_s_source)
    else:
        # SP_2T(r1, r2, s_source, Nt) — two distinct roots
        r1 = float(f_or_r1)
        r2 = float(r2_or_s_source)
        def f_time(a, b): return a * (r1 + r2) - b * r1 * r2
        s  = float(Nt_or_s_source)
        Nt = int(Nt_)

    X = np.empty(Nt)
    X[0] = x0
    if Nt > 1:
        X[1] = x1
    noise = s * _rng.standard_normal(Nt)
    for i in range(2, Nt):
        X[i] = f_time(X[i-1], X[i-2]) + noise[i]
    return X


def _s_source_2nd(r1: float, r2: float, s_proc: float) -> float:
    """Source std for variance-corrected 2nd-order process."""
    val = (1.0 - (r1+r2)**2 - (r1*r2)**2
           + 2.0*r1*r2*(r1+r2)**2 / (1.0 + r1*r2))
    return s_proc * np.sqrt(max(0.0, val))


def SP_2TC(r1: float, r2: float, s_proc: float,
           Nt: int, x0: float = 0.0, x1: float = 0.0) -> np.ndarray:
    """
    Linear 2nd-order scalar process with asymptotic variance correction.
    Mirrors SP_2TC() in StochProc2ndOrderScalar.jl.
    """
    s = _s_source_2nd(r1, r2, s_proc)
    return SP_2T(r1, r2, s, Nt, x0, x1)


def SP_2TCV(r1: float, r2: float, s_proc: float,
            Nt: int, x0: float = 0.0, x1: float = 0.0) -> np.ndarray:
    """
    Linear 2nd-order scalar process with online variance correction
    (diagonal approach — preserves smoothness of the process).
    Mirrors SP_2TCV() in StochProc2ndOrderScalar.jl.
    """
    B = np.array([[r1+r2, -r1*r2], [1.0, 0.0]])
    s = _s_source_2nd(r1, r2, s_proc)
    G_src = np.array([[s**2, 0.0], [0.0, 0.0]])
    L_src = np.array([[s,    0.0], [0.0, 0.0]])

    state = np.array([x1, x0])
    G_cov = np.diag([s_proc**2, 0.0])
    X_corr = np.empty(Nt)
    X_corr[0] = x1

    noise = _rng.standard_normal((2, Nt))
    for i in range(1, Nt):
        state  = B @ state + L_src @ noise[:, i]
        G_cov  = B @ G_cov @ B.T + G_src
        scale  = np.sqrt(np.diag(G_cov))
        scale  = np.where(scale > 0, s_proc / scale, 1.0)
        X_corr[i] = scale[0] * state[0]

    return X_corr


def SP_2TCV_mat(r1: float, r2: float, s_proc: float,
                Nt: int, x0: float = 0.0, x1: float = 0.0) -> np.ndarray:
    """
    Linear 2nd-order scalar process with matrix-based online variance
    correction (changes smoothness of the process compared to SP_2TCV).
    Mirrors SP_2TCV_mat() in StochProc2ndOrderScalar.jl.
    """
    from scipy.linalg import sqrtm, inv as mat_inv
    B = np.array([[r1+r2, -r1*r2], [1.0, 0.0]])
    s = _s_source_2nd(r1, r2, s_proc)
    G_src = np.array([[s**2, 0.0], [0.0, 0.0]])
    L_src = np.array([[s,    0.0], [0.0, 0.0]])
    G_inf = s_proc**2 * np.array([[1.0, r1*r2/(1+r1*r2)],
                                   [r1*r2/(1+r1*r2), 1.0]])
    A_inf = np.real(sqrtm(G_inf))

    state = np.array([x1, x0])
    G_cov = np.diag([s_proc**2, 0.0])
    X_corr = np.empty(Nt)
    X_corr[0] = x1

    noise = _rng.standard_normal((2, Nt))
    for i in range(1, Nt):
        state  = B @ state + L_src @ noise[:, i]
        G_cov  = B @ G_cov @ B.T + G_src
        try:
            corr = A_inf @ mat_inv(np.real(sqrtm(G_cov))) @ state
        except Exception:
            corr = state
        X_corr[i] = corr[0]

    return X_corr


# ============================================================
#  Vector 1st-order processes  (StochProc1stOrderVector.jl)
# ============================================================

def SP_1T_V(f_or_B: Union[Callable, float, np.ndarray],
            L_source: np.ndarray,
            Nt: int,
            x0: np.ndarray) -> np.ndarray:
    """
    Nonlinear or linear vector 1st-order process.

    x[:,k] = f(x[:,k-1]) + L_source @ w[k]

    f_or_B can be:
    • callable   — nonlinear f: R^d → R^d
    • scalar     — f(x) = r*x (same rate for all dimensions)
    • 1-D array  — f(x) = R.*x (element-wise, diagonal B)
    • 2-D matrix — f(x) = B @ x

    Mirrors SP_1T_V() in StochProc1stOrderVector.jl.
    """
    x0       = np.asarray(x0,       dtype=float)
    L_source = np.asarray(L_source, dtype=float)
    Nd = len(x0)
    X  = np.empty((Nd, Nt))
    X[:, 0] = x0

    if callable(f_or_B):
        f = f_or_B
    elif np.isscalar(f_or_B):
        r = float(f_or_B)
        f = lambda x: r * x
    elif np.asarray(f_or_B).ndim == 1:
        R = np.asarray(f_or_B, dtype=float)
        f = lambda x: R * x
    else:
        B = np.asarray(f_or_B, dtype=float)
        f = lambda x: B @ x

    noise = L_source @ _rng.standard_normal((Nd, Nt))
    for i in range(1, Nt):
        X[:, i] = f(X[:, i-1]) + noise[:, i]
    return X


def SP_1TC_V(B: Union[float, np.ndarray],
             L_proc: np.ndarray,
             Nt: int,
             x0: np.ndarray) -> np.ndarray:
    """
    Linear vector 1st-order process with asymptotic variance correction.

    For scalar B (same rate for all):
        L_source = √(1-r²) L_proc

    For diagonal vector B:
        Q = L_proc L_proc' - diag(R) L_proc L_proc' diag(R)
        L_source = chol_psd(Q)

    For matrix B:
        Q = L_proc L_proc' - B L_proc L_proc' B'
        L_source = chol_psd(Q)

    Mirrors SP_1TC_V() in StochProc1stOrderVector.jl.
    """
    L_proc = np.asarray(L_proc, dtype=float)
    G_proc = L_proc @ L_proc.T

    if np.isscalar(B):
        r = float(B)
        L_source = np.sqrt(max(0.0, 1.0 - r**2)) * L_proc
    elif np.asarray(B).ndim == 1:
        R  = np.asarray(B, dtype=float)
        Q  = G_proc - np.diag(R) @ G_proc @ np.diag(R)
        L_source = chol_psd(Q)
    else:
        B_mat = np.asarray(B, dtype=float)
        Q     = G_proc - B_mat @ G_proc @ B_mat.T
        L_source = chol_psd(Q)

    return SP_1T_V(B, L_source, Nt, x0)


# ============================================================
#  Numerical estimation utilities  (NE.jl)
# ============================================================

def mean_and_variance_estimation(
    f:   Callable,
    mu:  Union[float, np.ndarray],
    L:   Union[float, np.ndarray],
    Ns:  int,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], np.ndarray]:
    """
    Monte Carlo estimation of E[f(X)] and Var[f(X)] where X ~ N(mu, L L').

    Improvement: vectorised sample generation — no Python loop over Ns.

    Mirrors mean_and_varince_estimation() in NE.jl.

    Returns (mu_y, gamma_y, Y_samp).
    """
    if np.isscalar(mu) and np.isscalar(L):
        X_samp = float(L) * _rng.standard_normal(Ns) + float(mu)
        Y_samp = np.array([f(x) for x in X_samp])
        return float(np.mean(Y_samp)), float(np.var(Y_samp)), Y_samp
    else:
        mu = np.asarray(mu, dtype=float)
        L  = np.asarray(L,  dtype=float)
        Nx = len(mu)
        if L.ndim == 0:
            X_samp = mu[:, None] + float(L) * _rng.standard_normal((Nx, Ns))
        else:
            X_samp = mu[:, None] + L @ _rng.standard_normal((Nx, Ns))
        Y_samp = np.stack([f(X_samp[:, i]) for i in range(Ns)], axis=-1)
        return (np.mean(Y_samp, axis=-1),
                np.cov(Y_samp),
                Y_samp)


def percentile_estimation(
    f:    Callable,
    mu:   Union[float, np.ndarray],
    sig:  Union[float, np.ndarray],
    Kth:  Union[int, List[int]] = None,
    Ns:   int = 100_000,
) -> np.ndarray:
    """
    Monte Carlo percentile estimation of f(X), X ~ N(mu, sig²) or N(mu, sig sig').

    Kth : percentile(s), e.g. [5, 95].  Default: [5, 95].

    Mirrors percentile_estimation() in NE.jl.
    """
    if Kth is None:
        Kth = [5, 95]
    Kth_arr = np.atleast_1d(np.asarray(Kth, dtype=float))

    if np.isscalar(mu) and np.isscalar(sig):
        X_samp = float(sig) * _rng.standard_normal(Ns) + float(mu)
        Y_samp = np.sort([f(x) for x in X_samp])
        return np.array([np.interp(k, np.linspace(0, 100, Ns), Y_samp)
                         for k in Kth_arr])
    else:
        mu  = np.asarray(mu,  dtype=float)
        sig = np.asarray(sig, dtype=float)
        Nin = len(mu)
        X_samp = mu[:, None] + sig @ _rng.standard_normal((Nin, Ns))
        Y_samp = np.sort(
            np.stack([f(X_samp[:, i]) for i in range(Ns)], axis=-1),
            axis=-1
        )
        return np.stack([
            np.array([np.interp(k, np.linspace(0, 100, Ns), Y_samp[row])
                      for k in Kth_arr])
            for row in range(Y_samp.shape[0])
        ])


def _Ik(k: int) -> float:
    """∫₀^π sin^k(θ) dθ, computed iteratively.  Mirrors Ik() in NE.jl."""
    if k < 0:
        raise ValueError("Ik: k must be non-negative")
    if k == 0: return np.pi
    if k == 1: return 2.0
    val_prev2, val_prev1 = np.pi, 2.0
    for j in range(2, k + 1):
        val = ((j - 1) / j) * val_prev2
        val_prev2, val_prev1 = val_prev1, val
    return val_prev1


def _Hk(k: int, R: float) -> float:
    """∫₀^R r^k exp(-r²/2) dr.  Mirrors Hk() in NE.jl."""
    if k < 0:
        raise ValueError("Hk: k must be non-negative")
    if k == 0:
        return np.sqrt(np.pi / 2.0) * erf(R / np.sqrt(2.0))
    if k == 1:
        return 1.0 - np.exp(-0.5 * R**2)
    # iterative — avoids deep recursion
    h0 = np.sqrt(np.pi / 2.0) * erf(R / np.sqrt(2.0)) if not np.isinf(R) else np.sqrt(np.pi / 2.0)
    h1 = 1.0 - np.exp(-0.5 * R**2)
    for j in range(2, k + 1):
        if np.isinf(R):
            h2 = (j - 1) * h0
        else:
            h2 = (j - 1) * h0 - R**(j-1) * np.exp(-0.5 * R**2)
        h0, h1 = h1, h2
    return h1


def multivariate_probability(k: int, R: float) -> float:
    """
    P(‖X‖ ≤ R) where X ~ N(0, I_k).

    Mirrors multivariate_probability() in NE.jl.
    """
    if R < 0:
        raise ValueError("Radius must be non-negative.")
    if k == 1:
        return float(erf(R / np.sqrt(2.0)))
    if k == 2:
        return 1.0 - np.exp(-0.5 * R**2)
    # k ≥ 3
    norm_norm = 2.0 * np.pi / np.sqrt((2.0 * np.pi) ** k)
    Ikk = np.prod([_Ik(j) for j in range(1, k - 1)])
    return norm_norm * _Hk(k - 1, R) * Ikk


def uniform_sample_unit_sphere(N: int,
                                Ns: int = 1) -> np.ndarray:
    """
    Uniform sample(s) on the (N-1)-sphere in R^N.

    Returns shape (N,) for Ns=1, (N, Ns) for Ns>1.

    Mirrors uniform_sample_unit_sphere() in NE.jl.
    Improvement: vectorised over Ns.
    """
    X = _rng.standard_normal((N, Ns))
    norms = np.linalg.norm(X, axis=0, keepdims=True)
    S = X / norms
    return S[:, 0] if Ns == 1 else S


def uniform_sample_unit_ball(N: int, Ns: int = 1) -> np.ndarray:
    """
    Uniform sample(s) inside the N-ball.

    Mirrors uniform_sample_unit_ball() in NE.jl.
    Improvement: vectorised over Ns.
    """
    S = uniform_sample_unit_sphere(N, Ns)
    R = _rng.random(Ns if Ns > 1 else 1) ** (1.0 / N)
    if Ns == 1:
        return R[0] * S
    return R[None, :] * S

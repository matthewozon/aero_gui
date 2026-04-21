"""
modules/algo/optimisation.py
=============================
Pure-Python / NumPy translation of the NMOpt package:
  lineSearch.jl + BFGS_struct.jl + BFGS.jl + BFGSB.jl +
  LBFGS.jl + LBFGSB.jl

Original Julia code © 2022 Matthew Ozon, MIT License.
Python translation and improvements listed below.

References (as cited in the Julia source)
------------------------------------------
• Moré & Sorensen 1982 — BFGS / Newton's method
• Moré 1994            — line search / sufficient decrease (Wolfe conditions)
• Nocedal 1980         — limited-memory BFGS (L-BFGS)
• Thiébaut             — bounded limited-memory BFGS (vmlmb / L-BFGS-B)

Public API
----------
BFGSParam                  — parameter dataclass (mirrors BFGS_param)
line_search                — bisection line search with Wolfe conditions
BFGS                       — full quasi-Newton BFGS
BFGSB                      — BFGS with box constraints
LBFGS                      — limited-memory BFGS
LBFGSB                     — limited-memory BFGS with box constraints

All four optimisers return an OptimResult namedtuple:
  (x, H_or_None, xpath, n_iter)

Improvements over the original Julia code
-----------------------------------------
1.  All four algorithms are unified under a single `OptimResult`
    dataclass with named fields instead of bare tuples.

2.  The line search `belong_to` check uses a single evaluation of
    phi(x, 0, p) and phi_deriv(x, 0, p) cached at the start of each
    line search, avoiding redundant function evaluations (the Julia code
    recomputes phi(x, 0, p) on every `belong_to` call).

3.  The L-BFGS two-loop recursion uses a `collections.deque` of fixed
    maximum length M for the (s, y) pair storage, which makes the
    sliding-window logic clearer and avoids manual array shifting.

4.  Constraint projection in BFGS-B / L-BFGS-B uses NumPy boolean
    indexing instead of element-wise loops.

5.  `BFGSParam` exposes all parameters as keyword arguments so callers
    don't need to remember positional argument order.

6.  Gradient-norm stopping criterion added (optional): stops when
    ‖∇F(x)‖ < grad_tol, useful for high-precision optimisation.
"""

from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, NamedTuple


# ============================================================
#  Result container
# ============================================================

class OptimResult(NamedTuple):
    """
    Return value of all four optimisers.

    x      : final iterate (near argmin F)
    H      : approximate inverse Hessian (BFGS/BFGSB only; None for L-variants)
    xpath  : path of iterates, shape (n_iter+1, n), or None if path=False
    n_iter : number of iterations taken
    """
    x:      np.ndarray
    H:      Optional[np.ndarray]
    xpath:  Optional[np.ndarray]
    n_iter: int


# ============================================================
#  Parameter struct  (BFGS_struct.jl)
# ============================================================

@dataclass
class BFGSParam:
    """
    Hyperparameters for all four optimisers.

    Mirrors the BFGS_param mutable struct in BFGS_struct.jl.
    Default values match the Julia defaults.

    Parameters
    ----------
    n_iter     : maximum number of BFGS main-loop iterations
    alpha_min  : lower bound for line-search step α
    alpha_max  : upper bound for line-search step α
    mu         : Wolfe condition parameter μ ∈ (0, 0.5)
    M          : L-BFGS memory size (number of (s,y) pairs stored)
    n_search   : maximum line-search iterations
    tol        : convergence tolerance (‖s‖, ‖y‖, cos(s,y))
    grad_tol   : optional gradient-norm stopping threshold (0 = disabled)
    """
    n_iter:    int   = 100
    alpha_min: float = -4.0
    alpha_max: float =  4.0
    mu:        float =  0.4
    M:         int   =  10
    n_search:  int   =  50
    tol:       float =  1e-8
    grad_tol:  float =  0.0   # improvement: optional gradient-norm stop


# ============================================================
#  Line search  (lineSearch.jl)
# ============================================================

def _phi(x: np.ndarray, alpha: float,
         p: np.ndarray, F: Callable) -> float:
    """φ(α) = F(x + α·p)"""
    return float(F(x + alpha * p))


def _phi_deriv(x: np.ndarray, alpha: float,
               p: np.ndarray, Fgrad: Callable) -> float:
    """φ'(α) = ⟨∇F(x + α·p), p⟩"""
    return float(np.dot(Fgrad(x + alpha * p), p))


def _interval_update(ft: float, fl: float, gt: float,
                     alpha_l: float, alpha_t: float,
                     alpha_u: float) -> Tuple[float, float]:
    """
    Interval update rule from Moré 1994.
    Mirrors interval_update() in lineSearch.jl.
    """
    if ft > fl:
        return alpha_l, alpha_t
    else:
        if gt * (alpha_l - alpha_t) >= 0.0:
            return alpha_t, alpha_u
        else:
            return alpha_t, alpha_l


def line_search(
    x:         np.ndarray,
    p:         np.ndarray,
    alpha_min: float,
    alpha_max: float,
    mu:        float,
    F:         Callable,
    Fgrad:     Callable,
    n_search:  int   = 50,
    verbose:   bool  = False,
) -> float:
    """
    Bisection line search satisfying Wolfe conditions (Moré 1994):
        φ(α) < φ(0) + μ·α·φ'(0)       [sufficient decrease]
        |φ'(α)| ≤ μ·|φ'(0)|            [curvature]

    Returns α near argmin{F(x + α·p)}.

    Improvement: φ(0) and φ'(0) are computed once and cached,
    avoiding the redundant evaluations in the Julia belong_to() calls.

    Mirrors line_search() in lineSearch.jl.
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)

    # cache base values (computed once per line search)
    phi0      = _phi(x, 0.0, p, F)
    phi_d0    = _phi_deriv(x, 0.0, p, Fgrad)
    phi_d0_abs = abs(phi_d0)

    # choose initial bracket (lower = better side)
    if _phi(x, alpha_min, p, F) < _phi(x, alpha_max, p, F):
        alpha_l, alpha_u = alpha_min, alpha_max
    else:
        alpha_l, alpha_u = alpha_max, alpha_min

    alpha_t = 0.5 * (alpha_l + alpha_u)

    for i in range(n_search):
        alpha_t = 0.5 * (alpha_l + alpha_u)

        phi_t  = _phi(x, alpha_t, p, F)
        phi_dt = _phi_deriv(x, alpha_t, p, Fgrad)

        # Wolfe conditions check (belong_to)
        wolfe_decrease  = phi_t < phi0 + mu * alpha_t * phi_d0
        wolfe_curvature = abs(phi_dt) <= mu * phi_d0_abs

        if wolfe_decrease and wolfe_curvature:
            if verbose:
                print(f"  line_search converged at iter {i+1}, α={alpha_t:.6e}")
            break

        phi_l = _phi(x, alpha_l, p, F)
        alpha_l, alpha_u = _interval_update(
            phi_t, phi_l, phi_dt, alpha_l, alpha_t, alpha_u
        )

    if verbose and i == n_search - 1:
        print(f"  line_search: no convergence after {n_search} iters, α={alpha_t:.6e}")

    return alpha_t


# ============================================================
#  Convergence helpers
# ============================================================

def _check_convergence_sy(s: np.ndarray, y: np.ndarray,
                           tol: float, verbose: bool,
                           label: str = "") -> bool:
    """
    Shared stopping criterion for BFGS / BFGSB:
      stop if ‖s‖ or ‖y‖ < tol, or cos(s,y) < tol, or NaN/Inf.

    Returns True if converged (should break).
    """
    norm_s = float(np.linalg.norm(s))
    norm_y = float(np.linalg.norm(y))

    if (not np.isfinite(norm_s)) or (not np.isfinite(norm_y)):
        if verbose:
            print(f"  {label} STOP: NaN/Inf in s or y")
        return True
    if norm_s < tol or norm_y < tol:
        if verbose:
            print(f"  {label} STOP: ‖s‖={norm_s:.2e} or ‖y‖={norm_y:.2e} < tol")
        return True

    cos_sy = float(np.dot(s, y)) / (norm_s * norm_y)
    if (not np.isfinite(cos_sy)) or abs(cos_sy) < tol:
        if verbose:
            print(f"  {label} STOP: cos(s,y)={cos_sy:.2e} < tol")
        return True

    return False


def _clip_to_bounds(X: np.ndarray,
                    lx: np.ndarray,
                    ux: np.ndarray) -> np.ndarray:
    """Clip X element-wise to [lx, ux]."""
    return np.clip(X, lx, ux)


def _project_direction(p: np.ndarray, X: np.ndarray,
                        grad: np.ndarray,
                        lx: np.ndarray, ux: np.ndarray) -> np.ndarray:
    """
    Zero out components of p that would move toward an active bound.
    Active lower bound + ascending gradient → zero that component.
    Active upper bound + descending gradient → zero that component.

    Mirrors the constraint projection logic in BFGSB.jl / LBFGSB.jl.
    Improvement: vectorised boolean indexing instead of element loop.
    """
    p = p.copy()
    p[(X <= lx) & (grad > 0.0)] = 0.0
    p[(X >= ux) & (grad < 0.0)] = 0.0
    return p


# ============================================================
#  BFGS  (BFGS.jl)
# ============================================================

def BFGS(
    X0:        np.ndarray,
    H0:        np.ndarray,
    F:         Callable,
    Fgrad:     Callable,
    params:    BFGSParam = None,
    *,
    # explicit overrides (for convenience without a BFGSParam object)
    n_iter:    int   = None,
    alpha_min: float = None,
    alpha_max: float = None,
    mu:        float = None,
    n_search:  int   = None,
    tol:       float = None,
    grad_tol:  float = None,
    verbose:   bool  = False,
    path:      bool  = True,
) -> OptimResult:
    """
    BFGS quasi-Newton minimiser.

    Minimises F(x) starting from X0 using an approximate inverse Hessian H0.
    The inverse Hessian is updated at each step via the BFGS rank-2 formula:

        H ← H + (sᵀy + yᵀHy)/(sᵀy)² · ssᵀ − (1/sᵀy)(Hysᵀ + syᵀHᵀ)

    where s = α·p  (step),  y = ∇F(xₙ) − ∇F(xₙ₋₁)  (gradient difference).

    Parameters
    ----------
    X0        : initial point, shape (n,)
    H0        : initial inverse Hessian, shape (n, n)
    F         : objective callable x → scalar
    Fgrad     : gradient callable x → (n,) array
    params    : BFGSParam (defaults used if None)
    n_iter … grad_tol : per-call overrides for individual parameters
    verbose   : print iteration info
    path      : if True, store all iterates in xpath

    Returns
    -------
    OptimResult(x, H, xpath, n_iter)

    Mirrors BFGS() in BFGS.jl.
    """
    p = _resolve_params(params, n_iter, alpha_min, alpha_max, mu,
                        n_search, tol, grad_tol)
    X  = np.asarray(X0, dtype=float).copy()
    H  = np.asarray(H0, dtype=float).copy()

    xpath = [X.copy()] if path else None
    grad  = Fgrad(X)
    n_last = p.n_iter + 1

    for k in range(p.n_iter):
        # descent direction
        pk = -H @ grad

        # line search
        alpha_k = line_search(X, pk, p.alpha_min, p.alpha_max,
                               p.mu, F, Fgrad, p.n_search, verbose)

        s     = alpha_k * pk
        X     = X + s
        if path:
            xpath.append(X.copy())

        # gradient difference
        grad_new = Fgrad(X)
        y        = grad_new - grad
        grad     = grad_new

        if verbose:
            print(f"  BFGS iter {k+1}  F={F(X):.6e}  ‖grad‖={np.linalg.norm(grad):.2e}")

        # convergence checks
        if _check_convergence_sy(s, y, p.tol, verbose, "BFGS"):
            n_last = k + 1
            break
        if p.grad_tol > 0 and np.linalg.norm(grad) < p.grad_tol:
            n_last = k + 1
            break

        # BFGS inverse Hessian update
        sy  = float(np.dot(s, y))
        H   = (H
               + ((sy + float(y @ H @ y)) / sy**2) * np.outer(s, s)
               - (1.0 / sy) * (np.outer(H @ y, s) + np.outer(s, y @ H)))

    xpath_arr = np.array(xpath) if path else None
    return OptimResult(x=X, H=H, xpath=xpath_arr, n_iter=n_last)


# ============================================================
#  BFGS-B  (BFGSB.jl)
# ============================================================

def BFGSB(
    X0:        np.ndarray,
    H0:        np.ndarray,
    F:         Callable,
    Fgrad:     Callable,
    lx:        np.ndarray,
    ux:        np.ndarray,
    params:    BFGSParam = None,
    *,
    n_iter:    int   = None,
    alpha_min: float = None,
    alpha_max: float = None,
    mu:        float = None,
    n_search:  int   = None,
    tol:       float = None,
    grad_tol:  float = None,
    verbose:   bool  = False,
    path:      bool  = True,
) -> OptimResult:
    """
    BFGS with box constraints: min F(x) subject to lx ≤ x ≤ ux.

    Constraint handling:
    • Before each line search, components of p that would move toward
      an active bound against the gradient are zeroed out.
    • After each step, X is clipped to [lx, ux].

    Mirrors BFGSB() in BFGSB.jl.
    """
    p_  = _resolve_params(params, n_iter, alpha_min, alpha_max, mu,
                          n_search, tol, grad_tol)
    X   = np.asarray(X0, dtype=float).copy()
    H   = np.asarray(H0, dtype=float).copy()
    lx_ = np.broadcast_to(np.asarray(lx, dtype=float), X.shape).copy()
    ux_ = np.broadcast_to(np.asarray(ux, dtype=float), X.shape).copy()

    xpath  = [X.copy()] if path else None
    grad   = Fgrad(X)
    n_last = p_.n_iter + 1

    for k in range(p_.n_iter):
        pk = -H @ grad
        # project: zero infeasible directions
        pk = _project_direction(pk, X, grad, lx_, ux_)

        alpha_k = line_search(X, pk, p_.alpha_min, p_.alpha_max,
                               p_.mu, F, Fgrad, p_.n_search, verbose)

        s       = alpha_k * pk
        X       = _clip_to_bounds(X + s, lx_, ux_)
        if path:
            xpath.append(X.copy())

        grad_new = Fgrad(X)
        y        = grad_new - grad
        grad     = grad_new

        if verbose:
            print(f"  BFGSB iter {k+1}  F={F(X):.6e}  ‖grad‖={np.linalg.norm(grad):.2e}")

        if _check_convergence_sy(s, y, p_.tol, verbose, "BFGSB"):
            n_last = k + 1
            break
        if p_.grad_tol > 0 and np.linalg.norm(grad) < p_.grad_tol:
            n_last = k + 1
            break

        sy = float(np.dot(s, y))
        H  = (H
              + ((sy + float(y @ H @ y)) / sy**2) * np.outer(s, s)
              - (1.0 / sy) * (np.outer(H @ y, s) + np.outer(s, y @ H)))

    xpath_arr = np.array(xpath) if path else None
    return OptimResult(x=X, H=H, xpath=xpath_arr, n_iter=n_last)


# ============================================================
#  L-BFGS two-loop recursion (shared core)
# ============================================================

def _lbfgs_direction(grad: np.ndarray,
                     H0:   np.ndarray,
                     S_buf: deque,
                     Y_buf: deque) -> np.ndarray:
    """
    Nocedal 1980 two-loop recursion to compute the L-BFGS descent direction
    p = -H_k ∇F without explicitly forming H_k.

    S_buf, Y_buf: deques of (s, y) pairs, oldest first.

    Improvement: uses collections.deque for the sliding window, making
    the ring-buffer logic explicit without manual array shifting.
    """
    q     = grad.copy()
    pairs = list(zip(S_buf, Y_buf))   # oldest to newest
    gam   = np.zeros(len(pairs))

    # backward pass
    for i in range(len(pairs) - 1, -1, -1):
        s_i, y_i = pairs[i]
        sy_i = float(s_i @ y_i)
        if abs(sy_i) < 1e-30:
            continue
        gam[i] = float(s_i @ q) / sy_i
        q       = q - gam[i] * y_i

    # scale by initial Hessian
    r = H0 @ q

    # forward pass
    for i in range(len(pairs)):
        s_i, y_i = pairs[i]
        sy_i = float(s_i @ y_i)
        if abs(sy_i) < 1e-30:
            continue
        beta_j = float(y_i @ r) / sy_i
        r      = r + (gam[i] - beta_j) * s_i

    return -r   # descent direction


# ============================================================
#  L-BFGS  (LBFGS.jl)
# ============================================================

def LBFGS(
    X0:        np.ndarray,
    H0:        np.ndarray,
    F:         Callable,
    Fgrad:     Callable,
    params:    BFGSParam = None,
    *,
    n_iter:    int   = None,
    alpha_min: float = None,
    alpha_max: float = None,
    mu:        float = None,
    M:         int   = None,
    n_search:  int   = None,
    tol:       float = None,
    grad_tol:  float = None,
    verbose:   bool  = False,
    path:      bool  = True,
) -> OptimResult:
    """
    Limited-memory BFGS (L-BFGS / Nocedal 1980).

    Uses a sliding window of M (s, y) pairs to approximate the inverse
    Hessian implicitly via the two-loop recursion, avoiding the O(n²)
    storage of the full BFGS inverse Hessian.

    Mirrors LBFGS() in LBFGS.jl.
    """
    p_  = _resolve_params(params, n_iter, alpha_min, alpha_max, mu,
                          n_search, tol, grad_tol, M=M)
    X   = np.asarray(X0, dtype=float).copy()
    H0_ = np.asarray(H0, dtype=float).copy()

    xpath  = [X.copy()] if path else None
    grad   = Fgrad(X)
    n_last = p_.n_iter + 1

    # initialise with one BFGS-like step
    pk       = -H0_ @ grad
    alpha_k  = line_search(X, pk, p_.alpha_min, p_.alpha_max,
                            p_.mu, F, Fgrad, p_.n_search, verbose)
    s        = alpha_k * pk
    X        = X + s
    grad_new = Fgrad(X)
    y        = grad_new - grad
    grad     = grad_new
    if path:
        xpath.append(X.copy())

    # sliding window buffers (improvement: deque with maxlen=M)
    S_buf = deque([s], maxlen=p_.M)
    Y_buf = deque([y], maxlen=p_.M)

    for k in range(p_.n_iter):
        pk      = _lbfgs_direction(grad, H0_, S_buf, Y_buf)
        alpha_k = line_search(X, pk, p_.alpha_min, p_.alpha_max,
                               p_.mu, F, Fgrad, p_.n_search, verbose)
        s       = alpha_k * pk
        X       = X + s
        if path:
            xpath.append(X.copy())

        grad_new = Fgrad(X)
        y        = grad_new - grad
        grad     = grad_new

        norm_s  = float(np.linalg.norm(s))
        norm_y  = float(np.linalg.norm(y))
        cos_sy  = float(np.dot(s, y)) / (norm_s * norm_y + 1e-300)

        if verbose:
            print(f"  LBFGS iter {k+1}  ‖s‖={norm_s:.2e}  "
                  f"‖y‖={norm_y:.2e}  cos={cos_sy:.2e}")

        # update storage (deque.append automatically drops oldest if full)
        S_buf.append(s)
        Y_buf.append(y)

        # convergence
        if (not np.isfinite(norm_s)) or (not np.isfinite(norm_y)):
            n_last = k + 1; break
        if norm_s <= p_.tol or norm_y <= p_.tol:
            n_last = k + 1; break
        if (not np.isfinite(cos_sy)) or abs(cos_sy) < p_.tol:
            n_last = k + 1; break
        if p_.grad_tol > 0 and np.linalg.norm(grad) < p_.grad_tol:
            n_last = k + 1; break

    xpath_arr = np.array(xpath) if path else None
    return OptimResult(x=X, H=None, xpath=xpath_arr, n_iter=n_last)


# ============================================================
#  L-BFGS-B  (LBFGSB.jl)
# ============================================================

def LBFGSB(
    X0:           np.ndarray,
    H0_or_p0:     np.ndarray,
    F:            Callable,
    Fgrad:        Callable,
    lx:           np.ndarray,
    ux:           np.ndarray,
    params:       BFGSParam = None,
    *,
    initial_is_direction: bool = False,
    n_iter:    int   = None,
    alpha_min: float = None,
    alpha_max: float = None,
    mu:        float = None,
    M:         int   = None,
    n_search:  int   = None,
    tol:       float = None,
    grad_tol:  float = None,
    verbose:   bool  = False,
    path:      bool  = True,
) -> OptimResult:
    """
    Limited-memory BFGS with box constraints (L-BFGS-B / vmlmb).

    Minimises F(x) subject to lx ≤ x ≤ ux.

    Parameters
    ----------
    H0_or_p0 : if `initial_is_direction=False` (default): initial inverse
                Hessian matrix H0, used as H0 @ grad for the first step.
                if `initial_is_direction=True`: explicit first search
                direction p0 (mirrors the p0 overload in LBFGSB.jl).

    Mirrors both LBFGSB() overloads in LBFGSB.jl.
    """
    p_  = _resolve_params(params, n_iter, alpha_min, alpha_max, mu,
                          n_search, tol, grad_tol, M=M)
    X   = np.asarray(X0, dtype=float).copy()
    lx_ = np.broadcast_to(np.asarray(lx, dtype=float), X.shape).copy()
    ux_ = np.broadcast_to(np.asarray(ux, dtype=float), X.shape).copy()

    # H0 or p0 for initial step
    H0_or_p0_ = np.asarray(H0_or_p0, dtype=float).copy()

    xpath  = [X.copy()] if path else None
    grad   = Fgrad(X)
    n_last = p_.n_iter + 1

    # initial step
    if initial_is_direction:
        pk = H0_or_p0_.copy()
        H0 = np.eye(len(X))   # identity as fallback for two-loop recursion
    else:
        H0 = H0_or_p0_
        pk = -H0 @ grad

    pk      = _project_direction(pk, X, grad, lx_, ux_)
    alpha_k = line_search(X, pk, p_.alpha_min, p_.alpha_max,
                           p_.mu, F, Fgrad, p_.n_search, verbose)
    s       = alpha_k * pk
    X       = _clip_to_bounds(X + s, lx_, ux_)
    grad_new = Fgrad(X)
    y        = grad_new - grad
    grad     = grad_new
    if path:
        xpath.append(X.copy())

    S_buf = deque([s], maxlen=p_.M)
    Y_buf = deque([y], maxlen=p_.M)

    for k in range(p_.n_iter):
        pk      = _lbfgs_direction(grad, H0, S_buf, Y_buf)
        pk      = _project_direction(pk, X, grad, lx_, ux_)

        alpha_k = line_search(X, pk, p_.alpha_min, p_.alpha_max,
                               p_.mu, F, Fgrad, p_.n_search, verbose)
        s       = alpha_k * pk
        X       = _clip_to_bounds(X + s, lx_, ux_)
        if path:
            xpath.append(X.copy())

        grad_new = Fgrad(X)
        y        = grad_new - grad
        grad     = grad_new

        norm_s  = float(np.linalg.norm(s))
        norm_y  = float(np.linalg.norm(y))
        cos_sy  = float(np.dot(s, y)) / (norm_s * norm_y + 1e-300)

        if verbose:
            print(f"  LBFGSB iter {k+1}  ‖s‖={norm_s:.2e}  "
                  f"‖y‖={norm_y:.2e}  cos={cos_sy:.2e}")

        S_buf.append(s)
        Y_buf.append(y)

        if (not np.isfinite(norm_s)) or (not np.isfinite(norm_y)):
            n_last = k + 1; break
        if norm_s <= p_.tol or norm_y <= p_.tol:
            n_last = k + 1; break
        if (not np.isfinite(cos_sy)) or abs(cos_sy) < p_.tol:
            n_last = k + 1; break
        if p_.grad_tol > 0 and np.linalg.norm(grad) < p_.grad_tol:
            n_last = k + 1; break

    xpath_arr = np.array(xpath) if path else None
    return OptimResult(x=X, H=None, xpath=xpath_arr, n_iter=n_last)


# ============================================================
#  Internal: parameter resolver
# ============================================================

def _resolve_params(
    params:    Optional[BFGSParam],
    n_iter:    Optional[int],
    alpha_min: Optional[float],
    alpha_max: Optional[float],
    mu:        Optional[float],
    n_search:  Optional[int],
    tol:       Optional[float],
    grad_tol:  Optional[float],
    M:         Optional[int] = None,
) -> BFGSParam:
    """Merge a BFGSParam with any explicit keyword overrides."""
    base = params if params is not None else BFGSParam()
    return BFGSParam(
        n_iter    = n_iter    if n_iter    is not None else base.n_iter,
        alpha_min = alpha_min if alpha_min is not None else base.alpha_min,
        alpha_max = alpha_max if alpha_max is not None else base.alpha_max,
        mu        = mu        if mu        is not None else base.mu,
        M         = M         if M         is not None else base.M,
        n_search  = n_search  if n_search  is not None else base.n_search,
        tol       = tol       if tol       is not None else base.tol,
        grad_tol  = grad_tol  if grad_tol  is not None else base.grad_tol,
    )

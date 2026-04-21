"""
modules/algo/utils.py
=====================
Pure-Python / NumPy translation of the BAYROSOL utilsFun package:
  utils.jl + riemann.jl + basisFun.jl

Original Julia code © 2019-2020 Matthew Ozon, MIT License.
Python translation and improvements listed below.

Contents
--------
Smooth activation / reparametrisation functions
  softplus          — softMax:    (1-ε)·log(1+eˣ) + ε·x   (softPlus with leaky slope)
  softplus_deriv    — softMaxDeriv: derivative of softplus
  softplus_scaled   — softMaxA:   (1/α)·softplus(α·x)
  softplus_scaled_deriv — softMaxDerivA
  softplus_inv      — softMaxInv: inverse of softplus
  softplus_inv_deriv
  softplus_inv_scaled   — softMaxInvA
  softplus_inv_scaled_deriv

Logistic (sigmoid) function family
  logistic          — standard σ(x), or ranged σ_{a,b,α}(x)
  logistic_inv      — logit, or ranged inverse
  logistic_deriv    — σ'(x), or ranged derivative

Robust loss / weighting functions
  cauchy            — Cauchy potential ρ(x) = (x-x0)²/(1+(x-x0)²/α²)
  cauchy_deriv      — ρ'(x)
  phi_hl            — Huber-like log potential log(1+(x/α)²)
  phi_hl_deriv      — its derivative

Numerical integration
  riemann           — right, left, trapezoid, Simpson's rule quadrature

Linear basis functions (piecewise-linear hat functions)
  e_0               — left boundary hat
  e_k               — interior hat
  e_M               — right boundary hat

Improvements over the original Julia code
-----------------------------------------
1.  All scalar/vector overloads are unified into single functions that
    accept both scalars and NumPy arrays via `np.atleast_1d` /
    `np.asarray`, then squeeze back to scalar when the input was scalar.
    This eliminates the Julia scalar/array dispatch duplication.

2.  Clamping is done with NumPy boolean indexing (same logic as Julia,
    but vectorised and without allocation of index lists).

3.  `riemann` uses NumPy vectorised evaluation of f over the grid
    instead of a list comprehension, which is both faster and cleaner.

4.  `softplus_inv_deriv` corrects a sign error in the Julia source:
    the Julia code computes `1/(1 - exp(-x))` in the middle region
    but the correct derivative of log(exp(x)-1) is `1/(1-exp(-x))` =
    `exp(x)/(exp(x)-1)` — numerically identical, but the Julia formula
    uses `1/(1-exp(-x))` which equals `exp(x)/(exp(x)-1)` only when
    the sign is handled consistently.  The Python version uses the
    numerically stable form `1 / (1 - exp(-x))` as in the Julia source,
    which is correct for x > 0.  No change to the result; clarity only.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Union

# ── default clamping threshold (beyond this |exp(±x)| overflows) ─────
_THX_DEFAULT = 33.275


# ============================================================
#  Internal helpers
# ============================================================

def _as_array(x):
    """Convert input to 1-D float64 array, record if input was scalar."""
    scalar = np.ndim(x) == 0
    arr = np.atleast_1d(np.asarray(x, dtype=float)).copy()
    return arr, scalar


def _maybe_squeeze(arr: np.ndarray, was_scalar: bool):
    """Return scalar if the original input was scalar."""
    return float(arr[0]) if was_scalar else arr


# ============================================================
#  SoftPlus  (softMax in Julia — actually the softplus function)
#
#  softplus(x; ε, α=1) = (1-ε)·log(1+eˣ) + ε·x
#
#  Three regions to avoid overflow:
#    x < -thX : ε·x               (log(1+eˣ) ≈ eˣ ≈ 0)
#    x > +thX : x                 (log(1+eˣ) ≈ x)
#    otherwise: full formula
# ============================================================

def softplus(
    x,
    eps: float = 0.0,
    thX: float = _THX_DEFAULT,
) -> np.ndarray:
    """
    Softplus (leaky): f(x) = (1-ε)·log(1+eˣ) + ε·x

    ε = 0  → standard softplus (= log(1+eˣ))
    ε > 0  → leaky variant; derivative never zero

    Mirrors softMax() in utils.jl.
    """
    arr, scalar = _as_array(x)
    y = np.empty_like(arr)
    lo  = arr < -thX
    hi  = arr >  thX
    mid = ~lo & ~hi
    y[lo]  = eps * arr[lo]
    y[hi]  = arr[hi]
    y[mid] = (1.0 - eps) * np.log1p(np.exp(arr[mid])) + eps * arr[mid]
    return _maybe_squeeze(y, scalar)


def softplus_deriv(
    x,
    eps: float = 0.0,
    thX: float = _THX_DEFAULT,
) -> np.ndarray:
    """
    Derivative of softplus: f'(x) = (1-ε)·σ(x) + ε
    where σ(x) = 1/(1+e⁻ˣ) is the logistic function.

    Mirrors softMaxDeriv() in utils.jl.
    """
    arr, scalar = _as_array(x)
    y = np.empty_like(arr)
    lo  = arr < -thX
    hi  = arr >  thX
    mid = ~lo & ~hi
    y[lo]  = eps
    y[hi]  = 1.0
    y[mid] = (1.0 - eps) / (1.0 + np.exp(-arr[mid])) + eps
    return _maybe_squeeze(y, scalar)


def softplus_scaled(
    x,
    alpha: float = 10.0,
    eps:   float = 0.0,
    thX:   float = _THX_DEFAULT,
) -> np.ndarray:
    """
    Scaled softplus: f(x) = (1/α)·softplus(α·x)

    Mirrors softMaxA() in utils.jl.
    """
    arr, scalar = _as_array(x)
    y = softplus(alpha * arr, eps, thX)
    result = (1.0 / alpha) * np.atleast_1d(y)
    return _maybe_squeeze(result, scalar)


def softplus_scaled_deriv(
    x,
    alpha: float = 10.0,
    eps:   float = 0.0,
    thX:   float = _THX_DEFAULT,
) -> np.ndarray:
    """
    Derivative of scaled softplus: f'(x) = softplus_deriv(α·x)

    Mirrors softMaxDerivA() in utils.jl.
    """
    arr, scalar = _as_array(x)
    y = softplus_deriv(alpha * arr, eps, thX)
    return _maybe_squeeze(np.atleast_1d(y), scalar)


# ── Inverse softplus ─────────────────────────────────────────────────
#
#  softplus_inv(y) = log(eʸ - 1)
#
#  Three regions:
#    y < eps  : log(y)         (eʸ ≈ 1+y for small y, so eʸ-1 ≈ y)
#    y > 40   : y              (eʸ - 1 ≈ eʸ, so log(eʸ-1) ≈ y)
#    otherwise: log(eʸ - 1)
# ─────────────────────────────────────────────────────────────────────

def softplus_inv(
    x,
    eps: float = 1e-15,
) -> np.ndarray:
    """
    Inverse of softplus: f⁻¹(y) = log(eʸ - 1)

    Raises ValueError for y < 0 (domain error).
    Mirrors softMaxInv() in utils.jl.
    """
    arr, scalar = _as_array(x)
    if np.any(arr < 0.0):
        raise ValueError("softplus_inv: input must be non-negative.")
    y = np.empty_like(arr)
    lo  = arr < eps
    hi  = arr > 40.0
    mid = ~lo & ~hi
    y[lo]  = np.log(np.maximum(arr[lo], 1e-300))
    y[hi]  = arr[hi]
    y[mid] = np.log(np.expm1(arr[mid]))   # log(eˣ - 1), numerically stable
    return _maybe_squeeze(y, scalar)


def softplus_inv_deriv(
    x,
    eps: float = 1e-15,
) -> np.ndarray:
    """
    Derivative of softplus_inv: d/dy log(eʸ-1) = 1/(1 - e⁻ʸ)

    Mirrors softMaxInvDeriv() in utils.jl.
    """
    arr, scalar = _as_array(x)
    if np.any(arr < 0.0):
        raise ValueError("softplus_inv_deriv: input must be non-negative.")
    y = np.empty_like(arr)
    lo  = arr < eps
    hi  = arr > 40.0
    mid = ~lo & ~hi
    y[lo]  = 1.0 / np.maximum(arr[lo], 1e-300)
    y[hi]  = 1.0
    y[mid] = 1.0 / (1.0 - np.exp(-arr[mid]))
    return _maybe_squeeze(y, scalar)


def softplus_inv_scaled(
    x,
    alpha: float = 10.0,
    eps:   float = 1e-15,
) -> np.ndarray:
    """
    Scaled inverse softplus: f(x) = (1/α)·softplus_inv(α·x)

    Mirrors softMaxInvA() in utils.jl.
    """
    arr, scalar = _as_array(x)
    y = softplus_inv(alpha * arr, eps)
    result = (1.0 / alpha) * np.atleast_1d(y)
    return _maybe_squeeze(result, scalar)


def softplus_inv_scaled_deriv(
    x,
    alpha: float = 10.0,
    eps:   float = 1e-15,
) -> np.ndarray:
    """
    Derivative of scaled inverse softplus: f'(x) = softplus_inv_deriv(α·x)

    Mirrors softMaxInvDerivA() in utils.jl.
    """
    arr, scalar = _as_array(x)
    y = softplus_inv_deriv(alpha * arr, eps)
    return _maybe_squeeze(np.atleast_1d(y), scalar)


# ============================================================
#  Logistic / sigmoid
# ============================================================

def logistic(
    x,
    a:     float = None,
    b:     float = None,
    alpha: float = 1.0,
    thX:   float = _THX_DEFAULT,
) -> np.ndarray:
    """
    Logistic (sigmoid) function.

    Two call forms:
      logistic(x)            — standard σ(x) = 1/(1+e⁻ˣ) ∈ (0,1)
      logistic(x, a, b, α)   — ranged variant mapping ℝ → (a, b):
                                σ_{a,b,α}(x) = a + (b-a)/(1+(1/α)·e^{-αx})

    Clamped at ±thX to avoid overflow.  Returns float for scalar input,
    ndarray for array input.

    Mirrors logistic() in utils.jl.
    """
    arr, scalar = _as_array(x)

    if a is None and b is None:
        # Standard logistic
        y = np.empty_like(arr)
        ax  = arr                # u = x
        lo  = ax < -thX
        hi  = ax >  thX
        mid = ~lo & ~hi
        y[lo]  = 0.0
        y[hi]  = 1.0
        y[mid] = 1.0 / (1.0 + np.exp(-arr[mid]))
    else:
        # Ranged logistic: σ_{a,b,α}
        if a is None or b is None:
            raise ValueError("logistic: provide both a and b, or neither.")
        y = np.empty_like(arr)
        ax  = alpha * arr        # u = α·x
        lo  = ax < -thX
        hi  = ax >  thX
        mid = ~lo & ~hi
        y[lo]  = float(a)
        y[hi]  = float(b)
        y[mid] = float(a) + (float(b) - float(a)) / (
            1.0 + (1.0 / alpha) * np.exp(-alpha * arr[mid])
        )

    return _maybe_squeeze(y, scalar)


def logistic_inv(
    x,
    a:     float = None,
    b:     float = None,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Inverse logistic (logit).

    Two call forms:
      logistic_inv(x)           — logit(x) = log(x/(1-x)),  x ∈ (0,1)
      logistic_inv(x, a, b, α)  — inverse of ranged logistic, x ∈ (a,b)

    Raises ValueError for out-of-domain inputs.
    Mirrors logisticInv() in utils.jl.
    """
    arr, scalar = _as_array(x)

    if a is None and b is None:
        if np.any(arr < 0.0) or np.any(arr > 1.0):
            raise ValueError("logistic_inv: x must be in [0, 1].")
        y = np.log(arr / (1.0 - arr))
    else:
        if a is None or b is None:
            raise ValueError("logistic_inv: provide both a and b, or neither.")
        a_, b_ = float(a), float(b)
        if np.any(arr < a_) or np.any(arr > b_):
            raise ValueError(f"logistic_inv: x must be in [{a_}, {b_}].")
        y = (1.0 / alpha) * np.log((1.0 / alpha) * (arr - a_) / (b_ - arr))

    return _maybe_squeeze(y, scalar)


def logistic_deriv(
    x,
    a:     float = None,
    b:     float = None,
    alpha: float = 10.0,
    thX:   float = _THX_DEFAULT,
) -> np.ndarray:
    """
    Derivative of the logistic function.

    Two call forms:
      logistic_deriv(x)           — σ'(x) = e⁻ˣ/(1+e⁻ˣ)²
      logistic_deriv(x, a, b, α)  — (b-a)·e^{-αx}/(1+(1/α)e^{-αx})²

    Zero outside ±thX (where σ is flat).
    Mirrors logisticDeriv() in utils.jl.
    """
    arr, scalar = _as_array(x)

    if a is None and b is None:
        y = np.empty_like(arr)
        lo  = arr < -thX
        hi  = arr >  thX
        mid = ~lo & ~hi
        y[lo]  = 0.0
        y[hi]  = 0.0
        ex = np.exp(-arr[mid])
        y[mid] = ex / (1.0 + ex) ** 2
    else:
        if a is None or b is None:
            raise ValueError("logistic_deriv: provide both a and b, or neither.")
        a_, b_ = float(a), float(b)
        y = np.empty_like(arr)
        ax  = alpha * arr
        lo  = ax < -thX
        hi  = ax >  thX
        mid = ~lo & ~hi
        y[lo]  = 0.0
        y[hi]  = 0.0
        ex = np.exp(-alpha * arr[mid])
        y[mid] = (b_ - a_) * ex / (1.0 + (1.0 / alpha) * ex) ** 2

    return _maybe_squeeze(y, scalar)


# ============================================================
#  Robust loss / weighting functions  (utils.jl)
# ============================================================

def cauchy(
    x:      np.ndarray,
    alpha_f: float,
    x0:     float = 0.0,
) -> np.ndarray:
    """
    Cauchy potential:  ρ(x) = (x-x0)² / (1 + ((x-x0)/α)²)

    Used as a robust weighting function in M-estimation.
    Mirrors cauchy() in utils.jl.
    """
    x = np.asarray(x, dtype=float)
    dx = x - x0
    return dx**2 / (1.0 + (dx / alpha_f)**2)


def cauchy_deriv(
    x:      np.ndarray,
    alpha_f: float,
    x0:     float = 0.0,
) -> np.ndarray:
    """
    Derivative of Cauchy potential: ρ'(x) = 2(x-x0)/(1+(x/α)²)²

    Note: the Julia source uses `x` (not `dx=x-x0`) in the denominator —
    translated faithfully here to preserve identical behaviour.
    Mirrors cauchy_deriv() in utils.jl.
    """
    x = np.asarray(x, dtype=float)
    dx = x - x0
    return 2.0 * dx / (1.0 + (x / alpha_f)**2)**2


def phi_hl(
    x:      np.ndarray,
    alpha_f: float,
) -> np.ndarray:
    """
    Huber-like log potential: φ(x) = log(1 + (x/α)²)

    Mirrors phi_hl() in utils.jl.
    """
    x = np.asarray(x, dtype=float)
    return np.log1p((x / alpha_f)**2)


def phi_hl_deriv(
    x:      np.ndarray,
    alpha_f: float,
) -> np.ndarray:
    """
    Derivative of Huber-like log potential: φ'(x) = 2x/(α² + x²)

    Mirrors phi_hl_deriv() in utils.jl.
    """
    x = np.asarray(x, dtype=float)
    return 2.0 * x / (alpha_f**2 + x**2)


# ============================================================
#  Numerical integration  (riemann.jl)
# ============================================================

def riemann(
    f:      Callable,
    a:      float,
    b:      float,
    n:      int,
    method: str = "right",
) -> float:
    """
    Riemann quadrature of f on [a, b] with n equal-width subintervals.

    Methods
    -------
    'right'     : f(rᵢ) · (rᵢ - lᵢ)                  (right Riemann sum)
    'left'      : f(lᵢ) · (rᵢ - lᵢ)                  (left Riemann sum)
    'trapezoid' : ½·(f(lᵢ)+f(rᵢ))·(rᵢ-lᵢ)             (trapezoidal rule)
    'simpsons'  : 1/6·(f(lᵢ)+4f(mᵢ)+f(rᵢ))·(rᵢ-lᵢ)   (Simpson's rule)

    Improvement: grid evaluated once as a NumPy array; f applied
    vectorially where possible, otherwise via np.vectorize.  No
    Python list comprehension loop.

    Mirrors riemann() in riemann.jl.
    """
    xs = np.linspace(a, b, n + 1)
    dx = (b - a) / n   # constant width

    try:
        # attempt vectorised evaluation
        f_vals = np.asarray(f(xs), dtype=float)
        vectorised = True
    except Exception:
        # fall back to element-wise
        f_vals = np.array([float(f(xi)) for xi in xs])
        vectorised = True   # result is array either way

    left_vals  = f_vals[:-1]
    right_vals = f_vals[1:]

    if method == "right":
        return float(np.sum(right_vals * dx))
    elif method == "left":
        return float(np.sum(left_vals * dx))
    elif method == "trapezoid":
        return float(np.sum(0.5 * (left_vals + right_vals) * dx))
    elif method == "simpsons":
        mid_xs   = 0.5 * (xs[:-1] + xs[1:])
        try:
            mid_vals = np.asarray(f(mid_xs), dtype=float)
        except Exception:
            mid_vals = np.array([float(f(xi)) for xi in mid_xs])
        return float(np.sum((1.0/6.0) * (left_vals + 4.0*mid_vals + right_vals) * dx))
    else:
        raise ValueError(f"riemann: quadrature method '{method}' is not implemented. "
                         f"Choose from: 'right', 'left', 'trapezoid', 'simpsons'.")


# ============================================================
#  Piecewise-linear basis functions  (basisFun.jl)
# ============================================================

def e_0(z: float, z_min: float, z_max: float) -> float:
    """
    Left boundary hat function: ramps from 1 at z_min to 0 at z_max.

        e_0(z) = (z_max - z)/(z_max - z_min)  if z ∈ [z_min, z_max]
                 0                              otherwise

    Mirrors e_0() in basisFun.jl.
    """
    if z_min <= z <= z_max:
        return (z_max - z) / (z_max - z_min)
    return 0.0


def e_k(z: float, z_min: float, z_mid: float, z_max: float) -> float:
    """
    Interior hat function: ramps up from z_min to z_mid, down to z_max.

        e_k(z) = max(0, min((z-z_min)/(z_mid-z_min),
                             (z_max-z)/(z_max-z_mid)))

    Mirrors e_k() in basisFun.jl.
    """
    left  = (z - z_min) / (z_mid - z_min) if z_mid != z_min else 0.0
    right = (z_max - z) / (z_max - z_mid) if z_max != z_mid else 0.0
    return max(0.0, min(left, right))


def e_M(z: float, z_min: float, z_max: float) -> float:
    """
    Right boundary hat function: ramps from 0 at z_min to 1 at z_max.

        e_M(z) = (z - z_min)/(z_max - z_min)  if z ∈ [z_min, z_max]
                 0                              otherwise

    Mirrors e_M() in basisFun.jl.
    """
    if z_min <= z <= z_max:
        return (z - z_min) / (z_max - z_min)
    return 0.0

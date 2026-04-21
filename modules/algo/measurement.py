"""
modules/algo/measurement.py
============================
Pure-Python / NumPy translation of the BAYROSOL AeroMeas package:
  charge_probability.jl + DMA.jl + CPC.jl + SMPS3936.jl + SizeDistribution.jl

Original Julia code © 2021 Matthew Ozon, MIT License.
Python translation and improvements listed below.

SMPS-3936 signal chain
-----------------------
  particle size distribution u(s)  [#/m³/m]
        │
        ▼
  1. Impactor          – sigmoid cutoff for s > ~1 µm
        │
        ▼
  2. Kr-85 neutraliser – bipolar charger: for each charge state q,
                         compute electrical mobility K(s,q) and
                         charge-weighted density u_q(s) via
                         Wiedensohler 1988 polynomial approximation
        │
        ▼
  3. DMA classifier    – triangular transfer function in mobility space;
                         selects one mobility per channel; sums over q
        │
        ▼
  4. CPC detector      – sigmoid detection efficiency (lower cutoff ~10 nm)
        │
        ▼
  5. Counting          – φ · Δt · ∫ u_cpc(s) ds  →  Poisson draw → Y

The key public function is `smps3936_transfer_function` which returns
the (n_channels × n_sizes) kernel matrix A such that
  Y ≈ A @ u · Δs   (before Poisson noise)

Improvements over the original Julia code
-----------------------------------------
1.  `charge_probability` is fully vectorised over s (no Python loop).
    The Wiedensohler table lookup is expressed with NumPy polynomial
    evaluation (Horner) and boolean masks instead of per-element ifs.

2.  `size_from_mobility_and_charge` (Newton inversion) is vectorised
    and uses a convergence check instead of a fixed 20-iteration count.

3.  `DMA_size_density` builds the triangular window with vectorised
    boolean indexing (same logic, no inner loops over charge states).

4.  `smps3936_transfer_function` returns a named dataclass
    `SMPSKernel` instead of a bare array, making downstream code
    self-documenting.

5.  All functions accept both scalar and array diameter inputs
    using np.atleast_1d internally — no separate scalar overloads needed.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Union, Optional, Tuple
from numpy.random import default_rng

_rng = default_rng()   # module-level RNG; user can reseed via set_seed()

def set_seed(seed: int) -> None:
    """Reseed the module-level RNG (for reproducible simulations)."""
    global _rng
    _rng = default_rng(seed)


# ============================================================
#  Physical constants  (from charge_probability.jl)
# ============================================================

# Wiedensohler 1988 Table 1 coefficients: AiN[k, q+3] for q=-2..+2
# Columns: q=-2, q=-1, q=0, q=+1, q=+2  (index 0..4)
_AiN = np.array([
    [-26.3328, -2.3197, -0.0003, -2.3484, -44.4756],
    [ 35.9044,  0.6175, -0.1014,  0.6044,  79.3772],
    [-21.4608,  0.6201,  0.3073,  0.4800, -62.8900],
    [  7.0867, -0.1105, -0.3372,  0.0013,  26.4492],
    [ -1.3088, -0.1260,  0.1023, -0.1544,  -5.7480],
    [  0.1051,  0.0297, -0.0105,  0.0320,   0.5049],
], dtype=float)

# Wiedensohler 1988 Table 2: boundary values outside polynomial range
# Keys: (q, region)  where region: 'lo'=s<1nm, 'hi'=s>1µm
_BOUNDARY = {
     0: {'lo': 0.9909, 'hi': 0.1236},
     1: {'lo': 0.0044, 'hi': 0.1024},
    -1: {'lo': 0.0047, 'hi': 0.1333},
     2: {'lo': 0.0001, 'hi': 0.0759},
    -2: {'lo': 0.0001, 'hi': 0.1286},
}

E_CHARGE = 1.60217733e-19   # C
EPS_0    = 8.854187817e-12  # F/m
KB       = 1.380658e-23     # J/K
Z_P0     = 1.35e-4          # m²/(V·s)  positive ion mobility
Z_M0     = 1.60e-4          # m²/(V·s)  negative ion mobility
PR0      = 1.0e5            # Pa  reference pressure
Mair     = 28.96e-3         # kg/mol
Rg       = 8.314            # J/(mol·K)

# Cunningham slip coefficients (default set)
_ALPHA_CUN = 1.142
_BETA_CUN  = 0.558
_GAMMA_CUN = 0.999


# ============================================================
#  Cunningham slip correction  (charge_probability.jl)
# ============================================================

def cunningham_slip(Kn: np.ndarray,
                    alpha: float = _ALPHA_CUN,
                    beta:  float = _BETA_CUN,
                    gamma: float = _GAMMA_CUN) -> np.ndarray:
    """
    C_slip(Kn) = 1 + Kn * (α + β * exp(-γ/Kn))
    Mirrors C_slip() in charge_probability.jl.
    """
    return 1.0 + Kn * (alpha + beta * np.exp(-gamma / Kn))


def cunningham_slip_deriv(Kn: np.ndarray,
                          alpha: float = _ALPHA_CUN,
                          beta:  float = _BETA_CUN,
                          gamma: float = _GAMMA_CUN) -> np.ndarray:
    """dC_slip/dKn.  Mirrors C_slip_deriv() in charge_probability.jl."""
    return alpha + beta * (1.0 + gamma / Kn) * np.exp(-gamma / Kn)


# ============================================================
#  Electrical mobility  (charge_probability.jl)
# ============================================================

def _air_properties(T: float, Pr: float) -> Tuple[float, float]:
    """Returns (dyn_viscosity [Pa·s], gas_mean_free_path [m])."""
    eta   = 1.8e-5 * (T / 298.0) ** 0.85
    l_gas = 2.0 * eta / (Pr * np.sqrt(8.0 * Mair / (np.pi * Rg * T)))
    return eta, l_gas


def mobility_from_size_and_charge(
    s:  np.ndarray,
    q:  int,
    T:  float = 293.0,
    Pr: float = PR0,
) -> np.ndarray:
    """
    Electrical mobility Z_p(s, q) [m²/(V·s)].

    Z_p = q·e·C_slip(Kn) / (3π η s)

    Mirrors mobility_from_size_and_charge() in charge_probability.jl.
    Fully vectorised over s.
    """
    s = np.atleast_1d(np.asarray(s, dtype=float))
    eta, l_gas = _air_properties(T, Pr)
    Kn = 2.0 * l_gas / s
    return q * E_CHARGE * cunningham_slip(Kn) / (3.0 * np.pi * eta * s)


def mobility_from_size_and_charge_deriv(
    s:  np.ndarray,
    q:  int,
    T:  float = 293.0,
    Pr: float = PR0,
) -> np.ndarray:
    """dZ_p/ds.  Mirrors mobility_from_size_and_charge_deriv()."""
    s = np.atleast_1d(np.asarray(s, dtype=float))
    eta, l_gas = _air_properties(T, Pr)
    Kn      = 2.0 * l_gas / s
    Kn_d    = -2.0 * l_gas / s ** 2
    Cc      = cunningham_slip(Kn)
    Cc_d    = cunningham_slip_deriv(Kn)
    return (q * E_CHARGE / (3.0 * np.pi * eta)) * (
        (s * Kn_d * Cc_d - Cc) / s ** 2
    )


def size_from_mobility_and_charge(
    k:     np.ndarray,
    q:     int,
    T:     float = 293.0,
    Pr:    float = PR0,
    tol:   float = 1e-10,
    max_iter: int = 40,
) -> np.ndarray:
    """
    Invert Z_p(s,q) = k to find s [m] using Newton's method.

    Improvement: uses a convergence check (tol) instead of fixed 20 iters.
    Mirrors size_from_mobility_and_charge() in charge_probability.jl.
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    eta, l_gas = _air_properties(T, Pr)
    # solve: Kn · C_slip(Kn) = y  where y = 2 l_gas · (3π η k) / (q e)
    y = 2.0 * l_gas * 3.0 * np.pi * eta * np.abs(k) / (abs(q) * E_CHARGE)

    # f(Kn)  = Kn · C_slip(Kn) - y
    # f'(Kn) = 1 + 2α Kn + β(1 + 2Kn) exp(-γ/Kn)
    Kn = np.ones_like(k)
    for _ in range(max_iter):
        Cc  = cunningham_slip(Kn)
        Cc_d = cunningham_slip_deriv(Kn)
        f  = Kn * Cc - y
        fd = Cc + Kn * Cc_d   # = 1 + 2α Kn + β(1+2Kn)exp(-γ/Kn)
        delta = f / fd
        Kn -= delta
        Kn = np.maximum(Kn, 1e-6)   # keep Kn physical
        if np.max(np.abs(delta)) < tol:
            break

    return 2.0 * l_gas / Kn


# ============================================================
#  Bipolar charge probability  (charge_probability.jl)
# ============================================================

def charge_probability(
    q: int,
    s: np.ndarray,
    T:   float = 293.0,
    Z_p: float = Z_P0,
    Z_m: float = Z_M0,
) -> np.ndarray:
    """
    P(q | s) — conditional probability of q elementary charges on a
    particle of diameter s [m], after bipolar neutralisation.

    Implements:
    • Wiedensohler 1988 polynomial for |q| ≤ 1 (1–1000 nm)
    • Wiedensohler 1988 polynomial for |q| = 2 (20–1000 nm)
    • Gunn 1956 formula for |q| > 2

    Improvement: fully vectorised over s — no Python for-loop.
    Mirrors P_q_charges_knowing_size() in charge_probability.jl.

    Parameters
    ----------
    q   : number of elementary charges (signed integer)
    s   : particle diameters [m], shape (n,)
    T   : temperature [K]
    Z_p : positive ion mobility [m²/(V·s)]
    Z_m : negative ion mobility [m²/(V·s)]

    Returns
    -------
    val : shape (n,)
    """
    s   = np.atleast_1d(np.asarray(s, dtype=float))
    val = np.zeros(len(s))

    S_LO  = 1.0e-9   # 1 nm
    S_HI  = 1.0e-6   # 1 µm
    S_20  = 20.0e-9  # 20 nm

    if abs(q) <= 2:
        # ── Wiedensohler polynomial region ───────────────────────
        # column index in _AiN: q=-2→0, q=-1→1, q=0→2, q=1→3, q=2→4
        col = q + 2   # maps q ∈ {-2,-1,0,1,2} → col ∈ {0,1,2,3,4}
        coef = _AiN[:, col]   # (6,)

        # lower bound for polynomial validity
        s_poly_lo = S_20 if abs(q) == 2 else S_LO

        # masks
        in_range  = (s >= s_poly_lo) & (s <= S_HI)
        below     = s < s_poly_lo
        above     = s > S_HI

        # polynomial evaluation via Horner in log10(s/1nm)
        log_s_nm = np.log10(s * 1e9)   # log10(s [nm])
        # powers 0..5 broadcast: shape (6, n)
        powers = log_s_nm[None, :] ** np.arange(6)[:, None]
        poly   = coef @ powers   # shape (n,)  — dot over the 6 coefficients
        val[in_range]  = 10.0 ** poly[in_range]

        # boundary values (Wiedensohler 1988 Table 2)
        if q in _BOUNDARY:
            val[below] = _BOUNDARY[q]['lo']
            val[above] = _BOUNDARY[q]['hi']

    else:
        # ── Gunn 1956 formula for |q| > 2 ────────────────────────
        tmp1     = E_CHARGE / np.sqrt(4.0 * np.pi ** 2 * EPS_0 * s * KB * T)
        kappa    = 2.0 * np.pi * EPS_0 * s * KB * T / E_CHARGE ** 2
        q_star   = kappa * np.log(Z_p / Z_m)
        val      = tmp1 * np.exp(-(q - q_star) ** 2 / (2.0 * kappa))

    return val


# ============================================================
#  Impactor  (SMPS3936.jl)
# ============================================================

def impactor_efficiency(
    s:       np.ndarray,
    s50:     float = 1.0e-6,
    delta50: float = 0.1e-6,
) -> np.ndarray:
    """
    Sigmoid inertial impactor cutoff efficiency.
    η_imp(s) = 1 / (1 + exp((s - s50)/δ50))

    Mirrors impactor_eff() in SMPS3936.jl.
    """
    s = np.atleast_1d(np.asarray(s, dtype=float))
    return 1.0 / (1.0 + np.exp((s - s50) / delta50))


# ============================================================
#  Kr-85 neutraliser  (SMPS3936.jl)
# ============================================================

def neutraliser_Kr85(
    u_imp:  np.ndarray,
    s:      np.ndarray,
    T:      float = 293.0,
    Pr:     float = PR0,
    Nq:     int   = -6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Bipolar charger (Kr-85 source).

    For each charge state q in the range determined by Nq:
    • Compute electrical mobility K[|q|, :] = Z_p(s, q)
    • Compute charge probability P(q|s) via Wiedensohler/Gunn
    • Weighted density u_q_k[|q|, :] = P(q|s) · u_imp(s)

    Parameters
    ----------
    u_imp : size density after impactor, shape (n_sizes,)
    s     : particle diameters [m], shape (n_sizes,)
    Nq    : max charge magnitude (negative = use negative charges)

    Returns
    -------
    charge_range : array of charge values used, shape (|Nq|,)
    K            : mobilities, shape (|Nq|, n_sizes)
    u_q_k        : weighted densities, shape (|Nq|, n_sizes)
    Pqs          : charge probabilities, shape (|Nq|, n_sizes)

    Mirrors neutralizer_Kr_85() in SMPS3936.jl.
    """
    n_q    = abs(Nq)
    n_s    = len(s)
    sign   = 1 if Nq > 0 else -1
    charge_range = sign * np.arange(1, n_q + 1, dtype=int)

    K     = np.zeros((n_q, n_s))
    Pqs   = np.zeros((n_q, n_s))
    u_q_k = np.zeros((n_q, n_s))

    for idx, q in enumerate(charge_range):
        K[idx, :]     = mobility_from_size_and_charge(s, q, T=T, Pr=Pr)
        Pqs[idx, :]   = charge_probability(q, s, T=T)
        u_q_k[idx, :] = Pqs[idx, :] * u_imp

    return charge_range, K, u_q_k, Pqs


# ============================================================
#  DMA size density  (SMPS3936.jl)
# ============================================================


def triangle_tf(k: np.ndarray,
                k_star: float,
                half_width: float) -> np.ndarray:
    """
    Ideal triangular (non-diffusing) DMA transfer function.

        Psi(k) = max(0, 1 - |k - k_star| / half_width)

    This is the default DMA model (Knutson & Whitby 1975), corresponding
    to TF_model="" or TF_model="triangle" in DMA_size_density.

    Parameters
    ----------
    k          : mobility values, shape (n,)
    k_star     : centroid mobility selected by the DMA voltage
    half_width : triangular half-bandwidth dk = 0.5*(Qa+Qs)/(Qc+Qm)*k_star

    Returns
    -------
    Psi : shape (n,), values in [0, 1]
    """
    k = np.asarray(k, dtype=float)
    return np.maximum(0.0, 1.0 - np.abs(k - k_star) / abs(half_width))


def stolzenburg_ndtf(k:      np.ndarray,
                     k_star: float,
                     Qa:     float,
                     Qs:     float,
                     Qc:     float,
                     Qm:     float) -> np.ndarray:
    """
    Stolzenburg (1988) Non-Diffusing Transfer Function (NDTF).

    Accounts for finite aerosol and sample flow widths, producing a
    trapezoidal transfer function when Qa != Qs or Qc != Qm.
    For symmetric flows (Qa=Qs, Qc=Qm) this reduces to the ideal triangle.

    References: Stolzenburg 1988 PhD thesis; Stolzenburg & McMurry 2008,
    Aerosol Science and Technology 42:6, 421-432.

    Parameters
    ----------
    k      : mobility values, shape (n,)
    k_star : centroid mobility (set by DMA voltage)
    Qa, Qs : aerosol and sample flow rates [same units, e.g. L/min]
    Qc, Qm : classifier (sheath) and make-up (excess) flow rates

    Returns
    -------
    Psi_NDTF : shape (n,), values in [0, 1]
    """
    k      = np.asarray(k, dtype=float)
    k_abs  = abs(k_star)   # work in |mobility| space; sign handled below
    k_sign = np.sign(k_star) if k_star != 0.0 else 1.0
    k_work = k * k_sign    # flip negatives so k_work and k_abs are both positive
    Qt = Qc + Qm
    # normalised half-widths (Stolzenburg 2008, eq. 5-6)
    beta1 = (Qa + Qs) / (2.0 * Qt)           # total half-width / |k_star|
    beta2 = abs(Qc - Qm - Qa + Qs) / (2.0 * Qt)  # plateau half-width / |k_star|

    k_lo_outer = (1.0 - beta1) * k_abs
    k_lo_inner = (1.0 - beta2) * k_abs
    k_hi_inner = (1.0 + beta2) * k_abs
    k_hi_outer = (1.0 + beta1) * k_abs
    k_star = k_abs  # use abs from here on
    k = k_work      # use sign-flipped array

    Psi = np.zeros_like(k)
    Psi[(k >= k_lo_inner) & (k <= k_hi_inner)] = 1.0

    lo = (k >= k_lo_outer) & (k < k_lo_inner)
    if lo.any() and (k_lo_inner > k_lo_outer):
        Psi[lo] = (k[lo] - k_lo_outer) / (k_lo_inner - k_lo_outer)

    hi = (k > k_hi_inner) & (k <= k_hi_outer)
    if hi.any() and (k_hi_outer > k_hi_inner):
        Psi[hi] = (k_hi_outer - k[hi]) / (k_hi_outer - k_hi_inner)

    return Psi


def stolzenburg_dtf(k:      np.ndarray,
                    k_star: float,
                    Qa:     float,
                    Qs:     float,
                    Qc:     float,
                    Qm:     float,
                    G_dma:  float = 0.0,
                    D:      float = 0.0) -> np.ndarray:
    """
    Stolzenburg (1988) Diffusing Transfer Function (DTF).

    The DTF is the analytical convolution of the NDTF with a Gaussian
    broadening kernel whose width sigma = sqrt(G_dma * D / Qt) / k_star
    accounts for Brownian diffusion of particles in the DMA.

    Falls back to NDTF when D=0 or G_dma=0 (no diffusion), matching
    the TODO/stub in the Julia source.

    References: Stolzenburg (1988) PhD thesis, eq. 25/30-31;
    Stolzenburg & McMurry 2008, Aerosol Science and Technology 42:6.

    Parameters
    ----------
    k, k_star, Qa, Qs, Qc, Qm : same as stolzenburg_ndtf
    G_dma : DMA geometric diffusion factor G [dimensionless]
    D     : particle diffusivity [m^2/s]

    Returns
    -------
    Psi_DTF : shape (n,), values in [0, 1]
    """
    from scipy.special import erf as _erf
    k      = np.asarray(k, dtype=float)
    k_abs  = abs(k_star)
    k_sign = np.sign(k_star) if k_star != 0.0 else 1.0
    k      = k * k_sign    # work in absolute mobility space
    k_star = k_abs
    Qt = Qc + Qm

    if G_dma <= 0.0 or D <= 0.0:
        return stolzenburg_ndtf(k, k_star, Qa, Qs, Qc, Qm)

    beta1 = (Qa + Qs) / (2.0 * Qt)
    beta2 = abs(Qc - Qm - Qa + Qs) / (2.0 * Qt)
    sigma = np.sqrt(G_dma * D / Qt) / k_star   # dimensionless diffusion width

    def _Omega(x):
        # Omega(x) = x*erf(x/sqrt(2)) + sqrt(2/pi)*exp(-x^2/2)
        return (x * _erf(x / np.sqrt(2.0))
                + np.sqrt(2.0 / np.pi) * np.exp(-0.5 * x**2))

    # normalised mobility deviation
    xi   = (k / k_star - 1.0) / sigma
    b1p  = ( beta1 + beta2) / sigma
    b1m  = (-beta1 + beta2) / sigma
    b2p  = ( beta1 - beta2) / sigma
    b2m  = (-beta1 - beta2) / sigma

    Psi = (sigma / (2.0 * beta1)) * (
        _Omega(xi + b1p) - _Omega(xi + b1m)
        - _Omega(xi - b2p) + _Omega(xi - b2m)
    )
    return np.clip(Psi, 0.0, 1.0)


def dma_size_density(
    u_q_k:      np.ndarray,
    K:          np.ndarray,
    s:          np.ndarray,
    k_meas:     np.ndarray,
    sig_k_meas: np.ndarray,
    TF_model:   str   = "",
    Qa:         float = 0.3,
    Qs:         float = 0.3,
    Qc:         float = 3.0,
    Qm:         float = 3.0,
    G_dma:      float = 0.0,
    D_arr:      np.ndarray = None,
) -> np.ndarray:
    """
    Apply a DMA transfer function in mobility space and sum over charge states.

    Translates DMA_size_density() from the updated SMPS3936.jl, which adds
    support for multiple transfer function models via the TF_model parameter.

    Transfer function models
    ------------------------
    ""  or "triangle"     — ideal triangular TF (Knutson & Whitby 1975).
                            Half-bandwidth = 0.5*(Qa+Qs)/(Qc+Qm)*k_meas[i].
                            This is the default and was the only model in the
                            earlier version of this function.
    "Stolzenburg_NDTF"    — non-diffusing TF (Stolzenburg 1988).
                            Trapezoidal shape; reduces to triangle when
                            Qa=Qs and Qc=Qm.
    "Stolzenburg_DTF"     — diffusing TF (Stolzenburg 1988).
                            Requires G_dma and D_arr (particle diffusivity
                            per size bin).  Falls back to NDTF when
                            G_dma=0 or D_arr is None.

    Parameters
    ----------
    u_q_k      : charge-weighted density, shape (n_charges, n_sizes)
    K          : electrical mobilities per charge state, shape (n_charges, n_sizes)
    s          : diameters [m], shape (n_sizes,)   [unused in TF calc, kept for API]
    k_meas     : centroid mobilities per channel, shape (n_channels,)
    sig_k_meas : half-bandwidths per channel [used only for triangle model]
    TF_model   : one of "", "triangle", "Stolzenburg_NDTF", "Stolzenburg_DTF"
    Qa, Qs     : aerosol and sample flow rates [L/min or consistent units]
    Qc, Qm     : sheath and excess flow rates
    G_dma      : DMA geometric diffusion factor (DTF only)
    D_arr      : particle diffusivity [m^2/s] per size bin, shape (n_sizes,)
                 (DTF only; averaged over charge states for simplicity)

    Returns
    -------
    u_dma : size density at DMA outlet, shape (n_channels, n_sizes)

    Mirrors DMA_size_density() in the updated SMPS3936.jl.
    """
    n_ch = len(k_meas)
    n_s  = len(s)
    n_q  = K.shape[0]
    u_dma = np.zeros((n_ch, n_s))

    use_triangle = TF_model in ("", "triangle")
    use_ndtf     = TF_model == "Stolzenburg_NDTF"
    use_dtf      = TF_model == "Stolzenburg_DTF"

    if not (use_triangle or use_ndtf or use_dtf):
        raise ValueError(
            f"Unknown TF_model: '{TF_model}'. "
            "Choose '', 'triangle', 'Stolzenburg_NDTF', or 'Stolzenburg_DTF'."
        )

    for i in range(n_ch):
        k_c = float(k_meas[i])
        dk  = float(sig_k_meas[i])

        # build Psi: shape (n_q, n_s)
        Psi = np.zeros((n_q, n_s))
        for q in range(n_q):
            K_q = K[q, :]   # mobilities for charge state q, shape (n_s,)

            if use_triangle:
                # ideal triangle; half_width = 0.5*(Qa+Qs)/(Qc+Qm)*k_c
                hw = abs(0.5 * ((Qa + Qs) / (Qc + Qm)) * k_c)
                Psi[q, :] = triangle_tf(K_q, k_c, hw)

            elif use_ndtf:
                Psi[q, :] = stolzenburg_ndtf(K_q, k_c, Qa, Qs, Qc, Qm)

            else:  # DTF
                D_q = float(np.mean(D_arr)) if D_arr is not None else 0.0
                Psi[q, :] = stolzenburg_dtf(
                    K_q, k_c, Qa, Qs, Qc, Qm, G_dma, D_q
                )

        # sum over charge states: integral over dk of Psi * u_q
        u_dma[i, :] = np.sum(Psi * u_q_k, axis=0)

    return u_dma


# ============================================================
#  CPC detection efficiency  (SMPS3936.jl)
# ============================================================

def cpc_efficiency(
    s:       np.ndarray,
    s50:     float = 1.0e-8,
    delta50: float = 1.0e-9,
) -> np.ndarray:
    """
    Sigmoid CPC detection efficiency.
    η_cpc(s) = 1 / (1 + exp(-(s - s50)/δ50))

    Mirrors CPC_eff() in SMPS3936.jl.
    """
    s = np.atleast_1d(np.asarray(s, dtype=float))
    return 1.0 / (1.0 + np.exp(-(s - s50) / delta50))


def cpc_density(
    u_i_s:  np.ndarray,
    s:      np.ndarray,
    s50:    float = 1.0e-8,
    delta50: float = 1.0e-9,
) -> np.ndarray:
    """
    Apply CPC detection efficiency to each channel's density.

    u_cpc[i, :] = u_i_s[i, :] * η_cpc(s)

    Mirrors CPC_density() in SMPS3936.jl.  Vectorised.
    """
    eta = cpc_efficiency(s, s50=s50, delta50=delta50)   # (n_sizes,)
    return u_i_s * eta[None, :]                          # broadcast over channels


# ============================================================
#  Particle counting  (SMPS3936.jl)
# ============================================================

def nb_count(
    u:       np.ndarray,
    delta_s: np.ndarray,
    phi:     float = 1000.0,
    delta_t: float = 30.0,
) -> float:
    """
    Expected number of particles counted by the CPC.

        N = φ · Δt · Σ_s u(s) · Δs

    Parameters
    ----------
    u       : size density [#/m³/m], shape (n_sizes,)
    delta_s : bin widths [m], shape (n_sizes,)
    phi     : sample flow rate [cm³/s]  (note: Julia uses cm³, so caller
              must ensure consistent units)
    delta_t : counting time [s]

    Mirrors nb_count() in SMPS3936.jl.
    """
    return phi * delta_t * float(np.dot(u, delta_s))


def cpc_poisson_count(
    N_expected: float,
    phi:        float,
    delta_t:    float,
) -> float:
    """
    Draw a Poisson-distributed count and convert back to concentration.
    Y = Poisson(N_expected) / (φ · Δt)

    Mirrors the Poisson draw in SMPS3936() in SMPS3936.jl.
    """
    count = _rng.poisson(max(N_expected, 0.0))
    return count / (phi * delta_t)


# ============================================================
#  Simple DMA models  (DMA.jl / SizeDistribution.jl)
# ============================================================

def channel_efficiency_gate(
    bin_center: float,
    cst_ratio:  float,
    s:          np.ndarray,
) -> np.ndarray:
    """
    Gate (top-hat) DMA channel efficiency.
    η = 1 if s ∈ [bin_center/√r, bin_center·√r), else 0.

    Mirrors chanel_efficiency() in DMA.jl.
    """
    s = np.atleast_1d(np.asarray(s, dtype=float))
    lo = bin_center / np.sqrt(cst_ratio)
    hi = bin_center * np.sqrt(cst_ratio)
    return ((s >= lo) & (s < hi)).astype(float)


def channel_efficiency_gauss(
    bin_center: float,
    cst_ratio:  float,
    s:          np.ndarray,
) -> np.ndarray:
    """
    Gaussian DMA channel efficiency.
    σ = 0.5 · bin_center · (√r - 1/√r)

    Mirrors chanel_efficiency_gaus() in DMA.jl.
    """
    s    = np.atleast_1d(np.asarray(s, dtype=float))
    sig  = 0.5 * bin_center * (np.sqrt(cst_ratio) - 1.0 / np.sqrt(cst_ratio))
    return np.exp(-0.5 * ((s - bin_center) / sig) ** 2)


def dma_operator(
    psd:          np.ndarray,
    particle_s:   np.ndarray,
    delta_s:      np.ndarray,
    bin_centers:  np.ndarray,
    cst_ratio:    float,
    model:        str = "gauss",
) -> np.ndarray:
    """
    Simple DMA measurement operator: Riemann integration of
    psd × channel_efficiency for each bin centre.

    Mirrors DMA_gaus() / DMA_gate() in DMA.jl.
    """
    n_ch  = len(bin_centers)
    conc  = np.zeros(n_ch)
    for k, bc in enumerate(bin_centers):
        if model == "gauss":
            eta = channel_efficiency_gauss(bc, cst_ratio, particle_s)
        else:
            eta = channel_efficiency_gate(bc, cst_ratio, particle_s)
        conc[k] = np.dot(psd * eta, delta_s)
    return conc


def dma_operator_matrix(
    particle_s:  np.ndarray,
    delta_s:     np.ndarray,
    bin_centers: np.ndarray,
    cst_ratio:   float,
    model:       str = "gauss",
) -> np.ndarray:
    """
    Build the (n_channels × n_sizes) measurement operator matrix M
    such that conc = M @ psd.

    Mirrors DMA_gaus_OP() in DMA.jl.  Vectorised.
    """
    n_ch  = len(bin_centers)
    n_s   = len(particle_s)
    M     = np.zeros((n_ch, n_s))
    for k, bc in enumerate(bin_centers):
        if model == "gauss":
            eta = channel_efficiency_gauss(bc, cst_ratio, particle_s)
        else:
            eta = channel_efficiency_gate(bc, cst_ratio, particle_s)
        M[k, :] = eta * delta_s
    return M


# ============================================================
#  SMPS-3936 kernel and forward model
# ============================================================

@dataclass
class SMPSKernel:
    """
    Precomputed SMPS-3936 transfer kernel A.

    A[i, j] is the fractional contribution of size bin j to channel i,
    already multiplied by the CPC efficiency and summed over charge states.
    Applying A to a size density vector u gives the expected (noiseless)
    concentration at each channel outlet.

        conc_channel = A @ u   [#/m³]

    To get expected counts:
        N_counts[i] = phi * dt * sum(A[i,:] * delta_s)  * u_concentration
    """
    A:           np.ndarray   # (n_channels, n_sizes)
    s:           np.ndarray   # size grid [m], (n_sizes,)
    s_meas:      np.ndarray   # channel centres [m], (n_channels,)
    k_meas:      np.ndarray   # selected mobilities, (n_channels,)
    Pqs:         np.ndarray   # charge fractions, (n_charges, n_sizes)
    charge_range: np.ndarray  # charge values used


def smps3936_transfer_function(
    s:           np.ndarray,
    s_meas:      np.ndarray,
    cst_r_meas:  float,
    s50_imp:     float = 1.0e-6,
    delta50_imp: float = 0.1e-6,
    s50_cpc:     float = 1.0e-8,
    delta50_cpc: float = 1.0e-9,
    T:           float = 293.0,
    Pr:          float = PR0,
    Nq:          int   = -6,
    q_a:         float = 0.3,
    q_sh:        float = 3.0,
    TF_model:    str   = "",
    Qm:          float = None,
    G_dma:       float = 0.0,
) -> SMPSKernel:
    """
    Compute the SMPS-3936 transfer kernel matrix A.

    This is the main function for building the measurement model used
    in data inversion.  A unit size distribution (all ones) is passed
    through the full signal chain; the resulting (n_channels × n_sizes)
    matrix is A.

    Parameters
    ----------
    s           : true size grid [m], shape (n_sizes,)
    s_meas      : channel centre diameters [m], shape (n_channels,)
    cst_r_meas  : geometric ratio between adjacent channels
    s50_imp     : impactor 50% cut diameter [m]
    delta50_imp : impactor sigmoid width [m]
    s50_cpc     : CPC 50% cut diameter [m]
    delta50_cpc : CPC sigmoid width [m]
    T           : temperature [K]
    Pr          : pressure [Pa]
    Nq          : number of charge states (negative = negative charges)
    q_a         : aerosol flow [L/min]  (ratio q_a/q_sh = DMA β)
    q_sh        : sheath  flow [L/min]

    Returns
    -------
    SMPSKernel dataclass

    Mirrors SMPS3936_transfer_function() in SMPS3936.jl.
    Improvement: returns a named dataclass instead of a bare array.

    Additional parameters vs the original
    --------------------------------------
    TF_model : DMA transfer function model — "", "Stolzenburg_NDTF",
               or "Stolzenburg_DTF".  Default "" = ideal triangle.
    Qm       : make-up (excess) flow [L/min].  Defaults to q_sh
               (symmetric instrument: Qm = Qc = q_sh).
    G_dma    : DMA geometric diffusion factor (DTF only).
    """
    s      = np.asarray(s,      dtype=float)
    s_meas = np.asarray(s_meas, dtype=float)

    # --- selected mobilities and half-bandwidths -------------------------
    q_sign  = 1 if Nq > 0 else -1
    k_meas  = mobility_from_size_and_charge(s_meas, q_sign, T=T, Pr=Pr)
    dk_meas = np.abs(0.5 * (q_a / q_sh) * k_meas)

    # --- pass unit distribution through the instrument -------------------
    u_unit = np.ones(len(s))

    # 1. impactor
    u_imp = u_unit * impactor_efficiency(s, s50=s50_imp, delta50=delta50_imp)

    # 2. neutraliser
    charge_range, K, u_q_k, Pqs = neutraliser_Kr85(
        u_imp, s, T=T, Pr=Pr, Nq=Nq
    )

    # 3. DMA transfer function (model selected by TF_model)
    Qm_ = q_sh if Qm is None else Qm
    u_dma = dma_size_density(
        u_q_k, K, s, k_meas, dk_meas,
        TF_model=TF_model,
        Qa=q_a, Qs=q_a, Qc=q_sh, Qm=Qm_,
        G_dma=G_dma,
    )

    # 4. CPC detection efficiency
    A = cpc_density(u_dma, s, s50=s50_cpc, delta50=delta50_cpc)

    return SMPSKernel(
        A=A,
        s=s,
        s_meas=s_meas,
        k_meas=k_meas,
        Pqs=Pqs,
        charge_range=charge_range,
    )


def smps3936_forward(
    u:           np.ndarray,
    s:           np.ndarray,
    delta_s:     np.ndarray,
    s_meas:      np.ndarray,
    cst_r_meas:  float,
    phi:         float,
    dt:          float,
    kernel:      Optional[SMPSKernel] = None,
    add_noise:   bool = True,
    **kernel_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full SMPS-3936 forward model: from size distribution to noisy counts.

    Parameters
    ----------
    u           : size density [#/m³/m], shape (n_sizes,) or (n_sizes, n_t)
                  If 2-D, time-averaged before processing.
    s           : size grid [m]
    delta_s     : bin widths [m]
    s_meas      : channel centres [m]
    cst_r_meas  : geometric ratio between channels
    phi         : sample flow rate [cm³/s]
    dt          : counting time per channel [s]
    kernel      : pre-computed SMPSKernel (computed if None)
    add_noise   : if True, add Poisson counting noise
    **kernel_kwargs : passed to smps3936_transfer_function if kernel is None

    Returns
    -------
    Y        : noisy concentration per channel [#/m³], shape (n_channels,)
    N_counts : expected counts per channel (before noise), shape (n_channels,)

    Mirrors SMPS3936() in SMPS3936.jl.
    """
    u = np.asarray(u, dtype=float)

    # time-average if 2-D
    if u.ndim == 2:
        if u.shape[0] == len(s):
            u_avg = u.mean(axis=1)
        else:
            u_avg = u.mean(axis=0)
    else:
        u_avg = u

    # build kernel if not provided
    if kernel is None:
        kernel = smps3936_transfer_function(
            s, s_meas, cst_r_meas, **kernel_kwargs
        )

    # expected concentration per channel (kernel already encodes
    # impactor + charger + DMA + CPC efficiency)
    conc_expected = kernel.A @ u_avg   # (n_channels,)

    # expected counts
    N_counts = phi * dt * (kernel.A @ (u_avg * delta_s))

    # Poisson draw
    if add_noise:
        Y = np.array([
            cpc_poisson_count(N, phi, dt)
            for N in N_counts
        ])
    else:
        Y = conc_expected

    return Y, N_counts


# ============================================================
#  Simple DMPS  (SizeDistribution.jl)
# ============================================================

def dmps_forward(
    psd:         np.ndarray,
    particle_s:  np.ndarray,
    delta_s:     np.ndarray,
    bin_centers: np.ndarray,
    cst_ratio:   float,
    volume:      float,
    model:       str  = "gauss",
    add_noise:   bool = True,
) -> np.ndarray:
    """
    Simple DMPS (stepping, not scanning) forward model.
    Applies DMA then Poisson CPC counting.

    Mirrors DMPS_gaus() / DMPS_gate() in SizeDistribution.jl.
    """
    conc = dma_operator(psd, particle_s, delta_s, bin_centers,
                        cst_ratio, model=model)
    if add_noise:
        return np.array([
            _rng.poisson(max(c * volume, 0.0)) / volume
            for c in conc
        ], dtype=float)
    return conc

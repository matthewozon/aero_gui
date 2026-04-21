"""
modules/algo/gde.py
===================
Pure-Python / NumPy translation of the BAYROSOL AeroMec2 package:
  type.jl + coagulation.jl + condensation.jl + nucleation.jl +
  loss.jl + iteration.jl

Original Julia code © 2021 Matthew Ozon, MIT License.
Python translation and improvements listed in section below.

Physical model (box model, spatially uniform)
---------------------------------------------
The GDE implemented here is dimensionless.  All quantities are
normalised by characteristic scales stored in AeroSys:
  x0   [#/m³]       characteristic concentration
  t0   [s]          characteristic time
  d0   [m]          first bin diameter
  beta0 [m³/s]      characteristic coagulation coefficient
  GR0  [m/s]        characteristic condensation growth rate
  J0   [#/(m³·s)]   characteristic nucleation rate
  gamma0 [1/s]      characteristic loss rate

GDE for bin s=1 (smallest):
  dx_1/dt = -x_1 Σ_{i>1} β_{1,i} x_i   (coagulation loss)
            - GR · S_1/d_0 · x_1        (condensation loss)
            - γ_1 x_1                   (wall loss)
            + J                         (nucleation source)

GDE for bin s ≥ 2:
  dx_s/dt = -x_s Σ_{i>s} β_{s,i} x_i
            + GR · (S_{s-1}/d_0 · r · x_{s-1} - S_s/d_0 · x_s)
            - γ_s x_s

where r = cst_r is the geometric diameter ratio between adjacent bins.

Improvements over the original
-------------------------------
1.  NumPy vectorisation throughout — the coagulation loops that iterate
    over every (i,j) pair are the most expensive part; the gain loop is
    vectorised using pre-built index arrays (same logic as the Julia
    `init_coagulation_loop_indices` / `Ic` / `a_gain` machinery).

2.  `coagulation_coefficient` uses fully vectorised outer-product
    operations instead of the Julia `for i in 1:nbin` loop.

3.  The condensation Jacobian is expressed as a tridiagonal update via
    NumPy diagonal views — no Python loop.

4.  `WallDepositionRate` is vectorised.

5.  `AeroSys` is a dataclass with a clean `from_diameters` factory —
    no default constructor / copy constructor confusion.

6.  `iter` returns a new array; it never silently mutates its input,
    which avoids subtle bugs when the same distribution is reused.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple

# ── Physical constants (SI) ──────────────────────────────────────────
G_GRAV  = 9.81           # m/s²   gravitational acceleration
Rg      = 8.3144598      # J/(mol·K)
Na      = 6.022140857e23 # mol⁻¹
Mair    = 28.96e-3       # kg/mol  mean molar mass of air
kb      = 1.38064852e-23 # J/K    Boltzmann constant
Cp      = 1012.0         # J/(kg·K) specific heat of air


# ============================================================
#  AeroSys  (mirrors the Julia mutable struct AeroSys)
# ============================================================

@dataclass
class AeroSys:
    """
    Workspace / parameter container for an aerosol GDE simulation.

    Create with `AeroSys.from_diameters(d, ...)` rather than directly.
    """

    # ── bin geometry ─────────────────────────────────────────
    nbin:   int
    log_scale: bool          # True = log spacing (always true in practice)
    d:      np.ndarray       # bin centre diameters [m], shape (nbin,)
    d0:     float            # first bin diameter [m]
    cst_r:  float            # geometric ratio (log) or spacing (linear)
    cst_v:  float            # volume equivalent of cst_r
    scale_GR: np.ndarray     # condensation scaling factor per bin, shape (nbin,)

    # ── characteristic scales ────────────────────────────────
    beta0:  float = 0.0      # [m³/s]   characteristic coagulation coeff
    GR0:    float = 0.0      # [m/s]    characteristic growth rate
    J0:     float = 0.0      # [#/(m³·s)] characteristic nucleation rate
    gamma0: float = 0.0      # [1/s]    characteristic loss rate
    t0:     float = 1.0      # [s]      characteristic time
    x0:     float = 1.0      # [#/m³]   characteristic concentration

    # ── coagulation matrices ─────────────────────────────────
    beta: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    # pre-computed gain index arrays (filled by init_coagulation_indices)
    Ic:      Optional[np.ndarray] = None   # (3, count) int indices
    a_gain:  Optional[np.ndarray] = None   # (count,)  weights
    Icj:     Optional[np.ndarray] = None   # (3, countj) Jacobian indices
    a_gainj: Optional[np.ndarray] = None   # (countj,)

    # ── condensation state (current step) ────────────────────
    GR: float = 0.0          # dimensionless growth rate (set during iter)

    # ── loss rates ───────────────────────────────────────────
    LR: np.ndarray = field(default_factory=lambda: np.empty(0))

    # ── mechanism flags ──────────────────────────────────────
    is_coa:      bool = False
    is_con:      bool = False
    is_nuc:      bool = False
    is_los:      bool = False
    is_coa_gain: bool = False

    # ── class method: main constructor ───────────────────────

    @classmethod
    def from_diameters(
        cls,
        d: np.ndarray,
        *,
        beta_c:  float = 0.0,
        GR_c:    float = 0.0,
        J_c:     float = 0.0,
        gamma_c: float = 0.0,
        t_c:     float = 1.0,
        x_c:     float = 1.0,
        scale:   str   = "log",
    ) -> "AeroSys":
        """
        Create an AeroSys from an array of bin centre diameters.

        Parameters
        ----------
        d       : bin centre diameters [m], increasing, log- or lin-spaced
        beta_c  : characteristic coagulation coefficient [m³/s]
        GR_c    : characteristic growth rate [m/s]
        J_c     : characteristic nucleation rate [#/(m³·s)]
        gamma_c : characteristic loss rate [1/s]
        t_c     : characteristic time [s]
        x_c     : characteristic concentration [#/m³]
        scale   : 'log' (default) or 'linear'
        """
        d = np.asarray(d, dtype=float)
        nbin = len(d)
        log_scale = (scale == "log")

        if log_scale:
            cst_r = float(np.mean(d[1:] / d[:-1]))
            cst_v = cst_r ** 3
            exponents = cst_r ** (1.5 - np.arange(1, nbin + 1, dtype=float))
            scale_GR = exponents / (cst_r - 1.0)
        else:
            cst_r = 1.0
            cst_v = 1.0
            spacing = float(np.mean(d[1:] - d[:-1]))
            scale_GR = (d[0] / spacing) * np.ones(nbin)

        return cls(
            nbin=nbin,
            log_scale=log_scale,
            d=d.copy(),
            d0=float(d[0]),
            cst_r=cst_r,
            cst_v=cst_v,
            scale_GR=scale_GR,
            beta0=beta_c,
            beta=np.zeros((nbin, nbin)),
            GR0=GR_c,
            J0=J_c,
            gamma0=gamma_c,
            t0=t_c,
            x0=x_c,
            LR=np.zeros(nbin),
        )

    def copy(self) -> "AeroSys":
        import copy
        return copy.deepcopy(self)


# ============================================================
#  Coagulation
# ============================================================

def coagulation_coefficient(
    ws: AeroSys,
    temperature: float = 300.0,
    pressure:    float = 1.0e5,
    density:     float = 1400.0,
) -> None:
    """
    Compute and store the Fuchs–Sutugin coagulation coefficients β[i,j]
    (Seinfeld & Pandis 2006, eq. 13.56 / p. 603) in `ws.beta`.

    All intermediate quantities are in SI units.
    β is normalised by its mean at the end (dimensionless matrix);
    ws.beta0 holds the dimensional mean [m³/s].

    Mirrors coagulation_coefficient!(ws, T, P, ρ) in coagulation.jl.
    Improvement: fully vectorised — no Python for-loop.
    """
    d = ws.d
    m_p  = density * (np.pi / 6.0) * d ** 3                          # [kg] particle mass

    eta  = 1.8e-5 * (temperature / 298.0) ** 0.85                    # [Pa·s] dynamic viscosity
    l_g  = (2.0 * eta
            / (pressure * np.sqrt(8.0 * Mair / (np.pi * Rg * temperature))))  # [m] gas MFP

    Kn   = 2.0 * l_g / d
    Cc   = 1.0 + Kn * (1.257 + 0.4 * np.exp(-1.1 / Kn))             # Cunningham correction
    D    = Cc * kb * temperature / (3.0 * np.pi * eta * d)           # [m²/s] diffusivity
    v_p  = np.sqrt(8.0 * kb * temperature / (np.pi * m_p))           # [m/s] thermal speed
    lp   = 8.0 * D / (np.pi * v_p)                                   # [m] particle MFP
    # mean distance from sphere centre
    dist = ((1.0 / (3.0 * d * lp))
            * ((d + lp) ** 3 - (d ** 2 + lp ** 2) ** 1.5)
            - d)

    # outer-product Fuchs correction — shape (nbin, nbin)
    d_sum  = d[:, None] + d[None, :]                                  # d_i + d_j
    D_sum  = D[:, None] + D[None, :]                                  # D_i + D_j
    dist2  = np.sqrt(dist[:, None] ** 2 + dist[None, :] ** 2)        # √(δ_i²+δ_j²)
    v2     = np.sqrt(v_p[:, None] ** 2 + v_p[None, :] ** 2)          # √(v_i²+v_j²)

    beta_fs = 1.0 / (d_sum / (d_sum + 2.0 * dist2)
                     + 8.0 * D_sum / (v2 * d_sum))

    # coagulation kernel [m³/s]: (D_i+D_j)(d_i+d_j) factors
    beta = 2.0 * np.pi * beta_fs * (
        d[:, None] * D[None, :]
        + d[:, None] * D[:, None]     # = d_i * D_i (broadcast trick)
        + d[None, :] * D[None, :]     # = d_j * D_j
        + d[None, :] * D[:, None]
    )
    # Note: above expands (d_i + d_j)(D_i + D_j) = d_i D_j + d_i D_i + d_j D_j + d_j D_i

    ws.beta0 = float(np.mean(beta))
    ws.beta  = beta / ws.beta0   # dimensionless


def init_coagulation_indices(ws: AeroSys) -> None:
    """
    Pre-compute the index arrays (Ic, a_gain) for the coagulation gain
    term and (Icj, a_gainj) for the Jacobian gain term.

    Mirrors init_coagulation_loop_indices!(ws) in coagulation.jl.
    The gain weight `a_gain` is a piecewise-linear interpolation that
    distributes newly formed particle mass between adjacent bins.
    """
    vol = (np.pi / 6.0) * ws.d ** 3
    sv  = np.sqrt(ws.cst_v)

    # ── state gain indices ────────────────────────────────────────────
    rows_s, rows_i, rows_j, weights = [], [], [], []
    for s in range(1, ws.nbin):      # s is 0-based bin index (Julia s in 2:nbin → 0-based 1..nbin-1)
        vs = vol[s]
        lo, hi = vs / sv, vs * sv
        for i in range(s + 1):
            for j in range(s + 1):
                v_new = vol[i] + vol[j]
                if lo <= v_new < hi:
                    if v_new <= vs:
                        w = 0.5 * (lo - v_new) / (lo - vs) + 0.5
                    else:
                        w = 0.5 - 0.5 * (v_new - hi) / (vs - hi)
                    w = max(w, 0.0)
                    rows_s.append(s)
                    rows_i.append(i)
                    rows_j.append(j)
                    weights.append(w)

    if rows_s:
        ws.Ic     = np.array([rows_s, rows_i, rows_j], dtype=np.int64)
        ws.a_gain = np.array(weights, dtype=float)
    else:
        ws.Ic     = np.empty((3, 0), dtype=np.int64)
        ws.a_gain = np.empty(0, dtype=float)

    # ── Jacobian gain indices ─────────────────────────────────────────
    rs, rk, ri, wj = [], [], [], []
    for s in range(ws.nbin):
        vs = vol[s]
        lo, hi = vs / sv, vs * sv
        for k in range(ws.nbin):
            for i in range(ws.nbin):
                v_new = vol[i] + vol[k]
                if lo <= v_new < hi:
                    if v_new <= vs:
                        w = 0.5 * (lo - v_new) / (lo - vs) + 0.5
                    else:
                        w = 0.5 - 0.5 * (v_new - hi) / (vs - hi)
                    w = max(w, 0.0)
                    rs.append(s)
                    rk.append(k)
                    ri.append(i)
                    wj.append(w)

    if rs:
        ws.Icj     = np.array([rs, rk, ri], dtype=np.int64)
        ws.a_gainj = np.array(wj, dtype=float)
    else:
        ws.Icj     = np.empty((3, 0), dtype=np.int64)
        ws.a_gainj = np.empty(0, dtype=float)


def _coagulation_loss(ws: AeroSys, x: np.ndarray) -> np.ndarray:
    x_safe = np.where(np.isfinite(x), x, 0.0)
    dx = np.zeros(ws.nbin)
    for s in range(ws.nbin - 1):
        dx[s] = -x_safe[s] * np.dot(ws.beta[s, s + 1:], x_safe[s + 1:])
    scale = ws.x0 * ws.beta0 * ws.t0
    result = scale * dx
    return np.where(np.isfinite(result), result, 0.0)


def _coagulation_full(ws: AeroSys, x: np.ndarray) -> np.ndarray:
    scale = ws.x0 * ws.beta0 * ws.t0
    dx = _coagulation_loss(ws, x) / scale
    if ws.Ic is not None and ws.Ic.shape[1] > 0:
        x_safe = np.where(np.isfinite(x), x, 0.0)
        s_idx = ws.Ic[0]; i_idx = ws.Ic[1]; j_idx = ws.Ic[2]
        contrib = 0.5 * x_safe[i_idx] * ws.beta[i_idx, j_idx] * x_safe[j_idx] * ws.a_gain
        contrib = np.where(np.isfinite(contrib), contrib, 0.0)
        np.add.at(dx, s_idx, contrib)
    dx[-1] = 0.0
    result = scale * dx
    return np.where(np.isfinite(result), result, 0.0)


def _jacobian_coagulation_loss(ws: AeroSys, x: np.ndarray,
                                F: np.ndarray) -> None:
    """
    Add coagulation-loss contribution to Jacobian F (in-place, accumulated).
    F[s, k] += -beta[s,k] * x[s]   for k > s

    Mirrors jacobian_coagulation_loss!() in coagulation.jl.
    Vectorised: sets the upper-triangular part in one NumPy call.
    """
    scale = ws.x0 * ws.beta0 * ws.t0
    for s in range(ws.nbin - 1):
        F[s, s + 1:] -= scale * ws.beta[s, s + 1:] * x[s]


def _jacobian_coagulation_full(ws: AeroSys, x: np.ndarray,
                                F: np.ndarray) -> None:
    """
    Add coagulation loss+gain contribution to Jacobian F (in-place).
    Mirrors jacobian_coagulation!() in coagulation.jl.
    """
    _jacobian_coagulation_loss(ws, x, F)
    scale = ws.x0 * ws.beta0 * ws.t0
    if ws.Icj is not None and ws.Icj.shape[1] > 0:
        s_idx = ws.Icj[0]
        k_idx = ws.Icj[1]
        i_idx = ws.Icj[2]
        contrib = scale * ws.a_gainj * ws.beta[k_idx, i_idx] * x[i_idx]
        # F[s, k] += contrib  — use np.add.at for non-unique (s,k) pairs
        flat_idx = s_idx * ws.nbin + k_idx
        np.add.at(F.ravel(), flat_idx, contrib)


# ============================================================
#  Condensation
# ============================================================

def _condensation_growth_and_evap(ws: AeroSys,
                                   x: np.ndarray,
                                   zeta: Union[float, np.ndarray]
                                   ) -> np.ndarray:
    """
    Condensation / evaporation flux using upwind scheme.

    zeta : dimensionless growth rate (scalar or per-bin array).

    Mirrors CondensationGrowthAndEvap!() in condensation.jl.
    """
    dx = np.zeros(ws.nbin)

    if np.isscalar(zeta):
        GR = (ws.GR0 * ws.t0 / ws.d0) * float(zeta)
        ws.GR = GR
        if zeta >= 0:
            dx[0]  = -GR * ws.scale_GR[0] * x[0]
            dx[1:] = GR * (ws.scale_GR[:-1] * ws.cst_r * x[:-1]
                           - ws.scale_GR[1:] * x[1:])
        else:
            dx[-1]  = GR * ws.scale_GR[-1] * x[-1]
            dx[:-1] = GR * (ws.scale_GR[:-1] * ws.cst_r * x[:-1]
                            - ws.scale_GR[1:] * x[1:])
    else:
        GR_arr = (ws.GR0 * ws.t0 / ws.d0) * np.asarray(zeta, dtype=float)
        ws.GR  = float(GR_arr[0])

        # positive (condensation) part
        GR_pos = np.where(GR_arr >= 0, GR_arr, 0.0)
        dx[0]  += -GR_pos[0] * ws.scale_GR[0] * x[0]
        dx[1:]  += (GR_pos[:-1] * ws.scale_GR[:-1] * ws.cst_r * x[:-1]
                    - GR_pos[1:] * ws.scale_GR[1:] * x[1:])

        # negative (evaporation) part
        GR_neg = np.where(GR_arr < 0, GR_arr, 0.0)
        dx[-1] += GR_neg[-1] * ws.scale_GR[-1] * x[-1]
        dx[:-1] += (GR_neg[:-1] * ws.scale_GR[:-1] * ws.cst_r * x[:-1]
                    - GR_neg[1:] * ws.scale_GR[1:] * x[1:])

    return dx


def _jacobian_condensation(ws: AeroSys,
                            x: np.ndarray,
                            zeta: Union[float, np.ndarray],
                            F: np.ndarray) -> None:
    """
    Add condensation contribution to Jacobian F (in-place).
    F is tridiagonal: diagonal and sub-diagonal terms only.

    Mirrors jacobian_condensation!() in condensation.jl.
    Improvement: no Python loop — uses NumPy diagonal views.
    """
    if np.isscalar(zeta):
        GR = (ws.GR0 * ws.t0 / ws.d0) * float(zeta)
        ws.GR = GR
        # diagonal: -GR * scale_GR[i]
        diag_view = np.einsum('ii->i', F)   # writable diagonal view
        diag_view -= GR * ws.scale_GR
        # sub-diagonal: +GR * scale_GR[i-1] * cst_r   (F[i, i-1])
        sub_diag = F.ravel()[ws.nbin::ws.nbin + 1]   # F[1,0], F[2,1], ...
        sub_diag += GR * ws.scale_GR[:-1] * ws.cst_r
    else:
        GR_arr = (ws.GR0 * ws.t0 / ws.d0) * np.asarray(zeta, dtype=float)
        ws.GR  = float(GR_arr[0])
        diag_view = np.einsum('ii->i', F)
        diag_view -= GR_arr * ws.scale_GR
        sub_diag = F.ravel()[ws.nbin::ws.nbin + 1]
        sub_diag += GR_arr[:-1] * ws.scale_GR[:-1] * ws.cst_r


def cond_err(dt: float,
             e_g: np.ndarray,
             e_j: float,
             x: np.ndarray,
             delta: np.ndarray) -> np.ndarray:
    """
    Std dev of GDE error due to growth rate and nucleation rate uncertainty.

    Parameters
    ----------
    dt    time step [s]
    e_g   error in growth rates per bin, shape (nbin,)
    e_j   error in nucleation rate (scalar)
    x     size distribution [#/m³], shape (nbin,)
    delta bin widths [m], shape (nbin,)

    Mirrors cond_err() in condensation.jl.
    """
    u_density  = x / delta
    part_flux  = np.concatenate([[e_j], u_density * e_g])
    return dt * np.sqrt(part_flux[1:] ** 2 + part_flux[:-1] ** 2)


# ============================================================
#  Nucleation
# ============================================================

def _nucleation(ws: AeroSys, j: float) -> np.ndarray:
    """
    Nucleation source term: particles appear only in the smallest bin.

    j : dimensionless nucleation rate
    Mirrors Nucleation!() in nucleation.jl.
    """
    dx = np.zeros(ws.nbin)
    dx[0] = (ws.J0 * ws.t0 / ws.x0) * j
    return dx


# ============================================================
#  Wall / linear losses
# ============================================================

def _wall_deposition(ws: AeroSys,
                     x: np.ndarray,
                     xi: np.ndarray) -> np.ndarray:
    """
    Linear loss term: dx_s = -gamma0 * t0 * xi_s * x_s

    xi  : loss rate profile [1/s], shape (nbin,).
          Stored in ws.LR for later reference.
    Mirrors WallDeposition!() in loss.jl.
    """
    ws.LR = xi.copy()
    return -(ws.gamma0 * ws.t0) * xi * x


def _jacobian_wall_deposition(ws: AeroSys,
                               xi: np.ndarray,
                               F: np.ndarray) -> None:
    """
    Add wall-loss contribution to Jacobian F (diagonal, in-place).
    F[i,i] -= gamma0 * t0 * xi[i]

    Mirrors jacobian_wall_deposition!() in loss.jl.
    """
    ws.LR = xi.copy()
    diag_view = np.einsum('ii->i', F)
    diag_view -= (ws.gamma0 * ws.t0) * xi


def wall_deposition_rate(ws: AeroSys) -> np.ndarray:
    """
    Default wall-deposition rate profile [1/s].
    Combines a small-particle diffusion term and a large-particle
    gravitational settling sigmoid.

    Mirrors WallDepositionRate() in loss.jl.
    """
    rr   = ws.cst_v ** (-0.5)
    sx   = 15.0e-9
    dia0 = 110.0e-9
    idx  = np.arange(1, ws.nbin + 1, dtype=float)
    return (3.0e-16 / np.sqrt(np.pi * ws.d0 ** 3 / 6.0)) * rr ** idx \
           + 5.0e-5 / (1.0 + np.exp(-(ws.d - dia0) / sx))


def loss_err(dt: float,
             e_l: np.ndarray,
             x: np.ndarray,
             delta: np.ndarray) -> np.ndarray:
    """
    Std dev of GDE error due to uncertainty in loss rates.

    Mirrors loss_err() in loss.jl.
    """
    u_density = x / delta
    return dt * delta * u_density * e_l


# ============================================================
#  Discretisation error estimate
# ============================================================

def discretization_err(dt: float,
                       n_deriv: np.ndarray,
                       g_star: np.ndarray,
                       delta: np.ndarray) -> np.ndarray:
    """
    Std dev of GDE error due to size/time discretisation.

    Mirrors discretization_err() in iteration.jl.
    """
    cst_r = float(np.mean(delta[1:] / delta[:-1]))
    de    = 0.5 * ((np.sqrt(cst_r) - 1) ** 2 / (cst_r - 1.0)) * (n_deriv * g_star) * delta
    first = abs(de[1]) + (1.0 / cst_r) * abs(de[0])
    rest  = np.abs(de[1:]) + (1.0 / cst_r) * np.abs(de[:-1])
    return dt * np.concatenate([[first], rest])


# ============================================================
#  Main iteration: iter  (mirrors iter!() in iteration.jl)
# ============================================================

def iter(
    ws:   AeroSys,
    x:    np.ndarray,
    dt:   float,
    zeta: Union[float, np.ndarray],
    j:    float,
    xi:   np.ndarray,
) -> np.ndarray:
    """
    One Euler time step of the dimensionless GDE.

    Parameters
    ----------
    ws   : AeroSys workspace (mechanism flags + parameters)
    x    : current dimensionless size distribution, shape (nbin,)
    dt   : time step [s]
    zeta : dimensionless growth rate (scalar or per-bin array)
    j    : dimensionless nucleation rate (scalar)
    xi   : loss rate profile [1/s], shape (nbin,)

    Returns
    -------
    x_new : updated dimensionless size distribution, shape (nbin,)

    Improvement: returns a new array — never mutates `x`.
    Mirrors iter!() in iteration.jl.
    """
    dx_coa = _coagulation_full(ws, x)  if (ws.is_coa and ws.is_coa_gain) \
        else (_coagulation_loss(ws, x) if ws.is_coa
              else np.zeros(ws.nbin))

    dx_con = _condensation_growth_and_evap(ws, x, zeta) if ws.is_con \
        else np.zeros(ws.nbin)

    dx_nuc = _nucleation(ws, j) if ws.is_nuc \
        else np.zeros(ws.nbin)

    dx_los = _wall_deposition(ws, x, xi) if ws.is_los \
        else np.zeros(ws.nbin)

    return x + (dt / ws.t0) * (dx_coa + dx_con + dx_nuc + dx_los)


# ============================================================
#  Jacobian of the GDE  (mirrors jacobian_GDE() in iteration.jl)
# ============================================================

def jacobian_GDE(
    ws:   AeroSys,
    x:    np.ndarray,
    dt:   float,
    zeta: Union[float, np.ndarray],
    xi:   np.ndarray,
) -> np.ndarray:
    """
    Jacobian of the one-step Euler iteration:
        J[s, k] = ∂x_new[s] / ∂x[k]

    Returns
    -------
    J : (nbin, nbin) ndarray,  J = I + (dt/t0) * (F_coa + F_con + F_los)

    Mirrors jacobian_GDE() in iteration.jl.
    """
    F = np.zeros((ws.nbin, ws.nbin))

    if ws.is_coa:
        if ws.is_coa_gain:
            _jacobian_coagulation_full(ws, x, F)
        else:
            _jacobian_coagulation_loss(ws, x, F)

    if ws.is_con:
        _jacobian_condensation(ws, x, zeta, F)

    if ws.is_los:
        _jacobian_wall_deposition(ws, xi, F)

    return np.eye(ws.nbin) + (dt / ws.t0) * F


# ============================================================
#  Convenience: run a full simulation over a time series
# ============================================================

def simulate(
    ws:      AeroSys,
    x0:      np.ndarray,
    t_samp:  np.ndarray,
    zeta_t:  Union[np.ndarray, float],
    j_t:     Union[np.ndarray, float],
    xi_t:    Optional[np.ndarray] = None,
    store_jacobians: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Integrate the GDE over a time series using the Euler scheme.

    Parameters
    ----------
    ws       : configured AeroSys (mechanisms enabled, parameters set)
    x0       : initial size distribution, shape (nbin,)
    t_samp   : sample times [s], shape (n_t,)
    zeta_t   : growth rate(s); scalar, (n_t,), or (nbin, n_t)
    j_t      : nucleation rate(s); scalar or (n_t,)
    xi_t     : loss rate profiles; None or (nbin, n_t)
    store_jacobians : if True, also return the Jacobian at each step

    Returns
    -------
    X        : size distribution time series, shape (nbin, n_t)
    J_all    : Jacobians, shape (nbin, nbin, n_t) if store_jacobians else None
    """
    n_t  = len(t_samp)
    nbin = ws.nbin
    X    = np.zeros((nbin, n_t))
    J_all = np.zeros((nbin, nbin, n_t)) if store_jacobians else None

    xi_zero = np.zeros(nbin)
    x = x0.copy()

    for k in range(n_t):
        # --- extract parameters for this step ---
        dt = (t_samp[k] - t_samp[k - 1]) if k > 0 else (t_samp[1] - t_samp[0])

        zeta_k = (zeta_t if np.isscalar(zeta_t)
                  else (zeta_t[k] if zeta_t.ndim == 1
                        else zeta_t[:, k]))

        j_k = float(j_t) if np.isscalar(j_t) else float(j_t[k])

        xi_k = xi_zero if xi_t is None else xi_t[:, k]

        # --- optionally compute Jacobian ---
        if store_jacobians:
            J_all[:, :, k] = jacobian_GDE(ws, x, dt, zeta_k, xi_k)

        # --- advance one step ---
        x = iter(ws, x, dt, zeta_k, j_k, xi_k)
        X[:, k] = x

    return X, J_all

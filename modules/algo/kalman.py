"""
modules/algo/kalman.py
======================
Pure-Python / NumPy translation of the BAYROSOL EKF package
(type.jl + ekf_mod.jl + EKF.jl), with improvements.

Original Julia code © 2018-2019 Matthew Ozon, MIT License.
Python translation and improvements: see notes below.

Design
------
The original Julia design uses *stub functions* that the user overloads
(apply_model!, update_jacobian!, …).  In Python we express the same idea
with *callables passed as arguments*, which is more Pythonic and avoids
global state.  The workspace (KFWorkspace) mirrors wsKF exactly.

Improvements over the original
-------------------------------
1.  Joseph-form covariance update  (I - KH) P (I - KH)' + K R K'
    instead of the simpler (I - KH) P.  The Joseph form is numerically
    more stable and guarantees positive semi-definiteness even with
    floating-point errors.

2.  Optional symmetric enforcement  0.5*(P + P')  after each update,
    controlled by `enforce_symmetry` flag (default True).

3.  Numerically robust Kalman gain: uses `np.linalg.solve` (LU
    factorisation) instead of `pinv` when the innovation covariance is
    square and well-conditioned; falls back to `pinv` otherwise.

4.  `log_likelihood` accumulation: the filter accumulates the Gaussian
    log-likelihood of the innovations at each step, useful for parameter
    tuning.

5.  Time-varying noise covariances (Q, R) and measurement operators (H)
    are handled via optional callable arguments, consistent with the
    original time_dep_* flags.

Public API
----------
KFWorkspace          — workspace / result container (mirrors wsKF)
kalman_filter        — forward KF pass (mirrors KF)
kalman_filter_smoother — KF + Fixed Interval Smoother (mirrors KF_FIS)
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import pinv, solve
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple


# ============================================================
#  Workspace  (mirrors wsKF)
# ============================================================

@dataclass
class KFWorkspace:
    """
    Storage for a complete Kalman filter / smoother run.

    Shapes
    ------
    n_x  : state dimension      (model_dim in Julia)
    n_y  : observation dimension (meas_dim  in Julia)
    n_t  : number of time steps  (n_samp    in Julia)

    All *_all arrays have the time axis **last**, matching the Julia
    convention (Julia: [state, state, time]; Python: same with numpy).
    """

    # --- dimensions ---
    n_x: int
    n_y: int
    n_t: int

    # --- current-step working arrays ---
    x_fil: np.ndarray = field(init=False)   # (n_x,)
    x_pre: np.ndarray = field(init=False)   # (n_x,)
    P_fil: np.ndarray = field(init=False)   # (n_x, n_x)   o_fil in Julia
    P_pre: np.ndarray = field(init=False)   # (n_x, n_x)   o_pre in Julia
    y_pre: np.ndarray = field(init=False)   # (n_y,)
    y_t:   np.ndarray = field(init=False)   # (n_y,)  current observation
    F:     np.ndarray = field(init=False)   # (n_x, n_x)  state transition Jacobian
    H:     np.ndarray = field(init=False)   # (n_y, n_x)  measurement Jacobian
    K:     np.ndarray = field(init=False)   # (n_x, n_y)  Kalman gain
    Q:     np.ndarray = field(init=False)   # (n_x, n_x)  process noise cov
    R:     np.ndarray = field(init=False)   # (n_y, n_y)  observation noise cov

    # --- time-series history ---
    x_fil_all: np.ndarray = field(init=False)   # (n_x, n_t)
    x_pre_all: np.ndarray = field(init=False)   # (n_x, n_t)
    P_fil_all: np.ndarray = field(init=False)   # (n_x, n_x, n_t)
    P_pre_all: np.ndarray = field(init=False)   # (n_x, n_x, n_t)
    F_all:     np.ndarray = field(init=False)   # (n_x, n_x, n_t)
    H_all:     np.ndarray = field(init=False)   # (n_y, n_x, n_t)
    Q_all:     np.ndarray = field(init=False)   # (n_x, n_x, n_t)
    R_all:     np.ndarray = field(init=False)   # (n_y, n_y, n_t)

    # --- smoother output (filled by kalman_filter_smoother) ---
    x_smo_all: Optional[np.ndarray] = field(init=False, default=None)  # (n_x, n_t)
    P_smo_all: Optional[np.ndarray] = field(init=False, default=None)  # (n_x, n_x, n_t)

    # --- scalar diagnostics ---
    log_likelihood: float = field(init=False, default=0.0)

    def __post_init__(self):
        nx, ny, nt = self.n_x, self.n_y, self.n_t
        self.x_fil = np.zeros(nx)
        self.x_pre = np.zeros(nx)
        self.P_fil = np.zeros((nx, nx))
        self.P_pre = np.zeros((nx, nx))
        self.y_pre = np.zeros(ny)
        self.y_t   = np.zeros(ny)
        self.F     = np.zeros((nx, nx))
        self.H     = np.zeros((ny, nx))
        self.K     = np.zeros((nx, ny))
        self.Q     = np.zeros((nx, nx))
        self.R     = np.zeros((ny, ny))

        self.x_fil_all = np.zeros((nx, nt))
        self.x_pre_all = np.zeros((nx, nt))
        self.P_fil_all = np.zeros((nx, nx, nt))
        self.P_pre_all = np.zeros((nx, nx, nt))
        self.F_all     = np.zeros((nx, nx, nt))
        self.H_all     = np.zeros((ny, nx, nt))
        self.Q_all     = np.zeros((nx, nx, nt))
        self.R_all     = np.zeros((ny, ny, nt))
        self.log_likelihood = 0.0


# ============================================================
#  Internal single-step routines
# ============================================================

def _prediction(ws: KFWorkspace,
                apply_model: Callable,
                dt: float) -> None:
    """
    Prediction step.
        x_pre = f(x_fil, dt)         [nonlinear propagation]
        P_pre = F P_fil F' + Q       [covariance propagation]
        y_pre = H x_pre              [predicted observation, linear]

    Mirrors prediction!() in ekf_mod.jl.
    """
    ws.x_pre = apply_model(ws.x_fil, dt)
    ws.P_pre = ws.F @ ws.P_fil @ ws.F.T + ws.Q
    ws.y_pre = ws.H @ ws.x_pre


def _kalman_gain(ws: KFWorkspace) -> None:
    """
    Compute Kalman gain.
        S   = H P_pre H' + R          [innovation covariance]
        K   = P_pre H' S^{-1}

    Uses solve() for efficiency and numerical stability; falls back to
    pinv for ill-conditioned S.  Mirrors kalman_gain() in ekf_mod.jl.
    """
    S = ws.H @ ws.P_pre @ ws.H.T + ws.R
    try:
        # K = P H' S^{-1}  ⟺  S K' = H P'  →  solve for K'
        ws.K = np.linalg.solve(S.T, (ws.P_pre @ ws.H.T).T).T
    except np.linalg.LinAlgError:
        ws.K = ws.P_pre @ ws.H.T @ pinv(S)


def _filtering(ws: KFWorkspace,
               enforce_symmetry: bool = True) -> None:
    """
    Filtering (analysis) step.
        x_fil = x_pre + K (y_t - y_pre)
        P_fil = (I - KH) P_pre (I - KH)' + K R K'    [Joseph form]

    The Joseph form (improvement over the original Julia code) keeps
    P_fil symmetric positive semi-definite in finite-precision arithmetic.
    Mirrors filtering!() in ekf_mod.jl.
    """
    innovation = ws.y_t - ws.y_pre
    ws.x_fil   = ws.x_pre + ws.K @ innovation

    I_KH   = np.eye(ws.n_x) - ws.K @ ws.H
    ws.P_fil = I_KH @ ws.P_pre @ I_KH.T + ws.K @ ws.R @ ws.K.T

    if enforce_symmetry:
        ws.P_fil = 0.5 * (ws.P_fil + ws.P_fil.T)


def _log_likelihood_contribution(ws: KFWorkspace) -> float:
    """
    Gaussian log-likelihood of the innovation:
        ℓ = -½ [ k log(2π) + log|S| + v' S^{-1} v ]
    where v = y_t - y_pre,  S = H P_pre H' + R.
    """
    v = ws.y_t - ws.y_pre
    S = ws.H @ ws.P_pre @ ws.H.T + ws.R
    ny = ws.n_y
    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0:
        return 0.0   # degenerate — skip this step
    try:
        Sinv_v = np.linalg.solve(S, v)
    except np.linalg.LinAlgError:
        Sinv_v = pinv(S) @ v
    return -0.5 * (ny * np.log(2 * np.pi) + logdet + v @ Sinv_v)


def _one_step_smoother(ws: KFWorkspace,
                        i: int,
                        x_smo_next: np.ndarray,
                        P_smo_next: np.ndarray,
                        enforce_symmetry: bool = True
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    One backward step of the Rauch-Tung-Striebel smoother.
        G_i     = P_fil_i  F_{i+1}' P_pre_{i+1}^{-1}
        x_smo_i = x_fil_i + G_i (x_smo_{i+1} - x_pre_{i+1})
        P_smo_i = P_fil_i + G_i (P_smo_{i+1} - P_pre_{i+1}) G_i'

    Mirrors one_step_fixed_interval() in ekf_mod.jl.
    """
    P_fil_i  = ws.P_fil_all[:, :, i]
    F_next   = ws.F_all[:, :, i + 1]
    P_pre_next = ws.P_pre_all[:, :, i + 1]

    # Smoothing gain  G_i = P_fil_i F_{i+1}' P_pre_{i+1}^{-1}
    # solve P_pre_{i+1} G_i' = F_{i+1} P_fil_i'  →  G_i' then transpose
    try:
        G = np.linalg.solve(P_pre_next.T, (P_fil_i @ F_next.T).T).T
    except np.linalg.LinAlgError:
        G = P_fil_i @ F_next.T @ pinv(P_pre_next)

    x_smo_i = ws.x_fil_all[:, i] + G @ (x_smo_next - ws.x_pre_all[:, i + 1])
    P_smo_i = P_fil_i + G @ (P_smo_next - P_pre_next) @ G.T

    if enforce_symmetry:
        P_smo_i = 0.5 * (P_smo_i + P_smo_i.T)

    return x_smo_i, P_smo_i


# ============================================================
#  Public API — Forward Kalman Filter
# ============================================================

def kalman_filter(
    # --- initial conditions ---
    x0:          np.ndarray,
    P0_diag:     np.ndarray,       # 1-D array of diagonal variances → diagm
    # --- noise covariances (constant) ---
    Q:           np.ndarray,       # (n_x, n_x)  process noise
    R:           np.ndarray,       # (n_y, n_y)  observation noise
    # --- observations ---
    Y:           np.ndarray,       # (n_y, n_t) or (n_t, n_y) — see `obs_axis`
    t_samp:      np.ndarray,       # (n_t,)  sample times
    t0:          float,            # normalisation time [s] (as in Julia)
    # --- problem-specific callables ---
    apply_model:          Callable,   # (x, dt) → x_new
    update_jacobian:      Callable,   # (x, dt, F) → F_new   (EKF Jacobian)
    apply_measurement:    Callable,   # (x, H) → y_pred      (= H @ x for linear)
    set_jacobian:         Callable,   # (x, dt, F) → F_init  (initial linearisation)
    set_meas_jacobian:    Callable,   # (H,) → H_init
    # --- optional time-varying overrides ---
    update_Q:             Optional[Callable] = None,  # (ws) → Q_new
    update_R:             Optional[Callable] = None,  # (y_t, ws) → R_new
    update_H:             Optional[Callable] = None,  # (ws) → H_new
    # --- options ---
    obs_axis:             int  = 1,    # axis along which time runs in Y
    enforce_symmetry:     bool = True,
    compute_likelihood:   bool = True,
) -> KFWorkspace:
    """
    Forward Extended Kalman Filter.

    Faithfully mirrors KF() in ekf_mod.jl with the improvements listed
    at the top of this module.

    Parameters
    ----------
    x0              Initial state estimate, shape (n_x,).
    P0_diag         Diagonal of the initial covariance, shape (n_x,).
    Q               Process noise covariance, shape (n_x, n_x).
    R               Observation noise covariance, shape (n_y, n_y).
    Y               Observations.  If obs_axis=1 (default): shape (n_y, n_t).
                    If obs_axis=0: shape (n_t, n_y) — transposed internally.
    t_samp          Sample times, shape (n_t,).
    t0              Time normalisation constant (same role as t0_ in Julia).
    apply_model     Callable (x, dt) → x_new.  The nonlinear state evolution.
    update_jacobian Callable (x, dt, F) → F_new.  Jacobian of apply_model at x.
    apply_measurement Callable (x, H) → y_pred.  Usually just H @ x.
    set_jacobian    Callable (x, dt, F) → F.  Called once at initialisation.
    set_meas_jacobian Callable (H,) → H.  Called once at initialisation.
    update_Q        Optional callable (ws) → Q_new.  For time-varying Q.
    update_R        Optional callable (y_t, ws) → R_new.  For time-varying R.
    update_H        Optional callable (ws) → H_new.  For time-varying H.
    obs_axis        Axis of Y that indexes time (0 or 1).
    enforce_symmetry If True, symmetrise covariance matrices after each step.
    compute_likelihood If True, accumulate Gaussian log-likelihood.

    Returns
    -------
    KFWorkspace  populated with full filter history.
    """
    # -- normalise observation array to shape (n_y, n_t) --
    if obs_axis == 0:
        Y = Y.T
    n_y, n_t = Y.shape
    n_x = len(x0)

    ws = KFWorkspace(n_x=n_x, n_y=n_y, n_t=n_t)
    ws.t_samp = t_samp.copy() if hasattr(t_samp, 'copy') else np.array(t_samp)

    # -- initialise state and covariance --
    ws.x_fil = x0.copy()
    ws.P_fil = np.diag(P0_diag)
    ws.Q     = Q.copy()
    ws.R     = R.copy()

    # -- initialise Jacobians (linearise around x0) --
    dt_init = (t_samp[1] - t_samp[0]) / t0
    ws.F = set_jacobian(ws.x_fil, dt_init, ws.F.copy())
    ws.H = set_meas_jacobian(ws.H.copy())

    # -- forward filter loop --
    for k in range(n_t):
        # time step (normalised), same boundary logic as Julia
        if k == n_t - 1:
            dt = (t_samp[k] - t_samp[k - 1]) / t0
        else:
            dt = (t_samp[k + 1] - t_samp[k]) / t0

        # current observation
        ws.y_t = Y[:, k]

        # optional time-varying updates
        ws.F = update_jacobian(ws.x_fil, dt, ws.F.copy())
        if update_Q is not None:
            ws.Q = update_Q(ws)
        if update_R is not None:
            ws.R = update_R(ws.y_t, ws)
        if update_H is not None:
            ws.H = update_H(ws)

        # prediction
        _prediction(ws, apply_model, dt)

        # likelihood (before filtering, uses y_pre from prediction)
        if compute_likelihood:
            ws.log_likelihood += _log_likelihood_contribution(ws)

        # Kalman gain
        _kalman_gain(ws)

        # filtering (analysis)
        _filtering(ws, enforce_symmetry=enforce_symmetry)

        # store history
        ws.x_fil_all[:, k]    = ws.x_fil
        ws.x_pre_all[:, k]    = ws.x_pre
        ws.P_fil_all[:, :, k] = ws.P_fil
        ws.P_pre_all[:, :, k] = ws.P_pre
        ws.F_all[:, :, k]     = ws.F
        ws.H_all[:, :, k]     = ws.H
        ws.Q_all[:, :, k]     = ws.Q
        ws.R_all[:, :, k]     = ws.R

    return ws


# ============================================================
#  Public API — KF + Fixed Interval Smoother
# ============================================================

def kalman_filter_smoother(
    x0:          np.ndarray,
    P0_diag:     np.ndarray,
    Q:           np.ndarray,
    R:           np.ndarray,
    Y:           np.ndarray,
    t_samp:      np.ndarray,
    t0:          float,
    apply_model:          Callable,
    update_jacobian:      Callable,
    apply_measurement:    Callable,
    set_jacobian:         Callable,
    set_meas_jacobian:    Callable,
    update_Q:             Optional[Callable] = None,
    update_R:             Optional[Callable] = None,
    update_H:             Optional[Callable] = None,
    obs_axis:             int  = 1,
    enforce_symmetry:     bool = True,
    compute_likelihood:   bool = True,
) -> KFWorkspace:
    """
    Extended Kalman Filter + Fixed Interval Kalman Smoother (FIKS).

    Mirrors KF_FIS() in ekf_mod.jl.

    Runs the forward filter first, then a single backward pass using the
    Rauch-Tung-Striebel (RTS) smoother equations.  The smoothed
    trajectories are stored in ws.x_smo_all and ws.P_smo_all.

    Parameters
    ----------
    (same as kalman_filter)

    Returns
    -------
    KFWorkspace  with both filter and smoother histories populated.
    """
    # -- forward pass --
    ws = kalman_filter(
        x0=x0, P0_diag=P0_diag, Q=Q, R=R,
        Y=Y, t_samp=t_samp, t0=t0,
        apply_model=apply_model,
        update_jacobian=update_jacobian,
        apply_measurement=apply_measurement,
        set_jacobian=set_jacobian,
        set_meas_jacobian=set_meas_jacobian,
        update_Q=update_Q, update_R=update_R, update_H=update_H,
        obs_axis=obs_axis,
        enforce_symmetry=enforce_symmetry,
        compute_likelihood=compute_likelihood,
    )

    n_t = ws.n_t

    # -- backward smoother pass --
    ws.x_smo_all = np.empty_like(ws.x_fil_all)
    ws.P_smo_all = np.empty_like(ws.P_fil_all)

    # initialise at the last time step (smoother = filter at t=T)
    ws.x_smo_all[:, n_t - 1]    = ws.x_fil_all[:, n_t - 1]
    ws.P_smo_all[:, :, n_t - 1] = ws.P_fil_all[:, :, n_t - 1]

    # RTS backward sweep
    for k in range(n_t - 2, -1, -1):
        ws.x_smo_all[:, k], ws.P_smo_all[:, :, k] = _one_step_smoother(
            ws, k,
            ws.x_smo_all[:, k + 1],
            ws.P_smo_all[:, :, k + 1],
            enforce_symmetry=enforce_symmetry,
        )

    return ws


# ============================================================
#  Convenience: linear-system versions (no Jacobian computation)
# ============================================================

def make_linear_callbacks(
    F: np.ndarray,
    H: np.ndarray,
) -> dict:
    """
    Build the full set of callback functions for a **linear** state-space
    model with constant F and H matrices.

    Usage
    -----
    cbs = make_linear_callbacks(F, H)
    ws  = kalman_filter(x0, P0d, Q, R, Y, t, t0, **cbs)

    Returns
    -------
    dict with keys matching the callback parameter names of kalman_filter.
    """
    F_ = F.copy()
    H_ = H.copy()

    def apply_model(x, dt):
        return F_ @ x

    def update_jacobian(x, dt, Fold):
        return F_

    def apply_measurement(x, Hmat):
        return Hmat @ x

    def set_jacobian(x, dt, Fold):
        return F_

    def set_meas_jacobian(Hold):
        return H_

    return dict(
        apply_model=apply_model,
        update_jacobian=update_jacobian,
        apply_measurement=apply_measurement,
        set_jacobian=set_jacobian,
        set_meas_jacobian=set_meas_jacobian,
    )


# ============================================================
#  Result accessor helpers  (convenience for the GUI/plotting)
# ============================================================

def filtered_states(ws: KFWorkspace) -> np.ndarray:
    """Return filtered state trajectory, shape (n_t, n_x)."""
    return ws.x_fil_all.T

def filtered_stds(ws: KFWorkspace) -> np.ndarray:
    """Return std dev of filtered states, shape (n_t, n_x)."""
    diags = np.einsum('iit->ti', ws.P_fil_all)   # diagonal of P_fil_all over time
    return np.sqrt(np.maximum(diags, 0.0))

def smoothed_states(ws: KFWorkspace) -> np.ndarray:
    """Return smoothed state trajectory, shape (n_t, n_x). Requires FIKS."""
    if ws.x_smo_all is None:
        raise RuntimeError("Run kalman_filter_smoother(), not kalman_filter().")
    return ws.x_smo_all.T

def smoothed_stds(ws: KFWorkspace) -> np.ndarray:
    """Return std dev of smoothed states, shape (n_t, n_x). Requires FIKS."""
    if ws.P_smo_all is None:
        raise RuntimeError("Run kalman_filter_smoother(), not kalman_filter().")
    diags = np.einsum('iit->ti', ws.P_smo_all)
    return np.sqrt(np.maximum(diags, 0.0))

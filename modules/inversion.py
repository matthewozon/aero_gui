"""
modules/inversion.py
--------------------
Data inversion methods: recover the true size distribution N(Dp) from
raw instrument counts.

These are Python ports / stubs of your Julia implementations.
Each method takes the same inputs and returns the same outputs,
so they can be swapped in the GUI without changing anything else.
"""

import numpy as np
from scipy.linalg import svd
from typing import Optional, Tuple
from dataclasses import dataclass, field


# ============================================================
#  Result container
# ============================================================

@dataclass
class InversionResult:
    N_retrieved: np.ndarray         # dN/dlogDp [#/cm³], shape (n_dp,)
    dp_grid: np.ndarray             # diameter grid [nm]
    residual: float = 0.0           # ||A*x - y|| / ||y||
    info: dict = field(default_factory=dict)


# ============================================================
#  Base class
# ============================================================

class InversionMethod:
    """Abstract base — subclass and override solve()."""

    name: str = "Base"

    def solve(self, A: np.ndarray, counts: np.ndarray,
              dp_grid: np.ndarray) -> InversionResult:
        raise NotImplementedError


# ============================================================
#  Method 1 – Tikhonov regularisation (L-curve / fixed lambda)
# ============================================================

class TikhonovInversion(InversionMethod):
    """
    Minimise  ||A x - y||² + λ ||L x||²
    where L is the identity (zeroth-order) or a finite-difference matrix.
    """

    name = "Tikhonov Regularisation"

    def __init__(self, lambda_reg: float = 1e-3,
                 order: int = 0):
        """
        Parameters
        ----------
        lambda_reg : regularisation parameter λ
        order      : 0 = identity, 1 = first differences, 2 = second differences
        """
        self.lambda_reg = lambda_reg
        self.order = order

    def _reg_matrix(self, n: int) -> np.ndarray:
        if self.order == 0:
            return np.eye(n)
        elif self.order == 1:
            L = np.zeros((n - 1, n))
            for i in range(n - 1):
                L[i, i] = -1
                L[i, i + 1] = 1
            return L
        else:  # order == 2
            L = np.zeros((n - 2, n))
            for i in range(n - 2):
                L[i, i] = 1
                L[i, i + 1] = -2
                L[i, i + 2] = 1
            return L

    def solve(self, A: np.ndarray, counts: np.ndarray,
              dp_grid: np.ndarray) -> InversionResult:
        n = A.shape[1]
        L = self._reg_matrix(n)
        lam = self.lambda_reg

        # Normal equations: (A^T A + λ L^T L) x = A^T y
        AtA = A.T @ A
        LtL = L.T @ L
        M = AtA + lam * LtL
        rhs = A.T @ counts

        x = np.linalg.lstsq(M, rhs, rcond=None)[0]
        x = np.maximum(x, 0.0)   # non-negativity

        residual = np.linalg.norm(A @ x - counts) / (np.linalg.norm(counts) + 1e-30)
        return InversionResult(N_retrieved=x, dp_grid=dp_grid,
                               residual=residual,
                               info={"lambda": lam, "order": self.order})

    def l_curve(self, A: np.ndarray, counts: np.ndarray,
                dp_grid: np.ndarray,
                lambdas: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the L-curve: residual norm vs solution norm for a range of λ.
        Returns (lambdas, residual_norms, solution_norms).
        """
        if lambdas is None:
            lambdas = np.logspace(-6, 2, 40)
        res_norms = []
        sol_norms = []
        for lam in lambdas:
            self.lambda_reg = lam
            r = self.solve(A, counts, dp_grid)
            res_norms.append(r.residual)
            sol_norms.append(np.linalg.norm(r.N_retrieved))
        return lambdas, np.array(res_norms), np.array(sol_norms)


# ============================================================
#  Method 2 – Truncated SVD
# ============================================================

class TruncatedSVDInversion(InversionMethod):
    """
    Regularise by truncating small singular values.
    k = number of singular values to keep (None → keep all above threshold).
    """

    name = "Truncated SVD"

    def __init__(self, k: int = None, threshold: float = 1e-3):
        self.k = k
        self.threshold = threshold

    def solve(self, A: np.ndarray, counts: np.ndarray,
              dp_grid: np.ndarray) -> InversionResult:
        U, s, Vt = svd(A, full_matrices=False)

        if self.k is not None:
            k = min(self.k, len(s))
        else:
            k = int(np.sum(s / s[0] > self.threshold))

        s_inv = np.zeros_like(s)
        s_inv[:k] = 1.0 / s[:k]

        x = Vt.T @ np.diag(s_inv) @ U.T @ counts
        x = np.maximum(x, 0.0)

        residual = np.linalg.norm(A @ x - counts) / (np.linalg.norm(counts) + 1e-30)
        return InversionResult(N_retrieved=x, dp_grid=dp_grid,
                               residual=residual,
                               info={"k_used": k, "total_sv": len(s),
                                     "condition": s[0] / s[-1]})


# ============================================================
#  Method 3 – Non-negative Least Squares (NNLS)
# ============================================================

class NNLSInversion(InversionMethod):
    """
    Scipy's bounded NNLS — no regularisation, enforces x >= 0.
    Useful as a baseline.
    """

    name = "NNLS (non-negative least squares)"

    def solve(self, A: np.ndarray, counts: np.ndarray,
              dp_grid: np.ndarray) -> InversionResult:
        from scipy.optimize import nnls
        x, res = nnls(A, counts)
        residual = np.linalg.norm(A @ x - counts) / (np.linalg.norm(counts) + 1e-30)
        return InversionResult(N_retrieved=x, dp_grid=dp_grid,
                               residual=residual,
                               info={"nnls_residual": float(res)})


# ============================================================
#  Method 4 – Expectation-Maximisation (stub – port from Julia)
# ============================================================

class EMInversion(InversionMethod):
    """
    Poisson-statistics EM algorithm, suitable for count data.
    Port this from your Julia implementation.
    """

    name = "EM (Expectation-Maximisation)"

    def __init__(self, n_iter: int = 100, tol: float = 1e-6):
        self.n_iter = n_iter
        self.tol = tol

    def solve(self, A: np.ndarray, counts: np.ndarray,
              dp_grid: np.ndarray) -> InversionResult:
        n = A.shape[1]
        x = np.ones(n)   # uniform initialisation

        row_sums = A.sum(axis=1) + 1e-30
        col_sums = A.sum(axis=0) + 1e-30

        for iteration in range(self.n_iter):
            x_old = x.copy()
            expected = A @ x + 1e-30
            ratio = counts / expected                     # shape (n_ch,)
            update = A.T @ ratio / col_sums               # shape (n_dp,)
            x = x * update
            x = np.maximum(x, 0.0)

            change = np.linalg.norm(x - x_old) / (np.linalg.norm(x_old) + 1e-30)
            if change < self.tol:
                break

        residual = np.linalg.norm(A @ x - counts) / (np.linalg.norm(counts) + 1e-30)
        return InversionResult(N_retrieved=x, dp_grid=dp_grid,
                               residual=residual,
                               info={"iterations": iteration + 1})


# ============================================================
#  AeroInv helpers
# ============================================================

def _build_J(n_model: int, scale: float = 1e-3) -> np.ndarray:
    """
    Square second-order finite-difference regularisation operator,
    matching the construction used inside aeroinv_reg.
    Shape (n_model, n_model); boundary rows zeroed.
    """
    J = scale * (
        np.diag(-2.0 * np.ones(n_model))
        + np.diag(np.ones(n_model - 1), k=1)
        + np.diag(np.ones(n_model - 1), k=-1)
    )
    J[0, :2] = 0.0
    J[-1, -2:] = 0.0
    return J


def _build_W(K: np.ndarray) -> np.ndarray:
    """Data-driven diagonal preconditioner (column sums of K, normalised)."""
    W = K.sum(axis=1)
    W = np.where(W > 0, 1.0 / W, 0.0)
    total = W.sum()
    if total > 0:
        W = K.shape[0] * W / total
    return W


# ============================================================
#  Method 5 – AeroInv quick estimate
# ============================================================

class AeroInvQuick(InversionMethod):
    """
    Non-iterative rough estimate.  For each measurement channel, the
    effective diameter range where sensitivity ≥50% of peak is found
    and the counts are divided by that width.  Fast sanity-check only;
    may return small negative values (clipped to zero).
    """

    name = "Quick estimate (AeroInv)"

    def solve(self, A: np.ndarray, counts: np.ndarray,
              dp_grid: np.ndarray) -> InversionResult:
        from modules.aeroinv import aeroinv_quick
        # aeroinv_quick returns a (n_channels,) estimate at each channel's
        # effective diameter; interpolate back to dp_grid.
        N_ch = aeroinv_quick(A, dp_grid, counts)   # (n_channels,)
        N_ch = np.maximum(np.asarray(N_ch, dtype=float), 0.0)
        # Centre diameter for each channel ≈ argmax of each kernel row
        ch_dp = dp_grid[np.argmax(A, axis=1)]
        order = np.argsort(ch_dp)
        N = np.interp(dp_grid, ch_dp[order], N_ch[order])
        residual = np.linalg.norm(A @ N - counts) / (np.linalg.norm(counts) + 1e-30)
        return InversionResult(N_retrieved=N, dp_grid=dp_grid,
                               residual=residual, info={})


# ============================================================
#  Method 6 – AeroInv preconditioned regularised (closed-form)
# ============================================================

class AeroInvPrecondReg(InversionMethod):
    """
    Closed-form preconditioned regularised inversion.  Solves the normal
    equation ``(EK'EK + J'J) n = EK' En`` where E is a channel-wise
    preconditioner and J is a second-order smoothness prior.  No
    positivity constraint; clipped to zero after solve.
    """

    name = "Precond. regularised (AeroInv)"

    def __init__(self, u0: float = 10000.0, rel_un: float = 0.1):
        """
        Parameters
        ----------
        u0     : characteristic density scale
        rel_un : relative amplitude of model uncertainty in (0, 1)
        """
        self.u0 = u0
        self.rel_un = rel_un

    def solve(self, A: np.ndarray, counts: np.ndarray,
              dp_grid: np.ndarray) -> InversionResult:
        from modules.aeroinv import precond_reg_inv
        N = precond_reg_inv(A, counts, u0=self.u0, rel_un=self.rel_un)
        N = np.maximum(np.asarray(N, dtype=float), 0.0)
        residual = np.linalg.norm(A @ N - counts) / (np.linalg.norm(counts) + 1e-30)
        return InversionResult(N_retrieved=N, dp_grid=dp_grid,
                               residual=residual,
                               info={"u0": self.u0, "rel_un": self.rel_un})


# ============================================================
#  Method 7 – AeroInv Chambolle-Pock (Gaussian noise, positivity)
# ============================================================

class AeroInvRegCP(InversionMethod):
    """
    Chambolle-Pock Algorithm 2 with Gaussian-noise fidelity and positivity
    constraint.  Warm-starts consecutive scans.  Use ``solve_series`` when
    inverting multiple scans for efficiency.
    """

    name = "Chambolle-Pock / Gaussian (AeroInv)"

    def __init__(self, tau00: float = 1e15, Niter: int = 1000,
                 reg_scale: float = 1.0):
        """
        Parameters
        ----------
        tau00     : initial primal step size
        Niter     : Chambolle-Pock iterations per scan
        reg_scale : weight on the second-order regularisation operator J
        """
        self.tau00     = tau00
        self.Niter     = Niter
        self.reg_scale = reg_scale

    def solve(self, A: np.ndarray, counts: np.ndarray,
              dp_grid: np.ndarray) -> InversionResult:
        from modules.aeroinv import aeroinv_reg
        y2d = counts[:, np.newaxis]          # (n_data, 1)
        F_ALL, _ = aeroinv_reg(y2d, A, tau00=self.tau00, Niter=self.Niter,
                               reg_scale=self.reg_scale)
        N = F_ALL[0]
        residual = np.linalg.norm(A @ N - counts) / (np.linalg.norm(counts) + 1e-30)
        return InversionResult(N_retrieved=N, dp_grid=dp_grid,
                               residual=residual,
                               info={"tau00": self.tau00, "Niter": self.Niter,
                                     "reg_scale": self.reg_scale})

    def solve_series(self, A: np.ndarray, count_matrix: np.ndarray,
                     dp_grid: np.ndarray) -> np.ndarray:
        """
        Invert all scans at once, warm-starting each from the previous.

        Parameters
        ----------
        count_matrix : shape (n_scans, n_channels)

        Returns
        -------
        ndarray, shape (n_scans, n_dp_model)
        """
        from modules.aeroinv import aeroinv_reg
        # aeroinv_reg expects y shape (n_data, Nt)
        y = count_matrix.T                   # (n_channels, n_scans)
        F_ALL, _ = aeroinv_reg(y, A, tau00=self.tau00, Niter=self.Niter,
                               reg_scale=self.reg_scale)
        return F_ALL                         # (n_scans, n_dp_model)


# ============================================================
#  Method 8 – AeroInv Chambolle-Pock (Poisson noise, positivity)
# ============================================================

class AeroInvCPPoisson(InversionMethod):
    """
    Chambolle-Pock Algorithm 2 with exact Poisson-noise proximal operators
    and positivity constraint.  Suitable for low-count raw data.
    """

    name = "Chambolle-Pock / Poisson (AeroInv)"

    def __init__(self, tau0: float = 1.0, Niter: int = 500,
                 r_n_tol: float = 1e-6, r_y_tol: float = 0.5,
                 reg_scale: float = 1e-3):
        """
        Parameters
        ----------
        tau0      : initial primal step size
        Niter     : maximum iterations
        r_n_tol   : relative-step stopping tolerance
        r_y_tol   : median data-residual stopping tolerance
        reg_scale : second-order regularisation strength
        """
        self.tau0 = tau0
        self.Niter = Niter
        self.r_n_tol = r_n_tol
        self.r_y_tol = r_y_tol
        self.reg_scale = reg_scale

    def solve(self, A: np.ndarray, counts: np.ndarray,
              dp_grid: np.ndarray) -> InversionResult:
        from modules.aeroinv import alg2_cp_poisson
        n = A.shape[1]
        J = _build_J(n, scale=self.reg_scale)
        W = _build_W(A)
        A_stacked = np.vstack([np.diag(W) @ A, J])
        W_stop = np.ones(n)
        x0 = np.full(n, max(float(counts.mean()), 1.0))
        xn, _, _, _, _, _, N_last = alg2_cp_poisson(
            x0, counts, A_stacked, W, W_stop,
            tau0=self.tau0, Niter=self.Niter,
            r_n_tol=self.r_n_tol, r_y_tol=self.r_y_tol,
        )
        N = np.maximum(xn, 0.0)
        residual = np.linalg.norm(A @ N - counts) / (np.linalg.norm(counts) + 1e-30)
        return InversionResult(N_retrieved=N, dp_grid=dp_grid,
                               residual=residual,
                               info={"iterations": int(N_last),
                                     "tau0": self.tau0,
                                     "reg_scale": self.reg_scale})


# ============================================================
#  Method 9 – AeroInv Newton + Chambolle-Pock (Poisson, Ozon 2020)
# ============================================================

class AeroInvOzon2020(InversionMethod):
    """
    Newton-like outer loop with a Chambolle-Pock inner solver for Poisson
    noise (Ozon et al. 2020).  Uses a diagonal Hessian approximation and
    backtracking line search.  Best for raw photon/particle count data.
    """

    name = "Newton-CP / Poisson – Ozon 2020 (AeroInv)"

    def __init__(self, beta_f: float = 1e-5, max_iter: int = 5000,
                 N_max: int = 50, reg_scale: float = 1e-3):
        """
        Parameters
        ----------
        beta_f    : log-floor to prevent log(0)
        max_iter  : maximum inner Chambolle-Pock iterations per outer step
        N_max     : maximum outer Newton iterations
        reg_scale : second-order regularisation strength
        """
        self.beta_f = beta_f
        self.max_iter = max_iter
        self.N_max = N_max
        self.reg_scale = reg_scale

    def solve(self, A: np.ndarray, counts: np.ndarray,
              dp_grid: np.ndarray) -> InversionResult:
        from modules.aeroinv import ozon2020
        n = A.shape[1]
        J = _build_J(n, scale=self.reg_scale)
        W = _build_W(A)
        f0 = np.full(n, max(float(counts.mean()), 1.0))
        N = ozon2020(
            f0, counts, A, J, W,
            beta_f=self.beta_f,
            max_iter=self.max_iter,
            N_max=self.N_max,
        )
        N = np.maximum(N, 0.0)
        residual = np.linalg.norm(A @ N - counts) / (np.linalg.norm(counts) + 1e-30)
        return InversionResult(N_retrieved=N, dp_grid=dp_grid,
                               residual=residual,
                               info={"beta_f": self.beta_f,
                                     "N_max": self.N_max,
                                     "reg_scale": self.reg_scale})


# ============================================================
#  Methods 10 & 11 – Kalman Filter / Smoother
#  Evolution model : F = I  (identity)
#  State covariance: Q = covariance_second_order_process(α, β, σ₁, σ₂, σ₁₂, var_val·1)
#  Measurement noise: R_k = diag(max(y_k, r_min))  (Poisson, diagonal)
# ============================================================

class KalmanFilterInversion(InversionMethod):
    """
    Forward Extended Kalman Filter.

    State-space model
    -----------------
    x_k = x_{k-1} + w_k ,   w_k ~ N(0, Q)
    y_k = A x_k   + v_k ,   v_k ~ N(0, R_k)

    Q is built from covariance_second_order_process (2nd-order AR in the
    size dimension) with a constant variance vector var_val·ones(n_dp).

    R_k is diagonal with entries max(y_k[i], r_min): Poisson noise
    (variance ≈ count) with a numerical floor for stability.
    """

    name = "Kalman Filter (identity model)"

    def __init__(self, alpha: float = 0.9, beta: float = -0.1,
                 sig1: float = 1e6, sig2: float = 1e6, sig12: float = 0.0,
                 var_val: float = 1e6, r_min: float = 1.0,
                 p0_diag_val: float = 100.0,
                 use_reg: bool = False, reg_var: float = 1.0):
        self.alpha       = alpha
        self.beta        = beta
        self.sig1        = sig1
        self.sig2        = sig2
        self.sig12       = sig12
        self.var_val     = var_val
        self.r_min       = r_min
        self.p0_diag_val = p0_diag_val
        self.use_reg     = use_reg
        self.reg_var     = reg_var

    def _build_Q(self, n: int) -> np.ndarray:
        from modules.algo.stochproc import covariance_second_order_process
        import warnings
        var_vec = np.full(n, self.var_val)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Q = covariance_second_order_process(
                self.alpha, self.beta, self.sig1, self.sig2, self.sig12, var_vec
            )
        Q = 0.5 * (Q + Q.T)
        min_eig = np.linalg.eigvalsh(Q).min()
        if min_eig <= 0:
            Q += (-min_eig + 1e-10 * max(np.trace(Q) / n, 1.0)) * np.eye(n)
        return Q

    @staticmethod
    def _build_D(n: int) -> np.ndarray:
        """Second-order difference operator, shape (n-2, n). Rows are [1, -2, 1]."""
        D   = np.zeros((n - 2, n))
        idx = np.arange(n - 2)
        D[idx, idx]     =  1.0
        D[idx, idx + 1] = -2.0
        D[idx, idx + 2] =  1.0
        return D

    def _augment_for_reg(self, A: np.ndarray, Y: np.ndarray, R0: np.ndarray,
                         n_ch: int, n_dp: int):
        """
        Augment (A, Y, R0, update_R, cbs) with the second-order difference
        operator D and pseudo-measurements of zero.

        Returns (A_aug, Y_aug, R0_aug, update_R_aug, cbs_aug).
        """
        from modules.algo.kalman import make_linear_callbacks
        D      = self._build_D(n_dp)          # (n_dp-2, n_dp)
        n_reg  = n_dp - 2
        n_aug  = n_ch + n_reg
        A_aug  = np.vstack([A, D])            # (n_aug, n_dp)
        Y_aug  = np.vstack([Y, np.zeros((n_reg, Y.shape[1]))])

        R0_aug = np.zeros((n_aug, n_aug))
        R0_aug[:n_ch, :n_ch] = R0
        R0_aug[n_ch:, n_ch:] = self.reg_var * np.eye(n_reg)

        r_min   = self.r_min
        reg_var = self.reg_var

        def update_R_aug(y_t, ws):
            R = np.zeros((n_aug, n_aug))
            R[:n_ch, :n_ch] = np.diag(np.maximum(y_t[:n_ch], r_min))
            R[n_ch:, n_ch:] = reg_var * np.eye(n_reg)
            return R

        cbs_aug = make_linear_callbacks(np.eye(n_dp), A_aug)
        return A_aug, Y_aug, R0_aug, update_R_aug, cbs_aug

    def _make_kf_args(self, n_dp: int, A: np.ndarray,
                      first_counts: np.ndarray, n_scans: int):
        """Build all arguments needed by kalman_filter / kalman_filter_smoother."""
        from modules.algo.kalman import make_linear_callbacks
        Q    = self._build_Q(n_dp)
        R0   = np.diag(np.maximum(first_counts, self.r_min))
        x0   = np.zeros(n_dp)
        P0_d = np.full(n_dp, self.p0_diag_val)
        cbs  = make_linear_callbacks(np.eye(n_dp), A)
        r_min = self.r_min

        def update_R(y_t, ws):
            return np.diag(np.maximum(y_t, r_min))

        t_samp = np.arange(n_scans, dtype=float)
        return x0, P0_d, Q, R0, t_samp, cbs, update_R

    def solve(self, A: np.ndarray, counts: np.ndarray,
              dp_grid: np.ndarray) -> InversionResult:
        from modules.algo.kalman import kalman_filter
        n_dp = A.shape[1]
        x0, P0_d, Q, R0, _, cbs, update_R = self._make_kf_args(
            n_dp, A, counts, 1
        )
        Y  = counts[:, np.newaxis]
        ws = kalman_filter(x0, P0_d, Q, R0, Y, np.array([0.0]), 1.0,
                           update_R=update_R, **cbs)
        N = np.maximum(ws.x_fil_all[:, 0], 0.0)
        residual = np.linalg.norm(A @ N - counts) / (np.linalg.norm(counts) + 1e-30)
        return InversionResult(N_retrieved=N, dp_grid=dp_grid, residual=residual,
                               info={"log_likelihood": ws.log_likelihood})

    def _run_series(self, A: np.ndarray, count_matrix: np.ndarray):
        """Run the forward filter and return the KFWorkspace."""
        from modules.algo.kalman import kalman_filter
        n_scans, n_ch = count_matrix.shape
        n_dp          = A.shape[1]
        x0, P0_d, Q, R0, t_samp, cbs, update_R = self._make_kf_args(
            n_dp, A, count_matrix[0], n_scans
        )
        Y = count_matrix.T
        if self.use_reg and n_dp >= 3:
            _, Y, R0, update_R, cbs = self._augment_for_reg(A, Y, R0, n_ch, n_dp)
        return kalman_filter(x0, P0_d, Q, R0, Y, t_samp, 1.0,
                             update_R=update_R, **cbs)

    def _extract_results(self, ws):
        """Return (N_ALL, std_ALL) from a filter workspace, shape (n_scans, n_dp)."""
        from modules.algo.kalman import filtered_states, filtered_stds
        return np.maximum(filtered_states(ws), 0.0), filtered_stds(ws)

    def solve_series(self, A: np.ndarray, count_matrix: np.ndarray,
                     dp_grid: np.ndarray) -> np.ndarray:
        """Return filtered distributions, shape (n_scans, n_dp)."""
        ws = self._run_series(A, count_matrix)
        N, _ = self._extract_results(ws)
        return N


class KalmanSmootherInversion(KalmanFilterInversion):
    """
    Extended Kalman Filter + Fixed Interval Kalman Smoother (RTS).

    Same state-space model as KalmanFilterInversion; adds a backward
    Rauch-Tung-Striebel pass over the full time series.  For a single
    scan the smoother coincides with the filter.
    """

    name = "Kalman Smoother – FIKS (identity model)"

    def _run_series(self, A: np.ndarray, count_matrix: np.ndarray):
        """Run forward filter + backward RTS smoother, return KFWorkspace."""
        from modules.algo.kalman import kalman_filter_smoother
        n_scans, n_ch = count_matrix.shape
        n_dp          = A.shape[1]
        x0, P0_d, Q, R0, t_samp, cbs, update_R = self._make_kf_args(
            n_dp, A, count_matrix[0], n_scans
        )
        Y = count_matrix.T
        if self.use_reg and n_dp >= 3:
            _, Y, R0, update_R, cbs = self._augment_for_reg(A, Y, R0, n_ch, n_dp)
        return kalman_filter_smoother(x0, P0_d, Q, R0, Y, t_samp, 1.0,
                                      update_R=update_R, **cbs)

    def _extract_results(self, ws):
        """Return (N_ALL, std_ALL) from a smoother workspace, shape (n_scans, n_dp)."""
        from modules.algo.kalman import smoothed_states, smoothed_stds
        return np.maximum(smoothed_states(ws), 0.0), smoothed_stds(ws)

    def solve_series(self, A: np.ndarray, count_matrix: np.ndarray,
                     dp_grid: np.ndarray) -> np.ndarray:
        """Return smoothed distributions, shape (n_scans, n_dp)."""
        ws = self._run_series(A, count_matrix)
        N, _ = self._extract_results(ws)
        return N

    # solve() is inherited from KalmanFilterInversion (filter for single step)


# ============================================================
#  Registry  – used by the GUI to populate the combo-box
# ============================================================

INVERSION_METHODS = {
    TikhonovInversion.name:     TikhonovInversion,
    TruncatedSVDInversion.name: TruncatedSVDInversion,
    NNLSInversion.name:         NNLSInversion,
    EMInversion.name:           EMInversion,
    AeroInvQuick.name:          AeroInvQuick,
    AeroInvPrecondReg.name:     AeroInvPrecondReg,
    AeroInvRegCP.name:              AeroInvRegCP,
    AeroInvCPPoisson.name:          AeroInvCPPoisson,
    AeroInvOzon2020.name:           AeroInvOzon2020,
    KalmanFilterInversion.name:     KalmanFilterInversion,
    KalmanSmootherInversion.name:   KalmanSmootherInversion,
}

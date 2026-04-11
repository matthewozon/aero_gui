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
#  Registry  – used by the GUI to populate the combo-box
# ============================================================

INVERSION_METHODS = {
    TikhonovInversion.name: TikhonovInversion,
    TruncatedSVDInversion.name: TruncatedSVDInversion,
    NNLSInversion.name: NNLSInversion,
    EMInversion.name: EMInversion,
}

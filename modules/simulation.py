"""
modules/simulation.py
---------------------
Simulation of the time evolution of aerosol particle size distributions
via a discretised General Dynamic Equation (GDE).

  dN_i/dt = (nucleation)_i
           + (condensation)_{i-1→i} - (condensation)_{i→i+1}
           + sum_j K_{ij} N_i N_j / 2   (coagulation gain)
           - N_i * sum_j K_{ij} N_j     (coagulation loss)
           - (deposition)_i * N_i

Port your Julia implementations into the stub methods below.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Callable
from scipy.integrate import solve_ivp


# ============================================================
#  Physical kernels
# ============================================================

BOLTZMANN = 1.38065e-23   # J/K
AIR_VISCOSITY = 1.81e-5   # Pa·s
MEAN_FREE_PATH = 65e-9    # m


def cunningham_correction(dp_m: np.ndarray) -> np.ndarray:
    Kn = 2 * MEAN_FREE_PATH / dp_m
    return 1 + Kn * (1.165 + 0.483 * np.exp(-0.997 / Kn))


def brownian_coagulation_kernel(dp_nm: np.ndarray,
                                 T: float = 298.15,
                                 P: float = 101325.0) -> np.ndarray:
    """
    Fuchs coagulation kernel K[i,j] [m³/s].
    Returns matrix of shape (n_bins, n_bins).
    """
    dp = dp_nm * 1e-9
    Cc = cunningham_correction(dp)
    D = BOLTZMANN * T * Cc / (3 * np.pi * AIR_VISCOSITY * dp)  # diffusivity
    v_th = np.sqrt(8 * BOLTZMANN * T / (np.pi * (1e-18 * dp ** 3 * 1.2)))  # thermal velocity (rough)

    n = len(dp)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d_sum = dp[i] + dp[j]
            D_sum = D[i] + D[j]
            K[i, j] = 2 * np.pi * D_sum * d_sum
    return K


def gravitational_settling_velocity(dp_nm: np.ndarray,
                                     rho_p: float = 1500.0,
                                     T: float = 298.15) -> np.ndarray:
    """Stokes settling velocity [m/s]."""
    dp = dp_nm * 1e-9
    g = 9.81
    Cc = cunningham_correction(dp)
    return rho_p * dp ** 2 * g * Cc / (18 * AIR_VISCOSITY)


# ============================================================
#  GDE Simulation configuration
# ============================================================

@dataclass
class SimulationConfig:
    dp_grid_nm: np.ndarray = field(
        default_factory=lambda: np.logspace(np.log10(3), np.log10(1000), 64))
    t_start: float = 0.0          # s
    t_end: float = 3600.0         # s  (1 hour)
    dt_output: float = 60.0       # s  output interval
    temperature: float = 298.15   # K
    pressure: float = 101325.0    # Pa
    particle_density: float = 1500.0  # kg/m³

    # Process switches
    coagulation: bool = True
    condensation: bool = False
    nucleation: bool = False
    deposition: bool = False

    # Condensation: growth rate [nm/s] per bin (user-supplied or computed)
    growth_rate: Optional[np.ndarray] = None

    # Nucleation: source rate [#/(cm³·s)] as a function of time → scalar
    nucleation_source: Optional[Callable] = None

    # Deposition loss rate [1/s] per bin
    deposition_rate: Optional[np.ndarray] = None


# ============================================================
#  GDE Simulator
# ============================================================

class GDESimulator:
    """
    Solves the discretised GDE using scipy's ODE solver.
    Replace the RHS computation with your Julia-equivalent logic.
    """

    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        self.result_times: Optional[np.ndarray] = None
        self.result_N: Optional[np.ndarray] = None   # shape (n_times, n_bins)

    def _build_coag_kernel(self) -> np.ndarray:
        c = self.config
        return brownian_coagulation_kernel(c.dp_grid_nm, c.temperature, c.pressure)

    def _rhs(self, t: float, N: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Right-hand side of dN/dt = ... """
        c = self.config
        dNdt = np.zeros_like(N)

        # --- Coagulation ---
        if c.coagulation:
            # Gain: particles of size j and k coagulate to form i
            #   (simplified: Smoluchowski, sectional method)
            coag_loss = N * (K @ N)
            # Gain term (sectional approximation): pair j+k → i
            coag_gain = np.zeros_like(N)
            dp = c.dp_grid_nm
            for i in range(len(dp)):
                for j in range(i):
                    k_idx = np.searchsorted(dp ** 3, dp[i] ** 3 - dp[j] ** 3)
                    if k_idx < len(dp):
                        coag_gain[i] += 0.5 * K[j, k_idx] * N[j] * N[k_idx]
            dNdt += coag_gain - coag_loss

        # --- Condensation (flux-based, upwind) ---
        if c.condensation and c.growth_rate is not None:
            GR = c.growth_rate   # [nm/s]
            dp = c.dp_grid_nm
            ddp = np.gradient(dp)
            flux = GR * N
            dNdt -= np.gradient(flux) / ddp

        # --- Nucleation ---
        if c.nucleation and c.nucleation_source is not None:
            dNdt[0] += c.nucleation_source(t)

        # --- Deposition ---
        if c.deposition and c.deposition_rate is not None:
            dNdt -= c.deposition_rate * N

        return dNdt

    def run(self, N0: np.ndarray = None) -> None:
        """
        Integrate from t_start to t_end.
        N0 : initial size distribution [#/cm³ per bin], shape (n_bins,).
             Defaults to zero (empty atmosphere).
        """
        c = self.config
        n_bins = len(c.dp_grid_nm)
        if N0 is None:
            N0 = np.zeros(n_bins)

        # Pre-build coagulation kernel (expensive, do once)
        K = self._build_coag_kernel() if c.coagulation else None

        t_eval = np.arange(c.t_start, c.t_end + c.dt_output, c.dt_output)

        sol = solve_ivp(
            fun=lambda t, N: self._rhs(t, N, K),
            t_span=(c.t_start, c.t_end),
            y0=N0,
            method="RK45",
            t_eval=t_eval,
            vectorized=False,
            rtol=1e-4,
            atol=1e-6,
            max_step=c.dt_output,
        )

        self.result_times = sol.t
        self.result_N = sol.y.T   # shape (n_times, n_bins)

    # ------------------------------------------------------------------

    def get_total_number(self) -> np.ndarray:
        """Total number concentration [#/cm³] as function of time."""
        return self.result_N.sum(axis=1)

    def get_heatmap_data(self):
        """Return (times, dp_grid, N_matrix) for 2D heatmap plotting."""
        return self.result_times, self.config.dp_grid_nm, self.result_N

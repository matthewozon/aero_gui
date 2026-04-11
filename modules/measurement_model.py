"""
modules/measurement_model.py
-----------------------------
Forward measurement model for mobility-based particle sizing instruments
(e.g. SMPS, DMA + CPC).

Physical model (modular):
  Raw counts = Efficiency(Dp) * Transfer_function(Dp, V) * N(Dp) * Q * dt + noise

Each component is a callable that can be swapped independently.
"""

import numpy as np
from scipy.special import erf
from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple
import pandas as pd


# ============================================================
#  Parameter dataclasses  (instrument geometry & operating conditions)
# ============================================================

@dataclass
class DMAParameters:
    """Differential Mobility Analyser geometry and flow settings."""
    # Geometry
    inner_radius: float = 0.937e-2   # m  (TSI 3081 inner)
    outer_radius: float = 1.961e-2   # m  (TSI 3081 outer)
    length: float = 44.369e-2        # m  (TSI 3081 effective length)
    # Flows [L/min]
    sheath_flow: float = 5.0
    aerosol_flow: float = 0.5
    # Operating
    temperature: float = 298.15      # K
    pressure: float = 101325.0       # Pa
    n_charges: int = 1                # number of elementary charges on the particles


@dataclass
class CPCParameters:
    """Condensation Particle Counter efficiency parameters."""
    d50: float = 10.0          # nm  – 50% cut-off diameter
    sigma: float = 1.2         # nm  – width of the sigmoid
    max_efficiency: float = 1.0


@dataclass
class ChargerParameters:
    """Bipolar diffusion charger (Wiedensohler approximation)."""
    # Coefficients for +1 charge fraction (Wiedensohler 1988)
    a_pos: list = field(default_factory=lambda: [-26.3328, 35.9044, -21.4608,
                                                   7.0557, -1.3583, 0.1051])
    a_neg: list = field(default_factory=lambda: [-2.3197, 0.6175, 0.6201,
                                                  -0.1105, -0.1260, 0.0297])


@dataclass
class InstrumentParameters:
    """Top-level container passed to MeasurementModel."""
    dma: DMAParameters = field(default_factory=DMAParameters)
    cpc: CPCParameters = field(default_factory=CPCParameters)
    charger: ChargerParameters = field(default_factory=ChargerParameters)
    scan_time: float = 120.0    # s — total scan duration
    n_channels: int = 64        # voltage/diameter channels
    v_min: float = 10.0         # V
    v_max: float = 10000.0      # V


# ============================================================
#  Physical helper functions
# ============================================================

ELEMENTARY_CHARGE = 1.60218e-19   # C
BOLTZMANN = 1.38065e-23           # J/K
AVOGADRO = 6.02214e23
RG   = 8.314        # [J mol-1 K-1], universal gas constant
AIR_VISCOSITY = 1.81e-5           # Pa·s at ~20°C
MEAN_FREE_PATH = 65e-9            # m at ~20°C, 1 atm
ALPHA_CUN = 1.142   # Cunningham Slip coefficients
BETA_CUN  = 0.558   #
GAMMA_CUN = 0.999   #
MAIR = 28.96e-3        # kg/mol  – molar mass of air

def C_slip(Kn: np.ndarray, alpha: float = ALPHA_CUN, beta: float = BETA_CUN, gamma: float = GAMMA_CUN) -> np.ndarray:
    return 1.0 + Kn * (alpha + beta * np.exp(-gamma / Kn))


def cunningham_correction(dp_nm: np.ndarray) -> np.ndarray:
    """Cunningham slip correction factor Cc(Dp)."""
    dp = dp_nm * 1e-9
    Kn = 2 * MEAN_FREE_PATH / dp
    return 1 + Kn * (ALPHA_CUN + BETA_CUN * np.exp(-GAMMA_CUN / Kn))


def electrical_mobility(dp_nm: np.ndarray, n_charges: int,
                         T: float = 298.15, P: float = 101325.0) -> np.ndarray:
    """Electrical mobility Zp [m²/(V·s)] for particles with n_charges charges."""
    dp = dp_nm * 1e-9
    eta = AIR_VISCOSITY   # simplified; can add T/P correction
    Cc = cunningham_correction(dp_nm)
    return (n_charges * ELEMENTARY_CHARGE * Cc) / (3 * np.pi * eta * dp)

def size_from_mobility_and_charge(Zp_array:np.ndarray,n_charges:int,params: DMAParameters) -> np.ndarray:
    """Given an array of electrical mobilities Zp, return the corresponding particle diameters [nm] for n_charges charges."""
    dyn_visc = AIR_VISCOSITY # 1.8e-5*(params.temperature/298.0) ** 0.85                       # [kg.m^{-1}.s^{-1}] Dynamic viscosity of air
    l_gas= MEAN_FREE_PATH # 2.0*dyn_visc/(params.pressure*np.sqrt(8.0*MAIR/(np.pi*RG*params.temperature)))      # [m] Gas mean free path in air

    # use the root finding algorithm (Newtons' method, if necessary, change to a minimization problem) (it should be quite ok because it is a strictly increasing function for which we are looking for a root)
    y = 2.0*l_gas*3.0*np.pi*dyn_visc*Zp_array/(n_charges*ELEMENTARY_CHARGE)
    def f(x:np.ndarray) -> np.ndarray: # the function for which we are looking for a root
        return x*C_slip(x)-y
    
    def fd(x:np.ndarray) -> np.ndarray: # the derivative of f
        return 1.0 + 2.0*ALPHA_CUN*x + BETA_CUN*(1.0+2.0*x)*np.exp(-GAMMA_CUN/x)
    
    # Newton's method seems to work!
    Kn_est = np.full_like(Zp_array, 1.0)
    for _ in range(0,20):
        Kn_est = Kn_est - f(Kn_est)/fd(Kn_est)
    
    return 2.0*l_gas/Kn_est

def voltage_to_diameter(V: np.ndarray, params: DMAParameters,
                          n_charges: int = 1) -> np.ndarray:
    """
    Invert DMA transfer function: given voltage(s), return selected diameter [nm].
    Uses cylindrical DMA geometry.
    """
    r1, r2, L = params.inner_radius, params.outer_radius, params.length
    Qsh = params.sheath_flow / 60000.0   # L/min → m³/s
    # Zp_star = Qsh * ln(r2/r1) / (2*pi*L*V)
    Zp_star : np.ndarray = Qsh * np.log(r2 / r1) / (2 * np.pi * L * V)
    # Invert mobility → diameter (iterative would be exact; here analytical approx)
    # Approximate: use Cc≈1 first, then correct
    # eta = AIR_VISCOSITY
    # dp_approx = (n_charges * ELEMENTARY_CHARGE) / (3 * np.pi * eta * Zp_star) * 1e9  # nm
    return size_from_mobility_and_charge(Zp_star,n_charges,params) * 1.0e9
    # return size_from_mobility_and_charge(Zp_star,n_charges,params)
    # One Newton step with Cc correction
    # Cc = cunningham_correction(dp_approx)
    # dp_corrected = (n_charges * ELEMENTARY_CHARGE * Cc) / (3 * np.pi * eta * Zp_star) * 1e9
    # return dp_corrected




def diameter_to_voltage(dp_nm: np.ndarray, params: DMAParameters,
                         n_charges: int = 1) -> np.ndarray:
    """Compute DMA voltage corresponding to each diameter."""
    Zp = electrical_mobility(dp_nm, n_charges,
                              T=params.temperature, P=params.pressure)
    r1, r2, L = params.inner_radius, params.outer_radius, params.length
    Qsh = params.sheath_flow / 60000.0
    return Qsh * np.log(r2 / r1) / (2 * np.pi * L * Zp)


# ============================================================
#  Individual model components
# ============================================================

def dma_transfer_function_triangular(Zp_grid: np.ndarray,
                                      Zp_star: float,
                                      params: DMAParameters) -> np.ndarray:
    """
    Triangular (ideal) DMA transfer function Ω(Zp).
    beta = Qa / Qsh  (ratio of sample to sheath flow)
    """
    Qa = params.aerosol_flow / 60000.0
    Qsh = params.sheath_flow / 60000.0
    beta = Qa / Qsh
    x = (Zp_grid - Zp_star) / (beta * Zp_star)
    return np.maximum(0.0, 1 - np.abs(x))


def cpc_efficiency(dp_nm: np.ndarray, params: CPCParameters) -> np.ndarray:
    """Sigmoid CPC detection efficiency."""
    return params.max_efficiency * 0.5 * (1 + erf((dp_nm - params.d50) / params.sigma))


# TODO: extent to more charges (Wiedensohler 1988), or add separate charger model
def wiedensohler_charge_fraction(dp_nm: np.ndarray,
                                  n: int,
                                  params: ChargerParameters) -> np.ndarray:
    """
    Bipolar charge fraction f_n(Dp) from Wiedensohler (1988).
    Valid for |n| <= 2 and 1 <= Dp <= 1000 nm.
    """
    log_dp = np.log10(dp_nm)
    if n == 1:
        coef = params.a_pos
    elif n == -1:
        coef = params.a_neg
    else:
        # Boltzmann for higher charges (approximation)
        return np.full_like(dp_nm, 0.0)
    exponent = sum(coef[k] * log_dp ** k for k in range(len(coef)))
    return 10.0 ** exponent


# ============================================================
#  MeasurementModel  – orchestrates the components
# ============================================================

class MeasurementModel:
    """
    Computes the instrument kernel matrix A such that:
        counts_measured ≈ A @ N_true  +  noise

    where N_true is the true size distribution [dN/dlogDp].
    """

    MODELS = {
        "Triangular DMA + Sigmoid CPC": "triangular_sigmoid",
        "Triangular DMA + Step CPC": "triangular_step",
        # Add more variants here
    }

    def __init__(self, params: InstrumentParameters = None,
                 model_type: str = "triangular_sigmoid"):
        self.params = params or InstrumentParameters()
        self.model_type = model_type

        # Will be populated by compute()
        self.voltages: Optional[np.ndarray] = None
        self.diameters: Optional[np.ndarray] = None
        self.kernel: Optional[np.ndarray] = None    # shape (n_channels, n_dp)

    # ------------------------------------------------------------------

    def compute(self, dp_grid_nm: np.ndarray = None) -> None:
        """
        Build the kernel matrix A on dp_grid_nm [nm].
        If dp_grid_nm is None, use a log-spaced grid from 3 to 1000 nm.
        """
        if dp_grid_nm is None:
            dp_grid_nm = np.logspace(np.log10(3), np.log10(1000), 200)

        p = self.params
        # Voltage channel centres (log-spaced)
        self.voltages = np.logspace(np.log10(p.v_min), np.log10(p.v_max),
                                     p.n_channels)
        # self.voltages = np.linspace(p.v_min, p.v_max,p.n_channels)
        # Diameter selected by each voltage channel (singly charged, n=1)
        self.diameters = voltage_to_diameter(self.voltages, p.dma, n_charges=1)
        self.dp_grid = dp_grid_nm

        n_ch = p.n_channels
        n_dp = len(dp_grid_nm)
        A = np.zeros((n_ch, n_dp))

        # Mobility grid for the input particle sizes
        Zp_grid = electrical_mobility(dp_grid_nm, n_charges=1,
                                       T=p.dma.temperature, P=p.dma.pressure)

        for i, V in enumerate(self.voltages):
            Zp_star = electrical_mobility(
                np.array([self.diameters[i]]), 1,
                T=p.dma.temperature, P=p.dma.pressure
            )[0]

            # DMA transfer function
            Omega = dma_transfer_function_triangular(Zp_grid, Zp_star, p.dma)

            # Charge fraction (singly charged)
            f_charge = wiedensohler_charge_fraction(dp_grid_nm, n=self.params.dma.n_charges, params=p.charger)

            # CPC efficiency
            if self.model_type == "triangular_sigmoid":
                eta_cpc = cpc_efficiency(dp_grid_nm, p.cpc)
            else:  # step
                eta_cpc = (dp_grid_nm >= p.cpc.d50).astype(float)

            # Kernel row: A[i,:] = Omega * f_charge * eta_cpc * dlogDp
            dlogDp = np.gradient(np.log10(dp_grid_nm))
            A[i, :] = Omega * f_charge * eta_cpc * dlogDp

        self.kernel = A

    # ------------------------------------------------------------------

    def apply(self, N_true: np.ndarray) -> np.ndarray:
        """
        Compute expected counts from true distribution.
        N_true : shape (n_dp,)  – dN/dlogDp [#/cm³]
        Returns counts : shape (n_channels,)
        """
        if self.kernel is None:
            raise RuntimeError("Call compute() first.")
        return self.kernel @ N_true

    # ------------------------------------------------------------------

    def save(self, filepath: str, fmt: str = "csv") -> None:
        """Save the kernel matrix + metadata to file."""
        if self.kernel is None:
            raise RuntimeError("No kernel to save. Call compute() first.")

        df_kernel = pd.DataFrame(
            self.kernel,
            index=pd.Index(self.voltages, name="Voltage_V"),
            columns=pd.Index(self.dp_grid, name="Dp_nm")
        )
        df_params = pd.DataFrame([{
            "model_type": self.model_type,
            "inner_radius_m": self.params.dma.inner_radius,
            "outer_radius_m": self.params.dma.outer_radius,
            "length_m": self.params.dma.length,
            "sheath_flow_Lmin": self.params.dma.sheath_flow,
            "aerosol_flow_Lmin": self.params.dma.aerosol_flow,
            "temperature_K": self.params.dma.temperature,
            "pressure_Pa": self.params.dma.pressure,
            "cpc_d50_nm": self.params.cpc.d50,
            "v_min": self.params.v_min,
            "v_max": self.params.v_max,
            "n_channels": self.params.n_channels,
        }])

        if fmt == "csv":
            df_kernel.to_csv(filepath)
            param_path = filepath.replace(".csv", "_params.csv")
            df_params.to_csv(param_path, index=False)
        elif fmt == "xlsx":
            with pd.ExcelWriter(filepath) as writer:
                df_kernel.to_excel(writer, sheet_name="Kernel")
                df_params.to_excel(writer, sheet_name="Parameters", index=False)

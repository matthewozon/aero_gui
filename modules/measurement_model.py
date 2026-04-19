"""
modules/measurement_model.py
-----------------------------
Forward measurement model for mobility-based particle sizing instruments
(e.g. SMPS, DMA + CPC).

This module defines the parameter dataclasses used by the GUI and
delegates all numerical computation to the validated implementations in
modules/algo/measurement.py (translated from BAYROSOL AeroMeas.jl).

Changes from the original version
----------------------------------
1.  DMAParameters now carries max_charges (Nq): the maximum number of
    elementary charges considered when computing the kernel.  Negative
    values select negative-polarity DMA operation (standard for most
    SMPS instruments).

2.  Air dynamic viscosity and mean free path are now computed from T and P
    using the formula in modules/algo/measurement._air_properties():
        eta(T)   = 1.8e-5 * (T/298)^0.85          [Pa.s]
        lmfp(T,P) = 2*eta / (P * sqrt(8M/(pi*R*T)))  [m]
    This replaces the previous fixed constants (AIR_VISCOSITY = 1.81e-5,
    MEAN_FREE_PATH = 65e-9) which were only valid at ~20 C / 1 atm.

3.  MeasurementModel.compute() now calls smps3936_transfer_function()
    from modules/algo/measurement.py, implementing the full BAYROSOL
    signal chain (impactor -> Wiedensohler charger -> DMA TF -> CPC).

4.  The old physical helpers are kept as thin wrappers for backwards
    compatibility with any code that imports them directly.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

# ── validated physics from algo/measurement ──────────────────────────
from modules.algo.measurement import (
    _air_properties,
    cunningham_slip,
    mobility_from_size_and_charge,
    size_from_mobility_and_charge,
    smps3936_transfer_function,
    SMPSKernel,
)


# ============================================================
#  Physical constants  (kept for any direct imports)
# ============================================================

ELEMENTARY_CHARGE = 1.60218e-19   # C
BOLTZMANN         = 1.38065e-23   # J/K


# ============================================================
#  Parameter dataclasses
# ============================================================

@dataclass
class DMAParameters:
    """
    Differential Mobility Analyser geometry, flow settings and charge model.

    max_charges (int)
        Maximum number of elementary charges considered when building the
        kernel.  Sign sets DMA polarity:
          max_charges > 0  -> positive-polarity DMA
          max_charges < 0  -> negative-polarity DMA (standard SMPS)
        Absolute value determines how many charge states (1...|Nq|) are
        summed.  Typical range: -3 to -6 for a standard SMPS.
    """
    # Geometry
    inner_radius: float = 0.937e-2    # m  (TSI 3081 inner cylinder)
    outer_radius: float = 1.961e-2    # m  (TSI 3081 outer cylinder)
    length:       float = 44.369e-2   # m  (TSI 3081 effective length)
    # Flows [L/min]
    sheath_flow:  float = 5.0
    aerosol_flow: float = 0.5
    # Operating conditions
    temperature:  float = 298.15      # K
    pressure:     float = 101325.0    # Pa
    # Charge model: Nq (negative = negative-polarity DMA)
    max_charges:  int   = -6


@dataclass
class CPCParameters:
    """Condensation Particle Counter detection efficiency parameters."""
    d50:            float = 10.0    # nm   50% cut-off diameter
    delta50:        float = 1.0     # nm   sigmoid width
    max_efficiency: float = 1.0


@dataclass
class ImpactorParameters:
    """Inertial impactor pre-classifier upstream of the DMA."""
    d50:    float = 1000.0   # nm  50% cut diameter
    delta50: float = 100.0  # nm  sigmoid width


@dataclass
class InstrumentParameters:
    """Top-level container passed to MeasurementModel."""
    dma:      DMAParameters      = field(default_factory=DMAParameters)
    cpc:      CPCParameters      = field(default_factory=CPCParameters)
    impactor: ImpactorParameters = field(default_factory=ImpactorParameters)
    scan_time:  float = 120.0     # s
    n_channels: int   = 64        # number of voltage / diameter channels
    v_min:      float = 10.0      # V
    v_max:      float = 10000.0   # V
    tf_model:   str   = ""        # "" / "Stolzenburg_NDTF" / "Stolzenburg_DTF"


# ============================================================
#  Backwards-compatible physical helpers
#  (now T/P-aware via algo/measurement)
# ============================================================

def cunningham_correction(dp_nm: np.ndarray,
                           T: float = 298.15,
                           P: float = 101325.0) -> np.ndarray:
    """
    Cunningham slip correction Cc(Dp, T, P).
    Previously used a fixed lambda = 65 nm; now computed from T and P.
    """
    _, l_gas = _air_properties(T, P)
    Kn = 2.0 * l_gas / (dp_nm * 1e-9)
    return cunningham_slip(Kn)


def electrical_mobility(dp_nm: np.ndarray, n_charges: int,
                         T: float = 298.15,
                         P: float = 101325.0) -> np.ndarray:
    """
    Electrical mobility Zp(Dp, n, T, P) [m^2/(V.s)].
    Delegates to algo/measurement which uses T/P-dependent eta and lambda.
    """
    return mobility_from_size_and_charge(dp_nm * 1e-9, n_charges, T=T, Pr=P)


def voltage_to_diameter(V: np.ndarray,
                         params: DMAParameters,
                         n_charges: int = 1) -> np.ndarray:
    """
    Invert DMA transfer function: voltage -> selected diameter [nm].
    Uses Newton inversion via size_from_mobility_and_charge (T/P-aware).
    """
    r1, r2, L = params.inner_radius, params.outer_radius, params.length
    Qsh = params.sheath_flow / 60000.0   # L/min -> m^3/s
    Zp_star = Qsh * np.log(r2 / r1) / (2.0 * np.pi * L * V)
    s_m = size_from_mobility_and_charge(
        np.abs(Zp_star),
        abs(n_charges),
        T=params.temperature,
        Pr=params.pressure,
    )
    return s_m * 1e9   # m -> nm


def diameter_to_voltage(dp_nm: np.ndarray,
                         params: DMAParameters,
                         n_charges: int = 1) -> np.ndarray:
    """Diameter [nm] -> DMA voltage [V], T/P-aware."""
    Zp = electrical_mobility(dp_nm, n_charges,
                              T=params.temperature, P=params.pressure)
    r1, r2, L = params.inner_radius, params.outer_radius, params.length
    Qsh = params.sheath_flow / 60000.0
    return Qsh * np.log(r2 / r1) / (2.0 * np.pi * L * Zp)


# ============================================================
#  MeasurementModel
# ============================================================

class MeasurementModel:
    """
    Instrument kernel matrix A such that:
        counts_measured ~ A @ N_true  +  noise

    compute() delegates to smps3936_transfer_function() from
    modules/algo/measurement.py, which implements the full BAYROSOL
    SMPS signal chain and supports:
      - multiple charge states (controlled by DMAParameters.max_charges)
      - T/P-dependent air viscosity and mean free path
      - three DMA transfer function models (triangle, NDTF, DTF)
    """

    MODELS = {
        "Triangular DMA + Sigmoid CPC":   "",
        "Stolzenburg NDTF + Sigmoid CPC": "Stolzenburg_NDTF",
        "Stolzenburg DTF  + Sigmoid CPC": "Stolzenburg_DTF",
    }

    def __init__(self, params: InstrumentParameters = None,
                 model_type: str = ""):
        self.params     = params or InstrumentParameters()
        self.model_type = model_type   # one of the TF model strings above

        # populated by compute()
        self.kernel:    Optional[np.ndarray] = None   # (n_channels, n_dp)
        self.diameters: Optional[np.ndarray] = None   # Dp [nm] per channel
        self.voltages:  Optional[np.ndarray] = None   # V per channel
        self.dp_grid:   Optional[np.ndarray] = None   # evaluation grid [nm]
        self._smps_kernel: Optional[SMPSKernel] = None

    # ------------------------------------------------------------------

    def compute(self, dp_grid_nm: np.ndarray = None) -> None:
        """
        Build the kernel matrix A on dp_grid_nm [nm].

        Parameters
        ----------
        dp_grid_nm : evaluation size grid [nm].  Defaults to 200 log-spaced
                     points from 3 to 1000 nm.

        Signal chain (via smps3936_transfer_function):
          1. Inertial impactor   - sigmoid cutoff at imp.d50
          2. Bipolar charger     - Wiedensohler 1988 (|q|<=2) + Gunn 1956
          3. DMA classifier      - triangle / NDTF / DTF transfer function
          4. CPC counter         - sigmoid detection efficiency at cpc.d50

        max_charges (DMAParameters.max_charges = Nq):
          - |Nq| charge states are summed (1, 2, ..., |Nq|)
          - sign sets polarity: negative -> negative-polarity DMA
        """
        if dp_grid_nm is None:
            dp_grid_nm = np.logspace(np.log10(3), np.log10(1000), 200)

        p   = self.params
        dma = p.dma
        cpc = p.cpc
        imp = p.impactor

        # voltage channel centres (log-spaced) and corresponding diameters
        self.voltages  = np.logspace(np.log10(p.v_min), np.log10(p.v_max),
                                      p.n_channels)
        self.diameters = voltage_to_diameter(self.voltages, dma, n_charges=1)
        self.dp_grid   = dp_grid_nm

        # measurement channel centres in metres (the s_meas grid)
        s_meas = self.diameters * 1e-9   # nm -> m
        s      = dp_grid_nm    * 1e-9   # nm -> m

        # geometric ratio between adjacent channels
        cst_r = float(np.mean(s_meas[1:] / s_meas[:-1]))

        # build the full SMPS kernel via the validated algo implementation
        self._smps_kernel = smps3936_transfer_function(
            s           = s,
            s_meas      = s_meas,
            cst_r_meas  = cst_r,
            # impactor
            s50_imp     = imp.d50    * 1e-9,
            delta50_imp = imp.delta50 * 1e-9,
            # CPC
            s50_cpc     = cpc.d50    * 1e-9,
            delta50_cpc = cpc.delta50 * 1e-9,
            # operating conditions (T/P-dependent eta and mfp used internally)
            T           = dma.temperature,
            Pr          = dma.pressure,
            # charge model: max_charges = Nq
            Nq          = dma.max_charges,
            # DMA flows
            q_a         = dma.aerosol_flow,
            q_sh        = dma.sheath_flow,
            # transfer function model
            TF_model    = self.model_type,
        )

        self.kernel = self._smps_kernel.A   # (n_channels, n_dp)

    # ------------------------------------------------------------------

    def apply(self, N_true: np.ndarray) -> np.ndarray:
        """counts = A @ N_true"""
        if self.kernel is None:
            raise RuntimeError("Call compute() first.")
        return self.kernel @ N_true

    # ------------------------------------------------------------------

    def save(self, filepath: str, fmt: str = "csv") -> None:
        """Save kernel matrix and instrument parameters to CSV or Excel."""
        if self.kernel is None:
            raise RuntimeError("No kernel to save. Call compute() first.")

        dma = self.params.dma
        cpc = self.params.cpc
        imp = self.params.impactor
        eta, lmfp = _air_properties(dma.temperature, dma.pressure)

        df_kernel = pd.DataFrame(
            self.kernel,
            index   = pd.Index(self.voltages, name="Voltage_V"),
            columns = pd.Index(self.dp_grid,  name="Dp_nm"),
        )
        df_params = pd.DataFrame([{
            "tf_model":          self.model_type,
            "inner_radius_m":    dma.inner_radius,
            "outer_radius_m":    dma.outer_radius,
            "length_m":          dma.length,
            "sheath_flow_Lmin":  dma.sheath_flow,
            "aerosol_flow_Lmin": dma.aerosol_flow,
            "temperature_K":     dma.temperature,
            "pressure_Pa":       dma.pressure,
            "eta_Pas":           round(eta, 8),
            "mfp_m":             round(lmfp, 12),
            "max_charges_Nq":    dma.max_charges,
            "cpc_d50_nm":        cpc.d50,
            "cpc_delta50_nm":    cpc.delta50,
            "imp_d50_nm":        imp.d50,
            "imp_delta50_nm":    imp.delta50,
            "v_min_V":           self.params.v_min,
            "v_max_V":           self.params.v_max,
            "n_channels":        self.params.n_channels,
        }])

        if fmt == "csv":
            df_kernel.to_csv(filepath)
            df_params.to_csv(filepath.replace(".csv", "_params.csv"), index=False)
        elif fmt == "xlsx":
            with pd.ExcelWriter(filepath) as writer:
                df_kernel.to_excel(writer, sheet_name="Kernel")
                df_params.to_excel(writer, sheet_name="Parameters", index=False)
        else:
            raise ValueError(f"Unknown format: {fmt}")

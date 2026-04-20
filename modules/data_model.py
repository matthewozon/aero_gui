"""
modules/data_model.py
---------------------
Central data container for particle size distribution data.

Expected file structure (CSV or Excel):
  - A column for scan start timestamps (datetime or numeric)
  - A column (or set of columns) for control voltages or diameters
  - Particle count columns: one per (time, diameter) pair
  - Optional metadata columns: flow rate, temperature, pressure, instrument params...

The class is intentionally flexible: it stores the raw DataFrame and lets
the GUI layer decide which columns map to which role.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class AerosolDataset:
    """Holds one loaded dataset and provides views / transformations."""

    def __init__(self):
        self.raw_df: Optional[pd.DataFrame] = None        # full original table
        self.meta_df: Optional[pd.DataFrame] = None       # metadata / instrument params
        self.count_matrix: Optional[np.ndarray] = None    # shape (n_times, n_diameters)
        self.times: Optional[np.ndarray] = None           # scan start times (numeric or datetime)
        self.diameters: Optional[np.ndarray] = None       # particle diameters [nm]
        self.filepath: Optional[str] = None
        self.column_roles: dict = {}   # mapping: 'time', 'diameter', 'counts', 'meta'

        # TSI-specific fields (populated by load_tsi())
        self.tsi_meta: Optional[Dict[str, Any]] = None       # instrument header block
        self.tsi_scan_meta: Optional[List[dict]] = None      # per-scan metadata
        self.raw_counts: Optional[np.ndarray] = None         # (n_scans, n_times)
        self.raw_times: Optional[np.ndarray] = None          # time axis [s] for raw counts
        self.raw_diameters: Optional[np.ndarray] = None      # (n_scans, n_times) diameters [nm]
        self.dead_time_counts: Optional[np.ndarray] = None   # (n_scans, n_times)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def load(self, filepath: str,
             time_col: str = None,
             diameter_cols: list = None,
             meta_cols: list = None,
             sheet_name: str = 0) -> None:
        """
        Load CSV or Excel file.

        Parameters
        ----------
        filepath    : path to .csv, .xlsx or .xls
        time_col    : name of the column containing scan start timestamps
        diameter_cols : ordered list of column names whose values are particle counts,
                        one column per diameter bin.  If None, the user must call
                        set_column_roles() before the matrix is built.
        meta_cols   : columns to treat as metadata (not count data)
        sheet_name  : for Excel files, sheet to read (name or 0-based index)
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        if path.suffix.lower() == ".csv":
            self.raw_df = pd.read_csv(filepath)
        elif path.suffix.lower() in (".xlsx", ".xls"):
            self.raw_df = pd.read_excel(filepath, sheet_name=sheet_name)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        self.filepath = str(filepath)

        if time_col and diameter_cols:
            self.set_column_roles(time_col, diameter_cols, meta_cols or [])

    def load_tsi(self, filepath: str, sheet_name: Any = 0) -> "TSIResult":  # noqa: F821
        """
        Load a TSI SMPS Excel export file.

        Populates the standard AerosolDataset fields using the dN/dlogDp
        concentration block as the primary count_matrix (best for
        visualisation).  Raw count data, if present in the sheet, is stored
        in the tsi_* extra attributes for use by the inversion pipeline.

        Parameters
        ----------
        filepath   : path to the .xlsx file
        sheet_name : sheet to read — name (str) or 0-based index (int)

        Returns
        -------
        TSIResult  – the parsed TSI data object (also stored on the dataset
                     so callers can access instrument_hints() etc.)
        """
        from modules.tsi_loader import load_tsi_xlsx
        import pandas as pd

        tsi = load_tsi_xlsx(filepath, sheet_name)

        self.filepath = str(filepath)

        # --- standard fields from dN/dlogDp block ---
        self.diameters = tsi.diameters_dn.copy()
        self.count_matrix = tsi.dn_matrix.copy() if tsi.dn_matrix is not None else None

        # --- scan start times ---
        if tsi.scan_datetimes:
            try:
                self.times = pd.to_datetime(tsi.scan_datetimes, dayfirst=True).values
            except Exception:
                self.times = np.array(tsi.scan_datetimes)
        else:
            self.times = np.arange(tsi.n_scans, dtype=float)

        # --- a minimal raw_df so the column-list helpers keep working ---
        # Columns: DateTime + one column per diameter bin
        dp_cols = [f"{d:.3f}" for d in self.diameters]
        if self.count_matrix is not None:
            df_data = {
                "DateTime": list(tsi.scan_datetimes),
                **{dp_cols[k]: self.count_matrix[:, k]
                   for k in range(self.count_matrix.shape[1])},
            }
        else:
            df_data = {"DateTime": list(tsi.scan_datetimes)}
        self.raw_df = pd.DataFrame(df_data)

        self.column_roles = {
            "time":      "DateTime",
            "diameters": dp_cols,
            "meta":      [],
        }

        # --- TSI-specific extras ---
        self.tsi_meta       = tsi.instrument_meta
        self.tsi_scan_meta  = tsi.scan_meta
        self.raw_counts      = tsi.raw_counts
        self.raw_times       = tsi.raw_times
        self.raw_diameters   = tsi.raw_diameters
        self.dead_time_counts = tsi.dead_time_counts

        return tsi

    def set_column_roles(self, time_col: str,
                         diameter_cols: list,
                         meta_cols: list = None) -> None:
        """
        Assign columns to roles and build the count matrix.
        Call this after load() whenever column mapping is known.
        """
        if self.raw_df is None:
            raise RuntimeError("No data loaded. Call load() first.")

        self.column_roles = {
            "time": time_col,
            "diameters": diameter_cols,
            "meta": meta_cols or [],
        }

        # --- Times ---
        t_series = self.raw_df[time_col]
        if pd.api.types.is_datetime64_any_dtype(t_series):
            self.times = t_series.values
        else:
            try:
                self.times = pd.to_datetime(t_series).values
            except Exception:
                self.times = t_series.values  # keep as-is (numeric)

        # --- Diameters (extract from column names if possible) ---
        try:
            self.diameters = np.array([float(c) for c in diameter_cols])
        except ValueError:
            self.diameters = np.arange(len(diameter_cols), dtype=float)

        # --- Count matrix ---
        self.count_matrix = self.raw_df[diameter_cols].to_numpy(dtype=float)

        # --- Metadata ---
        if meta_cols:
            self.meta_df = self.raw_df[meta_cols].copy()

    # ------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------

    def get_counts(self, log_scale: bool = False) -> np.ndarray:
        """Return count matrix, optionally log10-transformed (zeros -> NaN)."""
        if self.count_matrix is None:
            raise RuntimeError("Count matrix not built. Call set_column_roles().")
        C = self.count_matrix.copy()
        if log_scale:
            C[C <= 0] = np.nan
            C = np.log10(C)
        return C

    def sum_over_time_interval(self, t_start, t_end) -> np.ndarray:
        """Sum counts over scans whose start time falls in [t_start, t_end]."""
        mask = (self.times >= t_start) & (self.times <= t_end)
        return self.count_matrix[mask, :].sum(axis=0)

    def average_over_time_interval(self, t_start, t_end) -> np.ndarray:
        mask = (self.times >= t_start) & (self.times <= t_end)
        return self.count_matrix[mask, :].mean(axis=0)

    def sum_over_diameter_range(self, d_min: float, d_max: float) -> np.ndarray:
        """Sum counts over diameter bins in [d_min, d_max] for each scan."""
        mask = (self.diameters >= d_min) & (self.diameters <= d_max)
        return self.count_matrix[:, mask].sum(axis=1)

    def total_number_concentration(self) -> np.ndarray:
        """Total particle count summed over all sizes, for each scan."""
        return self.count_matrix.sum(axis=1)

    def normalize_by_dlogdp(self) -> np.ndarray:
        """
        Return dN/dlogDp — normalise each column by the log-width of the bin.
        Assumes self.diameters are bin centres in nm.
        """
        d = self.diameters
        log_widths = np.gradient(np.log10(d))
        return self.count_matrix / log_widths[np.newaxis, :]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_subset(self, filepath: str,
                    columns: list = None,
                    fmt: str = "csv") -> None:
        """
        Save a subset of the raw DataFrame to CSV or Excel.

        Parameters
        ----------
        filepath : output path
        columns  : list of column names to include (None = all)
        fmt      : 'csv' or 'xlsx'
        """
        if self.raw_df is None:
            raise RuntimeError("No data to save.")
        df = self.raw_df[columns] if columns else self.raw_df.copy()

        if fmt == "csv":
            df.to_csv(filepath, index=False)
        elif fmt == "xlsx":
            df.to_excel(filepath, index=False)
        else:
            raise ValueError(f"Unknown format: {fmt}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def n_scans(self) -> int:
        return 0 if self.count_matrix is None else self.count_matrix.shape[0]

    @property
    def n_bins(self) -> int:
        return 0 if self.count_matrix is None else self.count_matrix.shape[1]

    @property
    def columns(self) -> list:
        return list(self.raw_df.columns) if self.raw_df is not None else []

    def summary(self) -> str:
        if self.raw_df is None:
            return "No dataset loaded."
        base = (
            f"File   : {self.filepath}\n"
            f"Scans  : {self.n_scans}\n"
            f"Bins   : {self.n_bins}\n"
            f"Diam   : {self.diameters[0]:.1f} – {self.diameters[-1]:.1f} nm"
        ) if self.count_matrix is not None else (
            f"File loaded ({len(self.raw_df)} rows, {len(self.raw_df.columns)} cols). "
            "Column roles not yet assigned."
        )
        extra = ""
        if self.raw_counts is not None:
            extra = (f"\nRaw cts: {self.raw_counts.shape[0]} scans × "
                     f"{self.raw_counts.shape[1]} time steps")
        return base + extra

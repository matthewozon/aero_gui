"""
ui/tab_inversion.py
-------------------
Tab 3: Data inversion — recover N(Dp) from raw instrument counts.

Requires
--------
  Data tab   : dataset.count_matrix must be populated (load a file in the
               Data tab and apply column mapping, or load a TSI file).
  Model tab  : tab_model.model.kernel must be computed (click "Compute
               Model Kernel" in the Measurement Model tab).

Algorithms
----------
  Classical (always available)
    • Tikhonov regularisation      – closed-form, L-curve available
    • Truncated SVD                – rank-k pseudo-inverse
    • NNLS                         – non-negative least squares (scipy)
    • EM                           – Poisson EM (iterative)

  AeroInv (require modules/aeroinv)
    • Quick estimate               – non-iterative, rough baseline
    • Precond. regularised         – closed-form with smoothness prior
    • Chambolle-Pock / Gaussian    – primal-dual, positivity, warm-starts
    • Chambolle-Pock / Poisson     – exact Poisson proximal, positivity
    • Newton-CP / Poisson (Ozon)   – Newton outer + CP inner, Poisson noise
"""

import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QComboBox, QLabel, QFileDialog,
    QGroupBox, QDoubleSpinBox, QSpinBox, QSplitter,
    QTextEdit, QMessageBox, QCheckBox, QTabWidget,
    QFrame, QSizePolicy, QScrollArea,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from modules.data_model import AerosolDataset
from modules.inversion import (
    INVERSION_METHODS, TikhonovInversion, AeroInvRegCP,
    KalmanFilterInversion, KalmanSmootherInversion,
)


# ── per-algorithm parameter definitions ──────────────────────────────────────
# Each entry: list of (label, attr_name, widget_type, min, max, default, decimals)
# widget_type: 'double' | 'int'

_PARAM_DEFS = {
    "Tikhonov Regularisation": [
        ("λ (regularisation):", "spn_lambda",  "double", 1e-10, 1e6,  1e-3, 8),
        ("Order (0/1/2):",      "spn_order",   "int",    0,     2,    0,    0),
    ],
    "Truncated SVD": [
        ("# singular values k:", "spn_k", "int", 1, 500, 20, 0),
    ],
    "EM (Expectation-Maximisation)": [
        ("Max iterations:", "spn_niter", "int",    10,  5000, 100,  0),
        ("Tolerance:",      "spn_tol",   "double", 1e-12, 1e-2, 1e-6, 10),
    ],
    "Quick estimate (AeroInv)": [],    # no user parameters
    "Precond. regularised (AeroInv)": [
        ("Density scale u₀:", "spn_u0",     "double", 1.0, 1e9, 1e4, 1),
        ("Rel. uncertainty:", "spn_rel_un", "double", 1e-4, 0.99, 0.1, 4),
    ],
    "Chambolle-Pock / Gaussian (AeroInv)": [
        ("Initial step τ₀:", "spn_tau00", "double", 1.0,  1e18, 1e15, 0),
        ("Iterations:",      "spn_niter", "int",    10,   5000, 1000, 0),
        ("Reg. scale:",      "spn_reg",   "double", 1e-6, 1e3,  1.0,  6),
    ],
    "Chambolle-Pock / Poisson (AeroInv)": [
        ("Initial step τ₀:",    "spn_tau0",    "double", 1e-3, 1e6,  1.0,  3),
        ("Max iterations:",     "spn_niter",   "int",    10,   5000, 500,  0),
        ("Step tol. rₙ:",       "spn_r_n_tol", "double", 1e-10,0.1,  1e-6, 10),
        ("Data resid. tol. rᵧ:","spn_r_y_tol", "double", 0.01, 2.0,  0.5,  3),
        ("Reg. scale:",         "spn_reg",     "double", 1e-6, 1.0,  1e-3, 6),
    ],
    "Newton-CP / Poisson – Ozon 2020 (AeroInv)": [
        ("Log-floor β:",      "spn_beta_f",   "double", 1e-10, 1.0,  1e-5, 10),
        ("Inner iterations:", "spn_max_iter", "int",    100,   50000,5000, 0),
        ("Outer iterations:", "spn_N_max",    "int",    1,     500,  50,   0),
        ("Reg. scale:",       "spn_reg",      "double", 1e-6,  1.0,  1e-3, 6),
    ],
    # NNLS has no parameters
    "NNLS (non-negative least squares)": [],
    # ── Kalman methods share the same parameter set ──────────────────
    "Kalman Filter (identity model)": [
        ("α (AR2 size-corr.):",  "spn_alpha",   "double", -1.0,  1.0,   0.9,  3),
        ("β (AR2 size-corr.):",  "spn_beta",    "double", -1.0,  0.0,  -0.1,  3),
        ("σ₁ (init. var [0]):",  "spn_sig1",    "double",  0.0,  1e12,  1e6,  2),
        ("σ₂ (init. var [1]):",  "spn_sig2",    "double",  0.0,  1e12,  1e6,  2),
        ("σ₁₂ (init. cov):",     "spn_sig12",   "double", -1e12, 1e12,  0.0,  2),
        ("Variance / bin:",      "spn_var_val", "double",  1e-6,  1e15,  1e6,  2),
        ("Min. count (R floor):","spn_r_min",   "double",  1e-6,  1e9,   1.0,  4),
    ],
    "Kalman Smoother – FIKS (identity model)": [
        ("α (AR2 size-corr.):",  "spn_alpha",   "double", -1.0,  1.0,   0.9,  3),
        ("β (AR2 size-corr.):",  "spn_beta",    "double", -1.0,  0.0,  -0.1,  3),
        ("σ₁ (init. var [0]):",  "spn_sig1",    "double",  0.0,  1e12,  1e6,  2),
        ("σ₂ (init. var [1]):",  "spn_sig2",    "double",  0.0,  1e12,  1e6,  2),
        ("σ₁₂ (init. cov):",     "spn_sig12",   "double", -1e12, 1e12,  0.0,  2),
        ("Variance / bin:",      "spn_var_val", "double",  1e-6,  1e15,  1e6,  2),
        ("Min. count (R floor):","spn_r_min",   "double",  1e-6,  1e9,   1.0,  4),
    ],
}


class InversionTab(QWidget):
    def __init__(self, dataset: AerosolDataset, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.main_window = parent
        self.last_result = None
        self._kernel_csv = None     # fallback kernel from CSV
        self._dp_grid_csv = None
        self._param_widgets: dict = {}   # attr_name → widget
        self._last_A      = None    # kernel used in last inversion
        self._last_counts = None    # aligned counts used in last inversion

        self._build_ui()

        # Poll data/model status every 2 s so labels update automatically
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_status)
        self._timer.start(2000)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QHBoxLayout(self)

        left_w = QWidget()
        left = QVBoxLayout(left_w)
        left.setSpacing(8)

        # ── Data status ───────────────────────────────────────────────
        grp_data = QGroupBox("Data  (from Data tab)")
        dl = QVBoxLayout(grp_data)
        self.lbl_data_status = QLabel("—")
        self.lbl_data_status.setWordWrap(True)
        dl.addWidget(self.lbl_data_status)
        left.addWidget(grp_data)

        # ── Model status ──────────────────────────────────────────────
        grp_model = QGroupBox("Measurement Model  (from Model tab)")
        ml = QVBoxLayout(grp_model)
        self.lbl_model_status = QLabel("—")
        self.lbl_model_status.setWordWrap(True)
        ml.addWidget(self.lbl_model_status)
        self.btn_load_kernel = QPushButton("Load fallback kernel CSV…")
        self.btn_load_kernel.clicked.connect(self._load_kernel)
        ml.addWidget(self.btn_load_kernel)
        left.addWidget(grp_model)

        # ── Algorithm selector ────────────────────────────────────────
        grp_alg = QGroupBox("Algorithm")
        al = QVBoxLayout(grp_alg)
        self.cmb_method = QComboBox()
        self.cmb_method.addItems(list(INVERSION_METHODS.keys()))
        self.cmb_method.currentTextChanged.connect(self._update_param_ui)
        al.addWidget(self.cmb_method)
        left.addWidget(grp_alg)

        # ── Algorithm parameters (dynamic) ────────────────────────────
        self.grp_params = QGroupBox("Algorithm Parameters")
        self.param_layout = QGridLayout(self.grp_params)
        left.addWidget(self.grp_params)

        # ── Scan selection ────────────────────────────────────────────
        grp_scan = QGroupBox("Scan Selection")
        sl = QGridLayout(grp_scan)
        sl.addWidget(QLabel("Scan index:"), 0, 0)
        self.spn_scan_idx = QSpinBox()
        self.spn_scan_idx.setRange(0, 9999)
        sl.addWidget(self.spn_scan_idx, 0, 1)
        self.chk_all_scans = QCheckBox("Invert all scans")
        sl.addWidget(self.chk_all_scans, 1, 0, 1, 2)
        left.addWidget(grp_scan)

        # ── Actions ───────────────────────────────────────────────────
        self.btn_run = QPushButton("Run Inversion")
        self.btn_run.clicked.connect(self._run_inversion)
        left.addWidget(self.btn_run)

        self.btn_save = QPushButton("Save result…")
        self.btn_save.clicked.connect(self._save_result)
        left.addWidget(self.btn_save)

        self.btn_lcurve = QPushButton("L-curve (Tikhonov only)")
        self.btn_lcurve.clicked.connect(self._plot_lcurve)
        left.addWidget(self.btn_lcurve)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(130)
        left.addWidget(self.log)
        left.addStretch()

        # ── Right: plot ───────────────────────────────────────────────
        right_w = QWidget()
        right = QVBoxLayout(right_w)
        self.fig = Figure(figsize=(9, 6), facecolor="#1e1e2e")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumSize(100, 100)
        self.toolbar = NavigationToolbar(self.canvas, self)
        right.addWidget(self.toolbar)
        right.addWidget(self.canvas)

        scroll_left = QScrollArea()
        scroll_left.setWidget(left_w)
        scroll_left.setWidgetResizable(True)
        scroll_left.setFixedWidth(326)
        scroll_left.setFrameShape(QFrame.NoFrame)
        scroll_left.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(scroll_left)
        splitter.addWidget(right_w)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter)

        # Initialise dynamic parameter section and status labels
        self._update_param_ui()
        self._refresh_status()

    # ------------------------------------------------------------------
    # Status polling
    # ------------------------------------------------------------------

    def _refresh_status(self):
        """Update the data and model status labels."""
        # Data
        if self.dataset.count_matrix is not None:
            dp = self.dataset.diameters
            self.lbl_data_status.setText(
                f"✓  {self.dataset.n_scans} scans  ×  {self.dataset.n_bins} bins\n"
                f"   Dp: {dp[0]:.1f} – {dp[-1]:.1f} nm"
            )
            self.lbl_data_status.setStyleSheet("color: #a6e3a1;")
        else:
            self.lbl_data_status.setText(
                "✗  No data loaded.\n"
                "   Load a file in the Data tab."
            )
            self.lbl_data_status.setStyleSheet("color: #f38ba8;")

        # Model
        K, dp_grid = self._get_kernel()
        if K is not None:
            src = "Model tab" if self._kernel_from_model() else "CSV file"
            self.lbl_model_status.setText(
                f"✓  {K.shape[0]} channels  ×  {K.shape[1]} dp points  [{src}]\n"
                f"   Dp: {dp_grid[0]:.1f} – {dp_grid[-1]:.1f} nm"
            )
            self.lbl_model_status.setStyleSheet("color: #a6e3a1;")
        else:
            self.lbl_model_status.setText(
                "✗  No kernel available.\n"
                "   Compute one in the Model tab, or load a CSV."
            )
            self.lbl_model_status.setStyleSheet("color: #f38ba8;")

    def _kernel_from_model(self) -> bool:
        if self.main_window and hasattr(self.main_window, "tab_model"):
            m = self.main_window.tab_model.model
            return m is not None and m.kernel is not None
        return False

    # ------------------------------------------------------------------
    # Dynamic parameter UI
    # ------------------------------------------------------------------

    def _update_param_ui(self):
        """Rebuild parameter widgets for the selected algorithm."""
        while self.param_layout.count():
            item = self.param_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._param_widgets.clear()

        method_name = self.cmb_method.currentText()
        defs = _PARAM_DEFS.get(method_name, [])

        if not defs:
            self.param_layout.addWidget(
                QLabel("No configurable parameters."), 0, 0, 1, 2
            )
            return

        for row, (label, attr, kind, lo, hi, default, decs) in enumerate(defs):
            self.param_layout.addWidget(QLabel(label), row, 0)
            if kind == "double":
                w = QDoubleSpinBox()
                w.setRange(lo, hi)
                w.setDecimals(decs)
                w.setValue(default)
            else:
                w = QSpinBox()
                w.setRange(int(lo), int(hi))
                w.setValue(int(default))
            self.param_layout.addWidget(w, row, 1)
            self._param_widgets[attr] = w

        # Enable L-curve button only for Tikhonov
        self.btn_lcurve.setEnabled("Tikhonov" in method_name)

    def _get_param(self, attr: str):
        w = self._param_widgets.get(attr)
        return w.value() if w is not None else None

    # ------------------------------------------------------------------
    # Build method instance from current UI state
    # ------------------------------------------------------------------

    def _get_method_instance(self):
        name = self.cmb_method.currentText()
        cls = INVERSION_METHODS[name]

        p = self._param_widgets

        if name == "Tikhonov Regularisation":
            return cls(lambda_reg=p["spn_lambda"].value(),
                       order=p["spn_order"].value())
        elif name == "Truncated SVD":
            return cls(k=p["spn_k"].value())
        elif name == "EM (Expectation-Maximisation)":
            return cls(n_iter=p["spn_niter"].value(),
                       tol=p["spn_tol"].value())
        elif name == "Precond. regularised (AeroInv)":
            return cls(u0=p["spn_u0"].value(),
                       rel_un=p["spn_rel_un"].value())
        elif name == "Chambolle-Pock / Gaussian (AeroInv)":
            return cls(tau00=p["spn_tau00"].value(),
                       Niter=p["spn_niter"].value(),
                       reg_scale=p["spn_reg"].value())
        elif name == "Chambolle-Pock / Poisson (AeroInv)":
            return cls(tau0=p["spn_tau0"].value(),
                       Niter=p["spn_niter"].value(),
                       r_n_tol=p["spn_r_n_tol"].value(),
                       r_y_tol=p["spn_r_y_tol"].value(),
                       reg_scale=p["spn_reg"].value())
        elif name == "Newton-CP / Poisson – Ozon 2020 (AeroInv)":
            return cls(beta_f=p["spn_beta_f"].value(),
                       max_iter=p["spn_max_iter"].value(),
                       N_max=p["spn_N_max"].value(),
                       reg_scale=p["spn_reg"].value())
        elif name in ("Kalman Filter (identity model)",
                      "Kalman Smoother – FIKS (identity model)"):
            return cls(alpha   = p["spn_alpha"].value(),
                       beta    = p["spn_beta"].value(),
                       sig1    = p["spn_sig1"].value(),
                       sig2    = p["spn_sig2"].value(),
                       sig12   = p["spn_sig12"].value(),
                       var_val = p["spn_var_val"].value(),
                       r_min   = p["spn_r_min"].value())
        # Quick estimate and NNLS have no parameters
        return cls()

    # ------------------------------------------------------------------
    # Kernel access
    # ------------------------------------------------------------------

    def _get_kernel(self):
        """Return (kernel, dp_grid): prefer model tab, fall back to CSV."""
        if self.main_window and hasattr(self.main_window, "tab_model"):
            m = self.main_window.tab_model.model
            if m is not None and m.kernel is not None:
                return m.kernel, m.dp_grid
        if self._kernel_csv is not None:
            return self._kernel_csv, self._dp_grid_csv
        return None, None

    def _load_kernel(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load kernel CSV", "", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            df = pd.read_csv(path, index_col=0)
            self._kernel_csv = df.values
            self._dp_grid_csv = df.columns.to_numpy(dtype=float)
            self._log(
                f"Fallback kernel loaded: {path.split('/')[-1]}\n"
                f"Shape: {self._kernel_csv.shape}"
            )
            self._refresh_status()
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))

    # ------------------------------------------------------------------
    # Run inversion
    # ------------------------------------------------------------------

    def _run_inversion(self):
        # Guard: data
        if self.dataset.count_matrix is None:
            QMessageBox.warning(
                self, "No data",
                "No data loaded.\nLoad a file in the Data tab first."
            )
            return

        # Guard: kernel
        A, dp_grid = self._get_kernel()
        if A is None:
            QMessageBox.warning(
                self, "No kernel",
                "No measurement kernel available.\n"
                "Compute one in the Measurement Model tab, or load a CSV."
            )
            return

        method = self._get_method_instance()
        name = self.cmb_method.currentText()

        try:
            if self.chk_all_scans.isChecked():
                self._run_all_scans(method, name, A, dp_grid)
            else:
                self._run_single_scan(method, A, dp_grid)
        except Exception as e:
            QMessageBox.critical(self, "Inversion error", str(e))

    def _run_single_scan(self, method, A, dp_grid):
        idx = self.spn_scan_idx.value()
        if idx >= self.dataset.n_scans:
            QMessageBox.warning(self, "Index out of range",
                                f"Scan index must be 0–{self.dataset.n_scans - 1}.")
            return
        counts = self._align_counts(self.dataset.count_matrix[idx], A.shape[0])
        result = method.solve(A, counts, dp_grid)
        self.last_result = result.N_retrieved
        self._last_A = A
        self._last_counts = counts
        self._plot_single(result, dp_grid, idx)
        self._log(
            f"[{method.name}]  scan {idx}\n"
            f"  Residual : {result.residual:.4e}\n"
            f"  N range  : [{result.N_retrieved.min():.3g}, "
            f"{result.N_retrieved.max():.3g}]\n"
            + ("  " + str(result.info) if result.info else "")
        )
        if self.main_window:
            self.main_window.set_status(f"Inversion complete — scan {idx}.")

    def _run_all_scans(self, method, name, A, dp_grid):
        """Invert all scans; use aeroinv_reg's native warm-start if selected."""
        count_matrix = self._align_matrix(self.dataset.count_matrix, A.shape[0])
        if hasattr(method, 'solve_series'):
            self.last_result = method.solve_series(A, count_matrix, dp_grid)
        else:
            results = []
            for i in range(count_matrix.shape[0]):
                r = method.solve(A, count_matrix[i], dp_grid)
                results.append(r.N_retrieved)
            self.last_result = np.array(results)

        self._last_A = A
        self._last_counts = count_matrix   # (n_scans, n_channels)
        self._plot_heatmap(dp_grid)
        self._log(
            f"[{method.name}]  all {self.dataset.n_scans} scans\n"
            f"  Result shape: {self.last_result.shape}"
        )
        if self.main_window:
            self.main_window.set_status(
                f"Inversion complete — {self.dataset.n_scans} scans."
            )

    # ------------------------------------------------------------------
    # Count alignment
    # ------------------------------------------------------------------

    @staticmethod
    def _align_counts(counts: np.ndarray, n_channels: int) -> np.ndarray:
        """Interpolate or truncate counts to match kernel channel count."""
        if len(counts) == n_channels:
            return counts
        x_old = np.linspace(0, 1, len(counts))
        x_new = np.linspace(0, 1, n_channels)
        return np.interp(x_new, x_old, counts)

    @staticmethod
    def _align_matrix(count_matrix: np.ndarray, n_channels: int) -> np.ndarray:
        """Row-wise interpolation of (n_scans, n_bins) to (n_scans, n_channels)."""
        if count_matrix.shape[1] == n_channels:
            return count_matrix
        out = np.zeros((count_matrix.shape[0], n_channels))
        x_old = np.linspace(0, 1, count_matrix.shape[1])
        x_new = np.linspace(0, 1, n_channels)
        for i, row in enumerate(count_matrix):
            out[i] = np.interp(x_new, x_old, row)
        return out

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def _get_channel_diameters(self, n_channels: int):
        """Return (dp_array, is_diameter_nm). Prefers model tab diameters."""
        if self.main_window and hasattr(self.main_window, 'tab_model'):
            m = self.main_window.tab_model.model
            if (m is not None and m.diameters is not None
                    and len(m.diameters) == n_channels):
                return m.diameters, True
        return np.arange(1, n_channels + 1), False

    def _style_ax(self, ax):
        ax.set_facecolor("#181825")
        ax.tick_params(colors="#cdd6f4")
        for s in ax.spines.values():
            s.set_edgecolor("#45475a")

    def _plot_single(self, result, dp_grid, scan_idx):
        N = result.N_retrieved
        self.fig.clear()

        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212)
        self._style_ax(ax1)
        self._style_ax(ax2)

        # ── Top: retrieved size distribution ──────────────────────
        ax1.semilogx(dp_grid, N, color="#a6e3a1", linewidth=2)
        ax1.fill_between(dp_grid, N, alpha=0.25, color="#a6e3a1")
        ax1.set_xlabel("Diameter Dp (nm)", color="#cdd6f4")
        ax1.set_ylabel("dN/dlogDp", color="#cdd6f4")
        ax1.set_title(f"Retrieved size distribution — scan {scan_idx}",
                      color="#cdd6f4")

        # ── Bottom: A · N̂ vs data ─────────────────────────────────
        Y_rec = self._last_A @ N
        dp_ch, is_dp = self._get_channel_diameters(len(Y_rec))
        ax2.plot(dp_ch, self._last_counts, color="#89b4fa", linewidth=1.5,
                 label="Data")
        ax2.plot(dp_ch, Y_rec, color="#f38ba8", linewidth=1.5,
                 linestyle="--", label="A · N̂")
        if is_dp:
            ax2.set_xscale("log")
        ax2.set_xlabel("Selected Dp (nm)" if is_dp else "Channel index",
                       color="#cdd6f4")
        ax2.set_ylabel("Counts", color="#cdd6f4")
        ax2.set_title("Data vs A · N̂  (reconstructed measurements)",
                      color="#cdd6f4")
        ax2.legend(fontsize=8, labelcolor="#cdd6f4",
                   facecolor="#313244", edgecolor="#45475a")

        self.fig.tight_layout()
        self.canvas.draw()

    def _plot_heatmap(self, dp_grid):
        N_ALL = self.last_result          # (n_scans, n_dp)
        self.fig.clear()

        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212)
        self._style_ax(ax1)
        self._style_ax(ax2)

        times = np.arange(N_ALL.shape[0])

        # ── Top: retrieved N(Dp, t) ────────────────────────────────
        im1 = ax1.pcolormesh(times, dp_grid, N_ALL.T,
                              cmap="viridis", shading="auto")
        ax1.set_yscale("log")
        cb1 = self.fig.colorbar(im1, ax=ax1)
        cb1.ax.tick_params(colors="#cdd6f4")
        ax1.set_xlabel("Scan index", color="#cdd6f4")
        ax1.set_ylabel("Dp (nm)", color="#cdd6f4")
        ax1.set_title("Retrieved dN/dlogDp — all scans", color="#cdd6f4")

        # ── Bottom: A · N̂ heatmap ─────────────────────────────────
        if self._last_A is not None:
            Y_ALL = self._last_A @ N_ALL.T    # (n_channels, n_scans)
            dp_ch, is_dp = self._get_channel_diameters(Y_ALL.shape[0])
            im2 = ax2.pcolormesh(times, dp_ch, Y_ALL,
                                  cmap="plasma", shading="auto")
            if is_dp:
                ax2.set_yscale("log")
            cb2 = self.fig.colorbar(im2, ax=ax2)
            cb2.ax.tick_params(colors="#cdd6f4")
            ax2.set_xlabel("Scan index", color="#cdd6f4")
            ax2.set_ylabel("Selected Dp (nm)" if is_dp else "Channel index",
                           color="#cdd6f4")
            ax2.set_title("A · N̂  (reconstructed measurements) — all scans",
                          color="#cdd6f4")

        self.fig.tight_layout()
        self.canvas.draw()

    # ------------------------------------------------------------------
    # L-curve (Tikhonov only)
    # ------------------------------------------------------------------

    def _plot_lcurve(self):
        A, dp_grid = self._get_kernel()
        if A is None or self.dataset.count_matrix is None:
            QMessageBox.warning(self, "L-curve", "Need both kernel and data.")
            return
        idx = self.spn_scan_idx.value()
        counts = self._align_counts(self.dataset.count_matrix[idx], A.shape[0])
        inv = TikhonovInversion()
        lambdas, res, sol = inv.l_curve(A, counts, dp_grid)

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("#181825")
        ax.loglog(res, sol, "o-", color="#f38ba8", markersize=4)
        ax.set_xlabel("Residual norm", color="#cdd6f4")
        ax.set_ylabel("Solution norm", color="#cdd6f4")
        ax.set_title("L-curve (Tikhonov)", color="#cdd6f4")
        ax.tick_params(colors="#cdd6f4")
        for s in ax.spines.values():
            s.set_edgecolor("#45475a")
        self.fig.tight_layout()
        self.canvas.draw()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save_result(self):
        if self.last_result is None:
            QMessageBox.warning(self, "Save", "Run inversion first.")
            return
        _, dp_grid = self._get_kernel()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save result", "inversion_result.csv", "CSV (*.csv)"
        )
        if not path:
            return
        if self.last_result.ndim == 1:
            df = pd.DataFrame({"Dp_nm": dp_grid,
                               "dNdlogDp": self.last_result})
        else:
            df = pd.DataFrame(
                self.last_result,
                columns=[f"Dp_{d:.1f}nm" for d in dp_grid]
            )
        df.to_csv(path, index=False)
        self._log(f"Saved → {path}")

    # ------------------------------------------------------------------

    def _log(self, msg: str):
        self.log.append(msg)

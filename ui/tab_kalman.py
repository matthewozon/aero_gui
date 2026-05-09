"""
ui/tab_kalman.py
----------------
Dedicated tab for sequential Bayesian inversion using the Kalman filter
and Fixed Interval Kalman Smoother (FIKS).

State-space model
-----------------
x_k = x_{k-1} + w_k ,   w_k ~ N(0, Q)     (identity evolution)
y_k = A x_k   + v_k ,   v_k ~ N(0, R_k)   (linear measurement)

Q  : built from covariance_second_order_process — 2nd-order AR in size space,
     controlled by α, β, σ₁, σ₂, σ₁₂, and a constant per-bin variance.
R_k: diagonal, entries max(raw_count_k[i], r_min) — Poisson noise with floor.

The right-hand panel has two sub-tabs:
  • "Process Covariance Q"  — heatmap of Q, updatable via "Preview Q" at any time.
  • "Inversion Results"     — N(Dp, t) + A·N̂(ch, t) heatmaps after a run.
"""

import numpy as np
import pandas as pd
import warnings

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QComboBox, QLabel, QFileDialog,
    QGroupBox, QDoubleSpinBox, QSpinBox, QCheckBox, QSplitter,
    QTextEdit, QMessageBox, QTabWidget,
    QFrame, QSizePolicy, QScrollArea,
)
from PyQt5.QtCore import Qt, QTimer

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from modules.data_model import AerosolDataset
from modules.inversion import (
    KalmanFilterInversion, KalmanSmootherInversion,
    GDEKalmanInversion, GDEKalmanSmootherInversion,
)


_METHODS = {
    "Kalman Filter":          KalmanFilterInversion,
    "Kalman Smoother (FIKS)": KalmanSmootherInversion,
    "GDE-EKF":                GDEKalmanInversion,
    "GDE-FIKS":               GDEKalmanSmootherInversion,
}

_GDE_METHODS = {"GDE-EKF", "GDE-FIKS"}


class KalmanTab(QWidget):
    def __init__(self, dataset: AerosolDataset, parent=None):
        super().__init__(parent)
        self.dataset      = dataset
        self.main_window  = parent
        self.last_result  = None
        self._last_std    = None    # sqrt(diag(P_fil or P_smo)), shape (n_scans, n_dp)
        self._last_A      = None
        self._last_counts = None
        self._scan_offset = 0       # first absolute scan index of the last run
        self._last_gde    = None    # GDEResult from a GDE run

        self._build_ui()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_status)
        self._timer.start(2000)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QHBoxLayout(self)

        # ── Left control panel ─────────────────────────────────────────
        left_w = QWidget()
        left   = QVBoxLayout(left_w)
        left.setSpacing(8)

        # Data status
        grp_data = QGroupBox("Data  (from Data tab)")
        dl = QVBoxLayout(grp_data)
        self.lbl_data = QLabel("—")
        self.lbl_data.setWordWrap(True)
        dl.addWidget(self.lbl_data)
        left.addWidget(grp_data)

        # Model status
        grp_model = QGroupBox("Measurement Model  (from Model tab)")
        ml = QVBoxLayout(grp_model)
        self.lbl_model = QLabel("—")
        self.lbl_model.setWordWrap(True)
        ml.addWidget(self.lbl_model)
        left.addWidget(grp_model)

        # Method selector
        grp_meth = QGroupBox("Method")
        mml = QVBoxLayout(grp_meth)
        self.cmb_method = QComboBox()
        self.cmb_method.addItems(list(_METHODS.keys()))
        self.cmb_method.currentTextChanged.connect(self._on_method_changed)
        mml.addWidget(self.cmb_method)
        left.addWidget(grp_meth)

        # Process covariance parameters
        grp_Q = QGroupBox("Evolution Model — Process Covariance Q")
        Ql = QGridLayout(grp_Q)

        self.spn_alpha   = self._dspin(-1.0,  1.0,   0.999, 3)
        self.spn_beta    = self._dspin(-1.0,  0.0,  -0.005, 3)
        self.spn_sig1    = self._dspin( 0.0,  1e12,  1.0, 2)
        self.spn_sig2    = self._dspin( 0.0,  1e12,  1.0, 2)
        self.spn_sig12   = self._dspin(-1e12, 1e12,  0.0, 2)
        self.spn_var_val = self._dspin( 1e-6, 1e15,  1.0, 2)

        _q_rows = [
            ("α  (AR2 size-corr.):",   self.spn_alpha),
            ("β  (AR2 size-corr.):",   self.spn_beta),
            ("σ₁  (init. var [0]):",   self.spn_sig1),
            ("σ₂  (init. var [1]):",   self.spn_sig2),
            ("σ₁₂  (init. cov):",      self.spn_sig12),
            ("Variance / bin:",         self.spn_var_val),
        ]
        for row, (lbl, w) in enumerate(_q_rows):
            Ql.addWidget(QLabel(lbl), row, 0)
            Ql.addWidget(w, row, 1)

        self.btn_preview_Q = QPushButton("Preview Q matrix")
        self.btn_preview_Q.clicked.connect(self._preview_Q)
        Ql.addWidget(self.btn_preview_Q, len(_q_rows), 0, 1, 2)
        left.addWidget(grp_Q)

        # Initial state covariance P₀
        grp_P0 = QGroupBox("Initial State Covariance P₀ = p₀ · I")
        P0l = QGridLayout(grp_P0)
        P0l.addWidget(QLabel("P₀ diagonal value:"), 0, 0)
        self.spn_p0 = self._dspin(1e-6, 1e15, 100.0, 4)
        P0l.addWidget(self.spn_p0, 0, 1)
        left.addWidget(grp_P0)

        # Measurement noise floor
        grp_R = QGroupBox("Measurement Noise")
        Rl = QGridLayout(grp_R)
        Rl.addWidget(QLabel("Min. count (R floor):"), 0, 0)
        self.spn_r_min = self._dspin(1e-6, 1e9, 2.0, 4)
        Rl.addWidget(self.spn_r_min, 0, 1)
        left.addWidget(grp_R)

        # Time range selection
        grp_time = QGroupBox("Time Range Selection")
        Tl = QGridLayout(grp_time)

        Tl.addWidget(QLabel("Start scan:"), 0, 0)
        self.spn_t_start = QSpinBox()
        self.spn_t_start.setMinimum(0)
        self.spn_t_start.setMaximum(0)
        Tl.addWidget(self.spn_t_start, 0, 1)
        self.lbl_t_start = QLabel("—")
        self.lbl_t_start.setStyleSheet("color: #a6adc8; font-size: 10px;")
        Tl.addWidget(self.lbl_t_start, 0, 2)

        Tl.addWidget(QLabel("End scan:"), 1, 0)
        self.spn_t_end = QSpinBox()
        self.spn_t_end.setMinimum(0)
        self.spn_t_end.setMaximum(0)
        Tl.addWidget(self.spn_t_end, 1, 1)
        self.lbl_t_end = QLabel("—")
        self.lbl_t_end.setStyleSheet("color: #a6adc8; font-size: 10px;")
        Tl.addWidget(self.lbl_t_end, 1, 2)

        btn_reset_range = QPushButton("Reset to full range")
        btn_reset_range.clicked.connect(self._reset_time_range)
        Tl.addWidget(btn_reset_range, 2, 0, 1, 3)

        self.spn_t_start.valueChanged.connect(self._update_time_labels)
        self.spn_t_end.valueChanged.connect(self._update_time_labels)
        left.addWidget(grp_time)

        # Spatial regularization via D₂ augmentation
        grp_reg = QGroupBox("Spatial Regularization  (D₂ augmentation)")
        Regl = QGridLayout(grp_reg)
        self.chk_reg = QCheckBox("Augment H with 2nd-order difference operator")
        self.chk_reg.setChecked(False)
        Regl.addWidget(self.chk_reg, 0, 0, 1, 2)
        Regl.addWidget(QLabel("Regularization variance:"), 1, 0)
        self.spn_reg_var = self._dspin(1e-12, 1e15, 1.0, 4)
        Regl.addWidget(self.spn_reg_var, 1, 1)
        left.addWidget(grp_reg)

        # GDE parameters (shown only when a GDE method is selected)
        grp_gde = QGroupBox("GDE Parameters")
        self.grp_gde = grp_gde
        Gl = QGridLayout(grp_gde)

        def _grow(lo, hi, val, dec):
            return self._dspin(lo, hi, val, dec)

        Gl.addWidget(QLabel("Scan duration (s):"),  0, 0)
        self.spn_dt_scan = _grow(0.1, 1e6,   60.0, 2);  Gl.addWidget(self.spn_dt_scan, 0, 1)
        Gl.addWidget(QLabel("Temperature (K):"),     1, 0)
        self.spn_T  = _grow(200.0, 400.0, 293.15, 2);  Gl.addWidget(self.spn_T, 1, 1)
        Gl.addWidget(QLabel("Pressure (Pa):"),        2, 0)
        self.spn_P  = _grow(1e3,  2e5, 1.013e5, 1); Gl.addWidget(self.spn_P, 2, 1)
        Gl.addWidget(QLabel("Density (kg/m³):"),      3, 0)
        self.spn_rho = _grow(100, 3000, 1400.0, 1); Gl.addWidget(self.spn_rho, 3, 1)

        Gl.addWidget(QLabel("Mechanisms:"), 4, 0, 1, 2)
        self.chk_gde_con = QCheckBox("Condensation/growth");  self.chk_gde_con.setChecked(True)
        self.chk_gde_nuc = QCheckBox("Nucleation");            self.chk_gde_nuc.setChecked(True)
        self.chk_gde_los = QCheckBox("Wall loss");             self.chk_gde_los.setChecked(True)
        self.chk_gde_coa = QCheckBox("Coagulation");           self.chk_gde_coa.setChecked(False)
        Gl.addWidget(self.chk_gde_con, 5, 0, 1, 2)
        Gl.addWidget(self.chk_gde_nuc, 6, 0, 1, 2)
        Gl.addWidget(self.chk_gde_los, 7, 0, 1, 2)
        Gl.addWidget(self.chk_gde_coa, 8, 0, 1, 2)

        Gl.addWidget(QLabel("GR₀ (m/s):"),          9, 0)
        self.spn_GR0 = _grow(1e-13, 1e-6, 1e-9, 12); Gl.addWidget(self.spn_GR0, 9, 1)
        Gl.addWidget(QLabel("J₀ (#/(m³·s)):"),      10, 0)
        self.spn_J0  = _grow(1e-3, 1e15,  1e6, 2);  Gl.addWidget(self.spn_J0, 10, 1)
        Gl.addWidget(QLabel("γ₀ (1/s):"),           11, 0)
        self.spn_gamma0 = _grow(1e-6, 10.0, 1e-3, 6); Gl.addWidget(self.spn_gamma0, 11, 1)

        Gl.addWidget(QLabel("σ²_GR (log-GR var):"), 12, 0)
        self.spn_var_GR = _grow(1e-12, 1e6, 1e-4, 8); Gl.addWidget(self.spn_var_GR, 12, 1)
        Gl.addWidget(QLabel("σ²_J (log-J var):"),   13, 0)
        self.spn_var_J  = _grow(1e-12, 1e6, 1e-4, 8); Gl.addWidget(self.spn_var_J, 13, 1)
        Gl.addWidget(QLabel("σ²_ξ (log-ξ var):"),   14, 0)
        self.spn_var_xi = _grow(1e-12, 1e6, 1e-8, 8); Gl.addWidget(self.spn_var_xi, 14, 1)

        Gl.addWidget(QLabel("log ζ₀ (init. log-GR):"), 15, 0)
        self.spn_log_GR_init = _grow(-30, 30, 0.0, 3); Gl.addWidget(self.spn_log_GR_init, 15, 1)
        Gl.addWidget(QLabel("log j₀ (init. log-J):"),  16, 0)
        self.spn_log_J_init  = _grow(-30, 30, 0.0, 3); Gl.addWidget(self.spn_log_J_init, 16, 1)

        Gl.addWidget(QLabel("P₀ diag (GR):"),  17, 0)
        self.spn_p0_GR = _grow(1e-6, 1e9, 4.0, 3); Gl.addWidget(self.spn_p0_GR, 17, 1)
        Gl.addWidget(QLabel("P₀ diag (J):"),   18, 0)
        self.spn_p0_J  = _grow(1e-6, 1e9, 4.0, 3); Gl.addWidget(self.spn_p0_J, 18, 1)
        Gl.addWidget(QLabel("P₀ diag (ξ):"),   19, 0)
        self.spn_p0_xi = _grow(1e-6, 1e9, 4.0, 3); Gl.addWidget(self.spn_p0_xi, 19, 1)

        grp_gde.setVisible(False)
        left.addWidget(grp_gde)

        # Actions
        self.btn_run = QPushButton("Run Inversion")
        self.btn_run.clicked.connect(self._run_inversion)
        left.addWidget(self.btn_run)

        self.btn_save = QPushButton("Save result…")
        self.btn_save.clicked.connect(self._save_result)
        left.addWidget(self.btn_save)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(120)
        left.addWidget(self.log)
        left.addStretch()

        # ── Right panel: two plot sub-tabs ─────────────────────────────
        right_w     = QWidget()
        right_layout = QVBoxLayout(right_w)

        self.plot_tabs = QTabWidget()

        # Sub-tab 1: Process Covariance Q
        q_widget = QWidget()
        q_layout = QVBoxLayout(q_widget)
        self.fig_Q    = Figure(figsize=(6, 5), facecolor="#1e1e2e")
        self.canvas_Q = FigureCanvas(self.fig_Q)
        self.canvas_Q.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar_Q = NavigationToolbar(self.canvas_Q, self)
        q_layout.addWidget(self.toolbar_Q)
        q_layout.addWidget(self.canvas_Q)

        # Sub-tab 2: Inversion Results
        res_widget = QWidget()
        res_layout = QVBoxLayout(res_widget)
        self.fig_res    = Figure(figsize=(6, 7), facecolor="#1e1e2e")
        self.canvas_res = FigureCanvas(self.fig_res)
        self.canvas_res.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar_res = NavigationToolbar(self.canvas_res, self)
        res_layout.addWidget(self.toolbar_res)
        res_layout.addWidget(self.canvas_res)

        # Sub-tab 3: 1D Size Distribution at selected scan
        dist_widget = QWidget()
        dist_layout = QVBoxLayout(dist_widget)

        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Scan index:"))
        self.spn_scan_idx = QSpinBox()
        self.spn_scan_idx.setMinimum(0)
        self.spn_scan_idx.setMaximum(0)
        self.spn_scan_idx.setFixedWidth(80)
        ctrl_row.addWidget(self.spn_scan_idx)
        self.btn_plot_1d = QPushButton("Plot")
        self.btn_plot_1d.clicked.connect(self._plot_1d)
        ctrl_row.addWidget(self.btn_plot_1d)
        ctrl_row.addStretch()
        dist_layout.addLayout(ctrl_row)

        self.fig_1d    = Figure(figsize=(6, 4), facecolor="#1e1e2e")
        self.canvas_1d = FigureCanvas(self.fig_1d)
        self.canvas_1d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar_1d = NavigationToolbar(self.canvas_1d, self)
        dist_layout.addWidget(self.toolbar_1d)
        dist_layout.addWidget(self.canvas_1d)

        # Sub-tab 4: GDE Results (4-panel)
        gde_widget = QWidget()
        gde_layout = QVBoxLayout(gde_widget)
        self.fig_gde    = Figure(figsize=(6, 9), facecolor="#1e1e2e")
        self.canvas_gde = FigureCanvas(self.fig_gde)
        self.canvas_gde.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar_gde = NavigationToolbar(self.canvas_gde, self)
        gde_layout.addWidget(self.toolbar_gde)
        gde_layout.addWidget(self.canvas_gde)

        self.plot_tabs.addTab(q_widget,    "Process Covariance Q")
        self.plot_tabs.addTab(res_widget,  "Inversion Results")
        self.plot_tabs.addTab(dist_widget, "1D Size Distribution")
        self.plot_tabs.addTab(gde_widget,  "GDE Results")
        right_layout.addWidget(self.plot_tabs)

        # Splitter
        scroll_left = QScrollArea()
        scroll_left.setWidget(left_w)
        scroll_left.setWidgetResizable(True)
        scroll_left.setFixedWidth(340)
        scroll_left.setFrameShape(QFrame.NoFrame)
        scroll_left.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(scroll_left)
        splitter.addWidget(right_w)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter)

        self._refresh_status()

    @staticmethod
    def _dspin(lo: float, hi: float, default: float,
               decimals: int) -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setDecimals(decimals)
        w.setValue(default)
        return w

    # ------------------------------------------------------------------
    # Status polling
    # ------------------------------------------------------------------

    def _refresh_status(self):
        if self.dataset.count_matrix is not None:
            dp = self.dataset.diameters
            n  = self.dataset.n_scans
            self.lbl_data.setText(
                f"✓  {n} scans  ×  {self.dataset.n_bins} bins\n"
                f"   Dp: {dp[0]:.1f} – {dp[-1]:.1f} nm"
            )
            self.lbl_data.setStyleSheet("color: #a6e3a1;")
            # Sync time-range spinboxes when the dataset changes
            if self.spn_t_end.maximum() != n - 1:
                self._set_range_spinboxes(0, n - 1)
        else:
            self.lbl_data.setText(
                "✗  No data loaded.\n   Load a file in the Data tab."
            )
            self.lbl_data.setStyleSheet("color: #f38ba8;")

        K, dp_grid = self._get_kernel()
        if K is not None:
            self.lbl_model.setText(
                f"✓  {K.shape[0]} channels  ×  {K.shape[1]} dp points\n"
                f"   Dp: {dp_grid[0]:.1f} – {dp_grid[-1]:.1f} nm"
            )
            self.lbl_model.setStyleSheet("color: #a6e3a1;")
        else:
            self.lbl_model.setText(
                "✗  No kernel.\n   Compute one in the Measurement Model tab."
            )
            self.lbl_model.setStyleSheet("color: #f38ba8;")

    # ------------------------------------------------------------------
    # Time range helpers
    # ------------------------------------------------------------------

    def _format_time(self, idx: int) -> str:
        """Human-readable label for scan at absolute index idx."""
        t = getattr(self.dataset, "times", None)
        if t is None or idx >= len(t):
            return f"scan {idx}"
        v = t[idx]
        try:
            ts = pd.Timestamp(v)
            return ts.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            try:
                return f"{float(v):.4g}"
            except Exception:
                return str(v)

    def _set_range_spinboxes(self, start: int, end: int):
        """Update both spinboxes and their labels without triggering loops."""
        n = self.dataset.n_scans if self.dataset.count_matrix is not None else 0
        for spn in (self.spn_t_start, self.spn_t_end):
            spn.blockSignals(True)
            spn.setMaximum(max(n - 1, 0))
        self.spn_t_start.setValue(start)
        self.spn_t_end.setValue(end)
        for spn in (self.spn_t_start, self.spn_t_end):
            spn.blockSignals(False)
        self._update_time_labels()

    def _update_time_labels(self):
        if self.dataset.count_matrix is None:
            self.lbl_t_start.setText("—")
            self.lbl_t_end.setText("—")
            return
        self.lbl_t_start.setText(self._format_time(self.spn_t_start.value()))
        self.lbl_t_end.setText(self._format_time(self.spn_t_end.value()))

    def _reset_time_range(self):
        if self.dataset.count_matrix is None:
            return
        self._set_range_spinboxes(0, self.dataset.n_scans - 1)

    # ------------------------------------------------------------------
    # Kernel helper
    # ------------------------------------------------------------------

    def _get_kernel(self):
        """Return (kernel, dp_grid) from the model tab, or (None, None)."""
        if self.main_window and hasattr(self.main_window, "tab_model"):
            m = self.main_window.tab_model.model
            if m is not None and m.kernel is not None:
                return m.kernel, m.dp_grid
        return None, None

    # ------------------------------------------------------------------
    # Method switching
    # ------------------------------------------------------------------

    def _on_method_changed(self, name: str):
        self.grp_gde.setVisible(name in _GDE_METHODS)

    # ------------------------------------------------------------------
    # Build method instance
    # ------------------------------------------------------------------

    def _make_method(self):
        cls = _METHODS[self.cmb_method.currentText()]
        return cls(
            alpha       = self.spn_alpha.value(),
            beta        = self.spn_beta.value(),
            sig1        = self.spn_sig1.value(),
            sig2        = self.spn_sig2.value(),
            sig12       = self.spn_sig12.value(),
            var_val     = self.spn_var_val.value(),
            r_min       = self.spn_r_min.value(),
            p0_diag_val = self.spn_p0.value(),
            use_reg     = self.chk_reg.isChecked(),
            reg_var     = self.spn_reg_var.value(),
        )

    def _make_gde_method(self):
        cls = _METHODS[self.cmb_method.currentText()]
        return cls(
            temperature      = self.spn_T.value(),
            pressure         = self.spn_P.value(),
            density          = self.spn_rho.value(),
            use_coagulation  = self.chk_gde_coa.isChecked(),
            use_condensation = self.chk_gde_con.isChecked(),
            use_nucleation   = self.chk_gde_nuc.isChecked(),
            use_wall_loss    = self.chk_gde_los.isChecked(),
            GR0              = self.spn_GR0.value(),
            J0               = self.spn_J0.value(),
            gamma0           = self.spn_gamma0.value(),
            t0               = self.spn_dt_scan.value(),
            x0               = 1e6,
            var_N            = self.spn_var_val.value(),
            var_GR           = self.spn_var_GR.value(),
            var_J            = self.spn_var_J.value(),
            var_xi           = self.spn_var_xi.value(),
            log_GR_init      = self.spn_log_GR_init.value(),
            log_J_init       = self.spn_log_J_init.value(),
            r_min            = self.spn_r_min.value(),
            p0_N             = self.spn_p0.value(),
            p0_GR            = self.spn_p0_GR.value(),
            p0_J             = self.spn_p0_J.value(),
            p0_xi            = self.spn_p0_xi.value(),
            dt_scan          = self.spn_dt_scan.value(),
        )

    # ------------------------------------------------------------------
    # Preview Q
    # ------------------------------------------------------------------

    def _preview_Q(self):
        """Build Q from current parameters and display it on the Q sub-tab."""
        A, dp_grid = self._get_kernel()
        if A is not None:
            n_dp = A.shape[1]
        else:
            n_dp = 50
            dp_grid = np.arange(n_dp, dtype=float)

        method = self._make_method()
        try:
            Q = method._build_Q(n_dp)
        except Exception as e:
            QMessageBox.critical(self, "Q build error", str(e))
            return

        self._plot_Q(Q, dp_grid)
        self.plot_tabs.setCurrentIndex(0)
        self._log(
            f"Q previewed  (n={n_dp},  diag ∈ "
            f"[{np.diag(Q).min():.3g}, {np.diag(Q).max():.3g}])"
        )

    def _plot_Q(self, Q: np.ndarray, dp_grid: np.ndarray):
        """Render Q as a symmetric heatmap with diameter axes."""
        self.fig_Q.clear()
        ax = self.fig_Q.add_subplot(111)
        ax.set_facecolor("#181825")
        ax.tick_params(colors="#cdd6f4")
        for spine in ax.spines.values():
            spine.set_edgecolor("#45475a")

        n = len(dp_grid)
        # Use imshow with manually placed diameter ticks
        vmax = np.abs(Q).max()
        im = ax.imshow(Q, aspect="auto", cmap="coolwarm",
                       vmin=-vmax, vmax=vmax, origin="upper",
                       interpolation="nearest")

        # Tick positions and labels in diameter space
        n_ticks = min(6, n)
        tick_idx = np.round(np.linspace(0, n - 1, n_ticks)).astype(int)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([f"{dp_grid[i]:.0f}" for i in tick_idx],
                           color="#cdd6f4", rotation=45, ha="right")
        ax.set_yticks(tick_idx)
        ax.set_yticklabels([f"{dp_grid[i]:.0f}" for i in tick_idx],
                           color="#cdd6f4")

        cb = self.fig_Q.colorbar(im, ax=ax)
        cb.ax.tick_params(colors="#cdd6f4")
        cb.set_label("Covariance", color="#cdd6f4")
        ax.set_xlabel("Dp (nm)", color="#cdd6f4")
        ax.set_ylabel("Dp (nm)", color="#cdd6f4")
        ax.set_title("Process covariance matrix Q", color="#cdd6f4")

        self.fig_Q.tight_layout()
        self.canvas_Q.draw()

    # ------------------------------------------------------------------
    # Run inversion
    # ------------------------------------------------------------------

    def _run_inversion(self):
        if self.dataset.count_matrix is None:
            QMessageBox.warning(
                self, "No data",
                "Load a data file in the Data tab first."
            )
            return

        A, dp_grid = self._get_kernel()
        if A is None:
            QMessageBox.warning(
                self, "No kernel",
                "Compute the measurement model kernel first."
            )
            return

        # Time range selection
        start = self.spn_t_start.value()
        end   = self.spn_t_end.value()
        if start > end:
            QMessageBox.warning(self, "Invalid range",
                                f"Start scan ({start}) must be ≤ end scan ({end}).")
            return
        full_matrix  = self.dataset.count_matrix[start:end + 1]
        count_matrix = self._align_matrix(full_matrix, A.shape[0])
        self._scan_offset = start
        range_note = (f"scans {start}–{end}"
                      if (start > 0 or end < self.dataset.n_scans - 1)
                      else f"all {count_matrix.shape[0]} scans")

        meth_name = self.cmb_method.currentText()

        # ── GDE path ──────────────────────────────────────────────────────────
        if meth_name in _GDE_METHODS:
            method = self._make_gde_method()
            try:
                gde = method.solve_series(A, count_matrix, dp_grid)
            except Exception as e:
                QMessageBox.critical(self, "GDE inversion error", str(e))
                return

            self._last_gde    = gde
            self.last_result  = gde.N_all
            self._last_std    = gde.std_N
            self._last_A      = A
            self._last_counts = count_matrix

            n_scans = gde.N_all.shape[0]
            self.spn_scan_idx.setMaximum(n_scans - 1)
            self.spn_scan_idx.setValue(0)

            self._plot_gde_results(dp_grid)
            self.plot_tabs.setCurrentIndex(3)   # GDE Results tab

            self._log(
                f"[{meth_name}]  {range_note}\n"
                f"  N ∈ [{gde.N_all.min():.3g}, {gde.N_all.max():.3g}]\n"
                f"  GR ∈ [{gde.GR_all.min():.3g}, {gde.GR_all.max():.3g}] m/s\n"
                f"  J ∈ [{gde.J_all.min():.3g}, {gde.J_all.max():.3g}] #/(m³·s)"
            )
            if self.main_window:
                self.main_window.set_status(
                    f"GDE-EKF inversion complete — {range_note}."
                )
            return

        # ── Identity Kalman path ──────────────────────────────────────────────
        method = self._make_method()
        try:
            ws            = method._run_series(A, count_matrix)
            N_ALL, std_ALL = method._extract_results(ws)
        except Exception as e:
            QMessageBox.critical(self, "Inversion error", str(e))
            return

        self.last_result  = N_ALL
        self._last_std    = std_ALL
        self._last_A      = A
        self._last_counts = count_matrix

        n_scans = N_ALL.shape[0]
        self.spn_scan_idx.setMaximum(n_scans - 1)
        self.spn_scan_idx.setValue(0)

        Q = method._build_Q(A.shape[1])
        self._plot_Q(Q, dp_grid)
        self._plot_results(dp_grid)
        self.plot_tabs.setCurrentIndex(1)

        reg_note = (f"  D₂ reg: σ²={self.spn_reg_var.value():.3g}"
                    if self.chk_reg.isChecked() else "  D₂ reg: off")
        self._log(
            f"[{meth_name}]  {range_note}\n"
            f"  N ∈ [{N_ALL.min():.3g}, {N_ALL.max():.3g}]\n"
            f"  std ∈ [{std_ALL.min():.3g}, {std_ALL.max():.3g}]\n"
            f"{reg_note}"
        )
        if self.main_window:
            self.main_window.set_status(
                f"Kalman inversion complete — {range_note}."
            )

    # ------------------------------------------------------------------
    # 1-D size distribution at a selected scan
    # ------------------------------------------------------------------

    def _plot_1d(self):
        if self.last_result is None:
            QMessageBox.warning(self, "No result", "Run inversion first.")
            return

        _, dp_grid = self._get_kernel()
        if dp_grid is None:
            dp_grid = np.arange(self.last_result.shape[1], dtype=float)

        t_idx = int(np.clip(self.spn_scan_idx.value(), 0, self.last_result.shape[0] - 1))
        N   = self.last_result[t_idx]                                      # (n_dp,)
        std = self._last_std[t_idx] if self._last_std is not None else None

        self.fig_1d.clear()
        ax = self.fig_1d.add_subplot(111)
        self._style_ax(ax)

        ax.plot(dp_grid, N, color="#89b4fa", linewidth=1.5, label="N̂(Dp)")
        if std is not None:
            ax.fill_between(dp_grid, np.maximum(N - std, 0.0), N + std,
                            alpha=0.3, color="#89b4fa", label="±1σ")

        abs_idx   = self._scan_offset + t_idx
        time_str  = self._format_time(abs_idx)

        ax.set_xscale("log")
        ax.set_xlabel("Dp (nm)", color="#cdd6f4")
        ax.set_ylabel("dN/dlogDp  (cm⁻³)", color="#cdd6f4")
        ax.set_title(
            f"Size distribution — scan {abs_idx}  ({time_str})"
            f"  [{self.cmb_method.currentText()}]",
            color="#cdd6f4"
        )
        leg = ax.legend(facecolor="#313244", edgecolor="#45475a")
        for txt in leg.get_texts():
            txt.set_color("#cdd6f4")

        self.fig_1d.tight_layout()
        self.canvas_1d.draw()

    # ------------------------------------------------------------------
    # Result plots
    # ------------------------------------------------------------------

    def _plot_gde_results(self, dp_grid: np.ndarray):
        """4-panel GDE result display: N heatmap, GR heatmap, J(t), xi(Dp) slices."""
        gde    = self._last_gde
        times  = np.arange(gde.N_all.shape[0]) + self._scan_offset
        dp_m   = dp_grid * 1e-9   # nm → m for GR scale context

        self.fig_gde.clear()
        ax1 = self.fig_gde.add_subplot(4, 1, 1)
        ax2 = self.fig_gde.add_subplot(4, 1, 2)
        ax3 = self.fig_gde.add_subplot(4, 1, 3)
        ax4 = self.fig_gde.add_subplot(4, 1, 4)
        for ax in (ax1, ax2, ax3, ax4):
            self._style_ax(ax)

        # ── N(Dp, t) ─────────────────────────────────────────────────────────
        im1 = ax1.pcolormesh(times, dp_grid, gde.N_all.T,
                              cmap="viridis", shading="auto")
        ax1.set_yscale("log")
        cb1 = self.fig_gde.colorbar(im1, ax=ax1)
        cb1.ax.tick_params(colors="#cdd6f4")
        cb1.set_label("dN/dlogDp", color="#cdd6f4")
        ax1.set_ylabel("Dp (nm)", color="#cdd6f4")
        ax1.set_title("Retrieved N(Dp, t)  [GDE-EKF]", color="#cdd6f4")

        # ── GR(Dp, t) in nm/h ────────────────────────────────────────────────
        GR_nmh = gde.GR_all * 1e9 * 3600.0   # m/s → nm/h
        im2 = ax2.pcolormesh(times, dp_grid, GR_nmh.T,
                              cmap="plasma", shading="auto")
        ax2.set_yscale("log")
        cb2 = self.fig_gde.colorbar(im2, ax=ax2)
        cb2.ax.tick_params(colors="#cdd6f4")
        cb2.set_label("GR (nm/h)", color="#cdd6f4")
        ax2.set_ylabel("Dp (nm)", color="#cdd6f4")
        ax2.set_title("Growth rate GR(Dp, t)", color="#cdd6f4")

        # ── J(t) nucleation rate ─────────────────────────────────────────────
        ax3.plot(times, gde.J_all, color="#a6e3a1", linewidth=1.2)
        ax3.fill_between(times, gde.J_all, alpha=0.2, color="#a6e3a1")
        ax3.set_ylabel("J (#/(m³·s))", color="#cdd6f4")
        ax3.set_title("Nucleation rate J(t)", color="#cdd6f4")
        ax3.set_yscale("log")

        # ── xi(Dp) wall loss — a few time slices ────────────────────────────
        n_slices = min(5, len(times))
        slice_idx = np.round(np.linspace(0, len(times) - 1, n_slices)).astype(int)
        colors = ["#89b4fa", "#f38ba8", "#a6e3a1", "#fab387", "#cba6f7"]
        for k, (si, col) in enumerate(zip(slice_idx, colors)):
            abs_idx = int(times[si])
            ax4.plot(dp_grid, gde.xi_all[si], color=col, linewidth=1.0,
                     label=f"scan {abs_idx}")
        ax4.set_xscale("log")
        ax4.set_yscale("log")
        ax4.set_xlabel("Dp (nm)", color="#cdd6f4")
        ax4.set_ylabel("ξ (1/s)", color="#cdd6f4")
        ax4.set_title("Wall-loss rate ξ(Dp) — time slices", color="#cdd6f4")
        leg4 = ax4.legend(framealpha=0.3, fontsize=8)
        leg4.get_frame().set_facecolor("#313244")
        leg4.get_frame().set_edgecolor("#45475a")
        for t in leg4.get_texts():
            t.set_color("#cdd6f4")

        self.fig_gde.tight_layout()
        self.canvas_gde.draw()

    def _style_ax(self, ax):
        ax.set_facecolor("#181825")
        ax.tick_params(colors="#cdd6f4")
        for spine in ax.spines.values():
            spine.set_edgecolor("#45475a")

    def _get_channel_diameters(self, n_channels: int):
        """Return (dp_array, is_diameter_nm). Prefers model tab diameters."""
        if self.main_window and hasattr(self.main_window, "tab_model"):
            m = self.main_window.tab_model.model
            if (m is not None and m.diameters is not None
                    and len(m.diameters) == n_channels):
                return m.diameters, True
        return np.arange(1, n_channels + 1), False

    def _plot_results(self, dp_grid: np.ndarray):
        N_ALL   = self.last_result      # (n_scans, n_dp)
        std_ALL = self._last_std        # (n_scans, n_dp)
        self.fig_res.clear()

        ax1 = self.fig_res.add_subplot(311)
        ax2 = self.fig_res.add_subplot(312)
        ax3 = self.fig_res.add_subplot(313)
        for ax in (ax1, ax2, ax3):
            self._style_ax(ax)

        times = np.arange(N_ALL.shape[0]) + self._scan_offset

        # ── Top: retrieved N(Dp, t) ────────────────────────────────────
        im1 = ax1.pcolormesh(times, dp_grid, N_ALL.T,
                              cmap="viridis", shading="auto")
        ax1.set_yscale("log")
        cb1 = self.fig_res.colorbar(im1, ax=ax1)
        cb1.ax.tick_params(colors="#cdd6f4")
        ax1.set_xlabel("Scan index", color="#cdd6f4")
        ax1.set_ylabel("Dp (nm)", color="#cdd6f4")
        ax1.set_title(
            f"Retrieved dN/dlogDp — {self.cmb_method.currentText()}",
            color="#cdd6f4"
        )

        # ── Middle: state uncertainty std(Dp, t) ──────────────────────
        if std_ALL is not None:
            im2 = ax2.pcolormesh(times, dp_grid, std_ALL.T,
                                  cmap="YlOrRd", shading="auto")
            ax2.set_yscale("log")
            cb2 = self.fig_res.colorbar(im2, ax=ax2)
            cb2.ax.tick_params(colors="#cdd6f4")
            ax2.set_xlabel("Scan index", color="#cdd6f4")
            ax2.set_ylabel("Dp (nm)", color="#cdd6f4")
            ax2.set_title(
                "State uncertainty  √diag(P)  (std dev)", color="#cdd6f4"
            )

        # ── Bottom: A·N̂(ch, t) ────────────────────────────────────────
        if self._last_A is not None:
            Y_ALL = self._last_A @ N_ALL.T          # (n_channels, n_scans)
            dp_ch, is_dp = self._get_channel_diameters(Y_ALL.shape[0])
            im3 = ax3.pcolormesh(times, dp_ch, Y_ALL,
                                  cmap="plasma", shading="auto")
            if is_dp:
                ax3.set_yscale("log")
            cb3 = self.fig_res.colorbar(im3, ax=ax3)
            cb3.ax.tick_params(colors="#cdd6f4")
            ax3.set_xlabel("Scan index", color="#cdd6f4")
            ax3.set_ylabel(
                "Selected Dp (nm)" if is_dp else "Channel index",
                color="#cdd6f4"
            )
            ax3.set_title(
                "A · N̂  (reconstructed measurements)", color="#cdd6f4"
            )

        self.fig_res.tight_layout()
        self.canvas_res.draw()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _align_matrix(count_matrix: np.ndarray, n_channels: int) -> np.ndarray:
        if count_matrix.shape[1] == n_channels:
            return count_matrix
        out   = np.zeros((count_matrix.shape[0], n_channels))
        x_old = np.linspace(0, 1, count_matrix.shape[1])
        x_new = np.linspace(0, 1, n_channels)
        for i, row in enumerate(count_matrix):
            out[i] = np.interp(x_new, x_old, row)
        return out

    def _save_result(self):
        if self.last_result is None:
            QMessageBox.warning(self, "Save", "Run inversion first.")
            return
        _, dp_grid = self._get_kernel()
        if dp_grid is None:
            dp_grid = np.arange(self.last_result.shape[1], dtype=float)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save result", "kalman_inversion.csv", "CSV (*.csv)"
        )
        if not path:
            return
        df = pd.DataFrame(
            self.last_result,
            columns=[f"Dp_{d:.1f}nm" for d in dp_grid]
        )
        df.to_csv(path, index=False)
        self._log(f"Saved → {path}")

    def _log(self, msg: str):
        self.log.append(msg)

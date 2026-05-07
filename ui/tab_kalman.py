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
from modules.inversion import KalmanFilterInversion, KalmanSmootherInversion


_METHODS = {
    "Kalman Filter":          KalmanFilterInversion,
    "Kalman Smoother (FIKS)": KalmanSmootherInversion,
}


class KalmanTab(QWidget):
    def __init__(self, dataset: AerosolDataset, parent=None):
        super().__init__(parent)
        self.dataset      = dataset
        self.main_window  = parent
        self.last_result  = None
        self._last_std    = None    # sqrt(diag(P_fil or P_smo)), shape (n_scans, n_dp)
        self._last_A      = None
        self._last_counts = None

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

        self.plot_tabs.addTab(q_widget,    "Process Covariance Q")
        self.plot_tabs.addTab(res_widget,  "Inversion Results")
        self.plot_tabs.addTab(dist_widget, "1D Size Distribution")
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
            self.lbl_data.setText(
                f"✓  {self.dataset.n_scans} scans  ×  {self.dataset.n_bins} bins\n"
                f"   Dp: {dp[0]:.1f} – {dp[-1]:.1f} nm"
            )
            self.lbl_data.setStyleSheet("color: #a6e3a1;")
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

        method       = self._make_method()
        count_matrix = self._align_matrix(self.dataset.count_matrix, A.shape[0])

        try:
            ws            = method._run_series(A, count_matrix)
            N_ALL, std_ALL = method._extract_results(ws)  # (n_scans, n_dp) each
        except Exception as e:
            QMessageBox.critical(self, "Inversion error", str(e))
            return

        self.last_result  = N_ALL
        self._last_std    = std_ALL
        self._last_A      = A
        self._last_counts = count_matrix

        # Keep scan spinbox in range
        n_scans = N_ALL.shape[0]
        self.spn_scan_idx.setMaximum(n_scans - 1)
        self.spn_scan_idx.setValue(0)

        # Refresh Q plot with the actual Q used
        Q = method._build_Q(A.shape[1])
        self._plot_Q(Q, dp_grid)

        # Render results and switch to that sub-tab
        self._plot_results(dp_grid)
        self.plot_tabs.setCurrentIndex(1)

        reg_note = (f"  D₂ reg: σ²={self.spn_reg_var.value():.3g}"
                    if self.chk_reg.isChecked() else "  D₂ reg: off")
        self._log(
            f"[{self.cmb_method.currentText()}]  {n_scans} scans\n"
            f"  N ∈ [{N_ALL.min():.3g}, {N_ALL.max():.3g}]\n"
            f"  std ∈ [{std_ALL.min():.3g}, {std_ALL.max():.3g}]\n"
            f"{reg_note}"
        )
        if self.main_window:
            self.main_window.set_status(
                f"Kalman inversion complete — {n_scans} scans."
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

        ax.set_xscale("log")
        ax.set_xlabel("Dp (nm)", color="#cdd6f4")
        ax.set_ylabel("dN/dlogDp  (cm⁻³)", color="#cdd6f4")
        ax.set_title(
            f"Size distribution — scan {t_idx}"
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

        times = np.arange(N_ALL.shape[0])

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

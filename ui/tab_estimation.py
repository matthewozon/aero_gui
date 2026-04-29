"""
ui/tab_estimation.py
--------------------
Tab 4: Parameter estimation using the pure-Python FIKS implementation
in modules/algo/kalman.py (translated from BAYROSOL EKF.jl).

No Julia runtime, no juliacall, no external bridge required.

Workflow
--------
1.  User configures the state-space dimensions and noise covariances.
2.  Optionally uses modules/algo/stochproc to build a size-correlated
    process noise covariance Q (encodes spatial correlation of aerosol
    parameters across diameter bins).
3.  Clicks Run → kalman_filter_smoother() runs in a background thread.
4.  Smoothed trajectory, uncertainty bands, and filtered-vs-smoothed
    comparison are plotted.
5.  Results can be saved as .npz or .csv.
"""

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QComboBox, QLabel, QFileDialog,
    QGroupBox, QDoubleSpinBox, QSpinBox, QSplitter,
    QTextEdit, QMessageBox, QCheckBox, QTabWidget,
    QProgressBar, QSizePolicy, QScrollArea, QFrame,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from modules.data_model import AerosolDataset
from modules.algo.kalman import (
    kalman_filter_smoother, make_linear_callbacks,
    filtered_states, smoothed_states, filtered_stds, smoothed_stds,
)
from modules.algo.stochproc import space_covariance_chol


# ============================================================
#  Background worker — keeps GUI responsive during FIKS
# ============================================================

class FIKSWorker(QThread):
    finished = pyqtSignal(object)   # KFWorkspace
    error    = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, Y, x0, P0_diag, Q, R, t_samp, callbacks):
        super().__init__()
        self.Y        = Y
        self.x0       = x0
        self.P0_diag  = P0_diag
        self.Q        = Q
        self.R        = R
        self.t_samp   = t_samp
        self.callbacks = callbacks

    def run(self):
        try:
            self.progress.emit("Running FIKS smoother…")
            ws = kalman_filter_smoother(
                x0        = self.x0,
                P0_diag   = self.P0_diag,
                Q         = self.Q,
                R         = self.R,
                Y         = self.Y,
                t_samp    = self.t_samp,
                t0        = 1.0,
                **self.callbacks,
            )
            self.finished.emit(ws)
        except Exception as e:
            self.error.emit(str(e))


# ============================================================
#  Tab widget
# ============================================================

class EstimationTab(QWidget):
    def __init__(self, dataset: AerosolDataset, parent=None):
        super().__init__(parent)
        self.dataset     = dataset
        self.main_window = parent
        self.result      = None   # KFWorkspace
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QHBoxLayout(self)

        left_w = QWidget()
        left   = QVBoxLayout(left_w); left.setSpacing(8)

        # ── Method selector ─────────────────────────────────────────
        grp_method = QGroupBox("Estimation Method")
        ml = QVBoxLayout(grp_method)
        self.cmb_method = QComboBox()
        self.cmb_method.addItems([
            "FIKS – Linear KF  (identity F, H)",
            "FIKS – Linear KF  (scalar persistence model)",
        ])
        ml.addWidget(self.cmb_method)
        left.addWidget(grp_method)

        # ── State space ─────────────────────────────────────────────
        grp_dims = QGroupBox("State Space Dimensions")
        dl = QGridLayout(grp_dims)
        dl.addWidget(QLabel("State dim  n_x:"), 0, 0)
        self.spn_n_state = QSpinBox()
        self.spn_n_state.setRange(1, 2048); self.spn_n_state.setValue(64)
        dl.addWidget(self.spn_n_state, 0, 1)
        dl.addWidget(QLabel("Obs dim  n_y:"), 1, 0)
        self.spn_n_obs = QSpinBox()
        self.spn_n_obs.setRange(1, 2048); self.spn_n_obs.setValue(64)
        dl.addWidget(self.spn_n_obs, 1, 1)
        left.addWidget(grp_dims)

        # ── Noise covariance ─────────────────────────────────────────
        grp_noise = QGroupBox("Noise Covariance (diagonal σ²)")
        nl = QGridLayout(grp_noise)
        for row, (lbl, attr, val) in enumerate([
            ("Process noise Q:", "spn_Q",  1e-2),
            ("Obs noise R:",     "spn_R",  1.0),
            ("Initial cov P0:", "spn_P0", 1.0),
        ]):
            nl.addWidget(QLabel(lbl), row, 0)
            spn = QDoubleSpinBox()
            spn.setRange(1e-12, 1e6); spn.setValue(val); spn.setDecimals(8)
            nl.addWidget(spn, row, 1)
            setattr(self, attr, spn)
        left.addWidget(grp_noise)

        # ── Spatial covariance (StochProc) ────────────────────────
        grp_sp = QGroupBox("Spatial Covariance of Q (optional)")
        sp = QGridLayout(grp_sp)
        sp.addWidget(QLabel("Use size-correlated Q:"), 0, 0)
        self.chk_spatial = QCheckBox()
        sp.addWidget(self.chk_spatial, 0, 1)
        sp.addWidget(QLabel("AR filter pole r:"), 1, 0)
        self.spn_rpol = QDoubleSpinBox()
        self.spn_rpol.setRange(0.01, 0.9999); self.spn_rpol.setValue(0.85)
        self.spn_rpol.setDecimals(4)
        sp.addWidget(self.spn_rpol, 1, 1)
        sp.addWidget(QLabel("(builds Toeplitz Q via StochProc)"), 2, 0, 1, 2)
        left.addWidget(grp_sp)

        # ── Persistence model scalar ──────────────────────────────
        grp_persist = QGroupBox("Persistence Model")
        pl = QGridLayout(grp_persist)
        pl.addWidget(QLabel("Time constant r (0–1):"), 0, 0)
        self.spn_r_persist = QDoubleSpinBox()
        self.spn_r_persist.setRange(0.0, 1.0); self.spn_r_persist.setValue(0.95)
        self.spn_r_persist.setDecimals(4)
        pl.addWidget(self.spn_r_persist, 0, 1)
        pl.addWidget(QLabel("(F = r·I,  Q corrected for stationarity)"), 1, 0, 1, 2)
        left.addWidget(grp_persist)

        # ── Scan range ───────────────────────────────────────────────
        grp_scans = QGroupBox("Data: Scan Range")
        sc = QGridLayout(grp_scans)
        sc.addWidget(QLabel("First scan:"), 0, 0)
        self.spn_scan_start = QSpinBox(); self.spn_scan_start.setRange(0, 99999)
        sc.addWidget(self.spn_scan_start, 0, 1)
        sc.addWidget(QLabel("Last scan (−1 = all):"), 1, 0)
        self.spn_scan_end = QSpinBox()
        self.spn_scan_end.setRange(-1, 99999); self.spn_scan_end.setValue(-1)
        sc.addWidget(self.spn_scan_end, 1, 1)
        left.addWidget(grp_scans)

        # ── Actions ──────────────────────────────────────────────────
        self.btn_run = QPushButton("▶  Run FIKS Estimation")
        self.btn_run.clicked.connect(self._run)
        left.addWidget(self.btn_run)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        left.addWidget(self.progress_bar)

        self.btn_save = QPushButton("Save results…")
        self.btn_save.clicked.connect(self._save_results)
        left.addWidget(self.btn_save)

        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.log.setMaximumHeight(160)
        left.addWidget(self.log)
        left.addStretch()

        # ── Right: plot tabs ─────────────────────────────────────────
        right_w = QWidget()
        right   = QVBoxLayout(right_w)
        self.plot_tabs = QTabWidget()

        for title, attr_fig, attr_canvas, attr_tb in [
            ("Smoothed State",        "fig_heat", "canvas_heat", "tb_heat"),
            ("Uncertainty (std dev)", "fig_unc",  "canvas_unc",  "tb_unc"),
            ("Filtered vs Smoothed",  "fig_comp", "canvas_comp", "tb_comp"),
        ]:
            fig    = Figure(figsize=(9, 4), facecolor="#1e1e2e")
            canvas = FigureCanvas(fig)
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            canvas.setMinimumSize(100, 100)
            tb     = NavigationToolbar(canvas, self)
            w = QWidget(); lyt = QVBoxLayout(w)
            lyt.addWidget(tb); lyt.addWidget(canvas)
            self.plot_tabs.addTab(w, title)
            setattr(self, attr_fig, fig)
            setattr(self, attr_canvas, canvas)
            setattr(self, attr_tb, tb)

        right.addWidget(self.plot_tabs)

        scroll_left = QScrollArea()
        scroll_left.setWidget(left_w)
        scroll_left.setWidgetResizable(True)
        scroll_left.setFixedWidth(336)
        scroll_left.setFrameShape(QFrame.NoFrame)
        scroll_left.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(scroll_left)
        splitter.addWidget(right_w)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter)

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _get_observations(self) -> np.ndarray:
        """Return count matrix slice, shape (n_y, n_t) — time axis last."""
        if self.dataset.count_matrix is None:
            raise RuntimeError("No data — load data and assign columns first.")
        C     = self.dataset.count_matrix   # (n_t, n_bins)
        start = self.spn_scan_start.value()
        end   = self.spn_scan_end.value()
        C_slice = C[start:, :] if end == -1 else C[start:end + 1, :]
        return C_slice.T.copy()   # → (n_bins, n_t)

    def _resize_obs(self, Y: np.ndarray, n_y: int) -> np.ndarray:
        """Interpolate observation rows to match declared n_y if needed."""
        if Y.shape[0] == n_y:
            return Y
        x_old = np.linspace(0, 1, Y.shape[0])
        x_new = np.linspace(0, 1, n_y)
        return np.array([np.interp(x_new, x_old, Y[:, t])
                         for t in range(Y.shape[1])]).T

    def _build_model_matrices(self, n_x: int, n_y: int):
        """Return (x0, P0_diag, Q, R, callbacks) based on current settings."""
        q_var  = self.spn_Q.value()
        r_var  = self.spn_R.value()
        p0_var = self.spn_P0.value()

        x0       = np.zeros(n_x)
        P0_diag  = np.full(n_x, p0_var)
        R        = np.eye(n_y) * r_var

        # ── process noise Q ─────────────────────────────────────────
        if self.chk_spatial.isChecked() and n_x > 1:
            r_pol    = self.spn_rpol.value()
            D_tilde  = np.full(n_x, q_var)
            method   = self.cmb_method.currentIndex()
            if method == 1:
                B = self.spn_r_persist.value() * np.eye(n_x)
                _, Q = space_covariance_chol(r_pol, D_tilde, B)
            else:
                _, Q = space_covariance_chol(r_pol, D_tilde)
            # Ensure PSD after the source-noise subtraction
            Q = 0.5 * (Q + Q.T)
            Q = np.maximum(Q, 0)   # clip tiny negatives
        else:
            Q = np.eye(n_x) * q_var

        # ── state-transition model (F) and observation (H) ──────────
        method = self.cmb_method.currentIndex()

        if method == 1:
            # Persistence: F = r·I
            r  = self.spn_r_persist.value()
            F  = r * np.eye(n_x)
        else:
            # Identity: F = I  (random walk)
            F  = np.eye(n_x)

        H = (np.eye(n_y) if n_y == n_x
             else np.hstack([np.eye(n_y), np.zeros((n_y, n_x - n_y))]))

        callbacks = make_linear_callbacks(F, H)
        return x0, P0_diag, Q, R, callbacks

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _run(self):
        try:
            Y = self._get_observations()
        except RuntimeError as e:
            QMessageBox.warning(self, "No data", str(e)); return

        n_x = self.spn_n_state.value()
        n_y = self.spn_n_obs.value()

        Y = self._resize_obs(Y, n_y)
        n_t = Y.shape[1]
        t_samp = np.arange(n_t, dtype=float)

        try:
            x0, P0_diag, Q, R, cbs = self._build_model_matrices(n_x, n_y)
        except Exception as e:
            QMessageBox.critical(self, "Model error", str(e)); return

        self.btn_run.setEnabled(False)
        self.progress_bar.show()
        self._log(f"FIKS: {n_t} steps  n_x={n_x}  n_y={n_y}  "
                  f"spatial_Q={self.chk_spatial.isChecked()}")

        self._worker = FIKSWorker(Y, x0, P0_diag, Q, R, t_samp, cbs)
        self._worker.finished.connect(self._on_fiks_done)
        self._worker.error.connect(self._on_fiks_error)
        self._worker.progress.connect(self._log)
        self._worker.start()

    def _on_fiks_done(self, ws):
        self.result = ws
        self.btn_run.setEnabled(True)
        self.progress_bar.hide()
        self._log(
            f"Done.  log-likelihood = {ws.log_likelihood:.4f}\n"
            f"Smoothed shape: {smoothed_states(ws).shape}"
        )
        self._plot_results(ws)
        if self.main_window:
            self.main_window.set_status("FIKS estimation complete.")

    def _on_fiks_error(self, msg: str):
        self.btn_run.setEnabled(True)
        self.progress_bar.hide()
        self._log(f"ERROR: {msg}")
        QMessageBox.critical(self, "FIKS error", msg)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _ax_style(self, ax):
        ax.set_facecolor("#181825")
        ax.tick_params(colors="#cdd6f4")
        for s in ax.spines.values():
            s.set_edgecolor("#45475a")

    def _colorbar(self, fig, im, ax, label=""):
        cb = fig.colorbar(im, ax=ax)
        cb.ax.tick_params(colors="#cdd6f4")
        if label:
            cb.set_label(label, color="#cdd6f4")

    def _plot_results(self, ws):
        x_smo  = smoothed_states(ws)    # (n_t, n_x)
        x_fil  = filtered_states(ws)
        std_smo = smoothed_stds(ws)     # (n_t, n_x)
        times   = np.arange(x_smo.shape[0])

        # ── Heatmap: smoothed state ──────────────────────────────────
        self.fig_heat.clear()
        ax = self.fig_heat.add_subplot(111); self._ax_style(ax)
        im = ax.pcolormesh(times, np.arange(x_smo.shape[1]),
                           x_smo.T, cmap="viridis", shading="auto")
        self._colorbar(self.fig_heat, im, ax)
        ax.set_xlabel("Time step", color="#cdd6f4")
        ax.set_ylabel("State index", color="#cdd6f4")
        ax.set_title("Smoothed state trajectory", color="#cdd6f4")
        self.fig_heat.tight_layout(); self.canvas_heat.draw()

        # ── Uncertainty: posterior std dev ───────────────────────────
        self.fig_unc.clear()
        ax2 = self.fig_unc.add_subplot(111); self._ax_style(ax2)
        im2 = ax2.pcolormesh(times, np.arange(std_smo.shape[1]),
                             std_smo.T, cmap="magma", shading="auto")
        self._colorbar(self.fig_unc, im2, ax2, "Std dev")
        ax2.set_xlabel("Time step", color="#cdd6f4")
        ax2.set_ylabel("State index", color="#cdd6f4")
        ax2.set_title("Posterior std dev of smoothed state", color="#cdd6f4")
        self.fig_unc.tight_layout(); self.canvas_unc.draw()

        # ── Filtered vs smoothed (middle state component) ────────────
        mid = x_smo.shape[1] // 2
        self.fig_comp.clear()
        ax3 = self.fig_comp.add_subplot(111); self._ax_style(ax3)
        ax3.plot(times, x_fil[:, mid], color="#89b4fa", lw=1.5, label="Filtered")
        ax3.plot(times, x_smo[:, mid], color="#a6e3a1", lw=2,   label="Smoothed")
        ax3.fill_between(times,
                         x_smo[:, mid] - 2 * std_smo[:, mid],
                         x_smo[:, mid] + 2 * std_smo[:, mid],
                         alpha=0.25, color="#a6e3a1", label="±2σ")
        ax3.set_xlabel("Time step", color="#cdd6f4")
        ax3.set_ylabel(f"State [{mid}]", color="#cdd6f4")
        ax3.set_title("Filtered vs Smoothed (middle component)", color="#cdd6f4")
        ax3.legend(facecolor="#313244", edgecolor="#45475a", labelcolor="#cdd6f4")
        self.fig_comp.tight_layout(); self.canvas_comp.draw()

        self.plot_tabs.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save_results(self):
        if self.result is None:
            QMessageBox.warning(self, "Save", "Run estimation first."); return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save FIKS results", "fiks_result.npz",
            "NumPy archive (*.npz);;CSV (*.csv)")
        if not path: return

        ws = self.result
        if path.endswith(".npz"):
            np.savez(path,
                     x_smoothed = smoothed_states(ws),
                     x_filtered = filtered_states(ws),
                     std_smoothed = smoothed_stds(ws),
                     std_filtered = filtered_stds(ws),
                     log_likelihood = np.array([ws.log_likelihood]))
        else:
            import pandas as pd
            x_smo = smoothed_states(ws)
            pd.DataFrame(x_smo,
                         columns=[f"x_{i}" for i in range(x_smo.shape[1])]
                         ).to_csv(path, index=False)
        self._log(f"Saved → {path}")

    def _log(self, msg: str):
        self.log.append(msg)

"""
ui/tab_inversion.py
-------------------
Tab 3: Data inversion — recover N(Dp) from raw counts.
"""

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QComboBox, QLabel, QFileDialog,
    QGroupBox, QDoubleSpinBox, QSpinBox, QSplitter,
    QTextEdit, QMessageBox, QCheckBox, QTabWidget
)
from PyQt5.QtCore import Qt

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from modules.data_model import AerosolDataset
from modules.inversion import INVERSION_METHODS, TikhonovInversion


class InversionTab(QWidget):
    def __init__(self, dataset: AerosolDataset, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.main_window = parent
        self.last_result = None
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)

        # ---- Left ----
        left_w = QWidget(); left_w.setFixedWidth(300)
        left = QVBoxLayout(left_w); left.setSpacing(8)

        grp_method = QGroupBox("Inversion Method")
        ml = QVBoxLayout(grp_method)
        self.cmb_method = QComboBox()
        self.cmb_method.addItems(list(INVERSION_METHODS.keys()))
        self.cmb_method.currentTextChanged.connect(self._update_param_ui)
        ml.addWidget(self.cmb_method)
        left.addWidget(grp_method)

        # Method-specific parameters
        self.grp_params = QGroupBox("Method Parameters")
        self.param_layout = QGridLayout(self.grp_params)
        left.addWidget(self.grp_params)

        # Scan selection
        grp_scan = QGroupBox("Scan Selection")
        sl = QGridLayout(grp_scan)
        sl.addWidget(QLabel("Scan index:"), 0, 0)
        self.spn_scan_idx = QSpinBox()
        self.spn_scan_idx.setRange(0, 9999)
        sl.addWidget(self.spn_scan_idx, 0, 1)
        self.chk_all_scans = QCheckBox("Invert all scans")
        sl.addWidget(self.chk_all_scans, 1, 0, 1, 2)
        left.addWidget(grp_scan)

        # Kernel source
        grp_kernel = QGroupBox("Kernel")
        kl = QVBoxLayout(grp_kernel)
        self.lbl_kernel = QLabel("No kernel loaded.\nCompute one in the Model tab or load CSV.")
        self.lbl_kernel.setWordWrap(True)
        kl.addWidget(self.lbl_kernel)
        self.btn_load_kernel = QPushButton("Load kernel CSV…")
        self.btn_load_kernel.clicked.connect(self._load_kernel)
        kl.addWidget(self.btn_load_kernel)
        left.addWidget(grp_kernel)

        self.btn_run = QPushButton("Run Inversion")
        self.btn_run.clicked.connect(self._run_inversion)
        left.addWidget(self.btn_run)

        self.btn_save_result = QPushButton("Save result…")
        self.btn_save_result.clicked.connect(self._save_result)
        left.addWidget(self.btn_save_result)

        # L-curve (Tikhonov only)
        self.btn_lcurve = QPushButton("L-curve (Tikhonov)")
        self.btn_lcurve.clicked.connect(self._plot_lcurve)
        left.addWidget(self.btn_lcurve)

        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(120)
        left.addWidget(self.log)
        left.addStretch()

        # ---- Right ----
        right_w = QWidget()
        right = QVBoxLayout(right_w)

        self.fig = Figure(figsize=(9, 6), facecolor="#1e1e2e")
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        right.addWidget(self.toolbar)
        right.addWidget(self.canvas)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_w)
        splitter.addWidget(right_w)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter)

        self._update_param_ui()

        # Internal kernel storage (shared from model tab via parent)
        self._kernel = None
        self._dp_grid = None

    # ------------------------------------------------------------------

    def _update_param_ui(self):
        """Rebuild the parameter widget area for the selected method."""
        # Clear existing widgets
        while self.param_layout.count():
            item = self.param_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        method_name = self.cmb_method.currentText()

        if "Tikhonov" in method_name:
            self.param_layout.addWidget(QLabel("λ (regularisation):"), 0, 0)
            self.spn_lambda = QDoubleSpinBox()
            self.spn_lambda.setRange(1e-10, 1e6)
            self.spn_lambda.setDecimals(8)
            self.spn_lambda.setValue(1e-3)
            self.param_layout.addWidget(self.spn_lambda, 0, 1)
            self.param_layout.addWidget(QLabel("Order (0/1/2):"), 1, 0)
            self.spn_order = QSpinBox(); self.spn_order.setRange(0, 2)
            self.param_layout.addWidget(self.spn_order, 1, 1)

        elif "SVD" in method_name:
            self.param_layout.addWidget(QLabel("# singular values k:"), 0, 0)
            self.spn_k = QSpinBox(); self.spn_k.setRange(1, 500); self.spn_k.setValue(20)
            self.param_layout.addWidget(self.spn_k, 0, 1)

        elif "EM" in method_name:
            self.param_layout.addWidget(QLabel("Max iterations:"), 0, 0)
            self.spn_niter = QSpinBox(); self.spn_niter.setRange(10, 5000); self.spn_niter.setValue(100)
            self.param_layout.addWidget(self.spn_niter, 0, 1)

    def _get_method_instance(self):
        name = self.cmb_method.currentText()
        cls = INVERSION_METHODS[name]
        if "Tikhonov" in name:
            return cls(lambda_reg=self.spn_lambda.value(),
                       order=self.spn_order.value())
        elif "SVD" in name:
            return cls(k=self.spn_k.value())
        elif "EM" in name:
            return cls(n_iter=self.spn_niter.value())
        return cls()

    def _get_kernel(self):
        """Try to get kernel from Model tab first, else from loaded file."""
        if self.main_window and hasattr(self.main_window, "tab_model"):
            mdl = self.main_window.tab_model.model
            if mdl is not None and mdl.kernel is not None:
                return mdl.kernel, mdl.dp_grid
        if self._kernel is not None:
            return self._kernel, self._dp_grid
        return None, None

    def _load_kernel(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load kernel CSV", "", "CSV (*.csv)")
        if not path:
            return
        import pandas as pd
        try:
            df = pd.read_csv(path, index_col=0)
            self._kernel = df.values
            self._dp_grid = df.columns.to_numpy(dtype=float)
            self.lbl_kernel.setText(
                f"Loaded: {path.split('/')[-1]}\n"
                f"Shape: {self._kernel.shape}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))

    def _run_inversion(self):
        if self.dataset.count_matrix is None:
            QMessageBox.warning(self, "No data", "Load data and apply column mapping first.")
            return

        A, dp_grid = self._get_kernel()
        if A is None:
            QMessageBox.warning(self, "No kernel",
                                "Compute a model kernel in the Model tab first.")
            return

        method = self._get_method_instance()

        try:
            if self.chk_all_scans.isChecked():
                results = []
                for i in range(self.dataset.n_scans):
                    counts = self.dataset.count_matrix[i, :]
                    # Resize counts to match kernel rows if needed
                    counts_r = self._resize_counts(counts, A.shape[0])
                    r = method.solve(A, counts_r, dp_grid)
                    results.append(r.N_retrieved)
                self.last_result = np.array(results)
                self._plot_heatmap_result(dp_grid)
                self._log(f"Inverted {self.dataset.n_scans} scans.")
            else:
                idx = self.spn_scan_idx.value()
                counts = self.dataset.count_matrix[idx, :]
                counts_r = self._resize_counts(counts, A.shape[0])
                r = method.solve(A, counts_r, dp_grid)
                self.last_result = r.N_retrieved
                self._plot_single_result(r, dp_grid, idx)
                self._log(
                    f"Inversion done. Residual: {r.residual:.4e}\n{r.info}"
                )
            if self.main_window:
                self.main_window.set_status("Inversion complete.")
        except Exception as e:
            QMessageBox.critical(self, "Inversion error", str(e))

    @staticmethod
    def _resize_counts(counts, n_channels):
        """Interpolate or truncate counts to match kernel channel count."""
        if len(counts) == n_channels:
            return counts
        x_old = np.linspace(0, 1, len(counts))
        x_new = np.linspace(0, 1, n_channels)
        return np.interp(x_new, x_old, counts)

    def _plot_single_result(self, result, dp_grid, scan_idx):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("#181825")
        ax.semilogx(dp_grid, result.N_retrieved, color="#a6e3a1", linewidth=2)
        ax.fill_between(dp_grid, result.N_retrieved, alpha=0.3, color="#a6e3a1")
        ax.set_xlabel("Diameter Dp (nm)", color="#cdd6f4")
        ax.set_ylabel("dN/dlogDp  [#/cm³]", color="#cdd6f4")
        ax.set_title(f"Retrieved size distribution – scan {scan_idx}", color="#cdd6f4")
        ax.tick_params(colors="#cdd6f4")
        for s in ax.spines.values(): s.set_edgecolor("#45475a")
        self.fig.tight_layout()
        self.canvas.draw()

    def _plot_heatmap_result(self, dp_grid):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("#181825")
        times = np.arange(self.last_result.shape[0])
        im = ax.pcolormesh(times, dp_grid, self.last_result.T,
                            cmap="viridis", shading="auto")
        ax.set_yscale("log")
        cb = self.fig.colorbar(im, ax=ax)
        cb.ax.tick_params(colors="#cdd6f4")
        ax.set_xlabel("Scan index", color="#cdd6f4")
        ax.set_ylabel("Dp (nm)", color="#cdd6f4")
        ax.set_title("Retrieved dN/dlogDp – all scans", color="#cdd6f4")
        ax.tick_params(colors="#cdd6f4")
        for s in ax.spines.values(): s.set_edgecolor("#45475a")
        self.fig.tight_layout()
        self.canvas.draw()

    def _plot_lcurve(self):
        A, dp_grid = self._get_kernel()
        if A is None or self.dataset.count_matrix is None:
            QMessageBox.warning(self, "L-curve", "Need kernel and data.")
            return
        idx = self.spn_scan_idx.value()
        counts = self._resize_counts(self.dataset.count_matrix[idx, :], A.shape[0])
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
        for s in ax.spines.values(): s.set_edgecolor("#45475a")
        self.fig.tight_layout()
        self.canvas.draw()

    def _save_result(self):
        if self.last_result is None:
            QMessageBox.warning(self, "Save", "Run inversion first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save result", "inversion_result.csv",
                                               "CSV (*.csv)")
        if not path:
            return
        import pandas as pd
        if self.last_result.ndim == 1:
            _, dp_grid = self._get_kernel()
            df = pd.DataFrame({"Dp_nm": dp_grid, "dNdlogDp": self.last_result})
        else:
            _, dp_grid = self._get_kernel()
            df = pd.DataFrame(self.last_result,
                               columns=[f"Dp_{d:.1f}nm" for d in dp_grid])
        df.to_csv(path, index=False)
        self._log(f"Saved → {path}")

    def _log(self, msg):
        self.log.append(msg)

"""
ui/tab_estimation.py
--------------------
Tab 4: Parameter estimation.
"""

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QComboBox, QLabel, QFileDialog,
    QGroupBox, QDoubleSpinBox, QSplitter, QTextEdit,
    QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from modules.data_model import AerosolDataset
from modules.param_estimation import ESTIMATION_METHODS
from modules.measurement_model import MeasurementModel, InstrumentParameters


class EstimationTab(QWidget):
    def __init__(self, dataset: AerosolDataset, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.main_window = parent
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)

        left_w = QWidget(); left_w.setFixedWidth(320)
        left = QVBoxLayout(left_w); left.setSpacing(8)

        # Method
        grp_m = QGroupBox("Estimation Method")
        ml = QVBoxLayout(grp_m)
        self.cmb_method = QComboBox()
        self.cmb_method.addItems(list(ESTIMATION_METHODS.keys()))
        ml.addWidget(self.cmb_method)
        left.addWidget(grp_m)

        # Parameters to estimate
        grp_p = QGroupBox("Parameters to Estimate")
        pl = QVBoxLayout(grp_p)
        pl.addWidget(QLabel(
            "Define initial values and bounds for each parameter.\n"
            "Edit the table below (double-click to edit)."
        ))
        self.tbl_params = QTableWidget(4, 4)
        self.tbl_params.setHorizontalHeaderLabels(["Name", "Initial", "Min", "Max"])
        self.tbl_params.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_params.setMaximumHeight(160)
        # Pre-populate with typical DMA calibration parameters
        defaults = [
            ("sheath_flow", "5.0", "3.0", "8.0"),
            ("aerosol_flow", "0.5", "0.3", "0.8"),
            ("d50_cpc", "10.0", "5.0", "20.0"),
            ("sigma_cpc", "1.2", "0.5", "5.0"),
        ]
        for row, (n, init, mn, mx) in enumerate(defaults):
            for col, val in enumerate([n, init, mn, mx]):
                self.tbl_params.setItem(row, col, QTableWidgetItem(val))
        pl.addWidget(self.tbl_params)
        self.btn_add_row = QPushButton("+ Add parameter")
        self.btn_add_row.clicked.connect(lambda: self.tbl_params.insertRow(self.tbl_params.rowCount()))
        pl.addWidget(self.btn_add_row)
        left.addWidget(grp_p)

        # Scan to use
        grp_s = QGroupBox("Observations")
        sl = QGridLayout(grp_s)
        sl.addWidget(QLabel("Scan index:"), 0, 0)
        self.spn_scan = QDoubleSpinBox(); self.spn_scan.setRange(0, 9999); self.spn_scan.setValue(0)
        sl.addWidget(self.spn_scan, 0, 1)
        left.addWidget(grp_s)

        self.btn_run = QPushButton("Run Estimation")
        self.btn_run.clicked.connect(self._run)
        left.addWidget(self.btn_run)

        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(150)
        left.addWidget(self.log)
        left.addStretch()

        # Right: convergence + residual plot
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

    # ------------------------------------------------------------------

    def _read_params_table(self):
        initial, bounds = {}, {}
        for row in range(self.tbl_params.rowCount()):
            try:
                name  = self.tbl_params.item(row, 0).text().strip()
                init  = float(self.tbl_params.item(row, 1).text())
                lo    = float(self.tbl_params.item(row, 2).text())
                hi    = float(self.tbl_params.item(row, 3).text())
                if name:
                    initial[name] = init
                    bounds[name] = (lo, hi)
            except (AttributeError, ValueError):
                pass
        return initial, bounds

    def _build_forward_model(self, base_params: InstrumentParameters):
        """
        Returns a callable forward_model(param_dict) → counts array.
        Modifies only the parameters listed in the table.
        """
        def forward(p: dict) -> np.ndarray:
            import copy
            ip = copy.deepcopy(base_params)
            if "sheath_flow" in p:  ip.dma.sheath_flow = p["sheath_flow"]
            if "aerosol_flow" in p: ip.dma.aerosol_flow = p["aerosol_flow"]
            if "d50_cpc" in p:      ip.cpc.d50 = p["d50_cpc"]
            if "sigma_cpc" in p:    ip.cpc.sigma = p["sigma_cpc"]
            mdl = MeasurementModel(ip)
            mdl.compute()
            # Simulate counts from a test distribution (flat for demo)
            N_test = np.ones(mdl.dp_grid.shape)
            return mdl.apply(N_test)
        return forward

    def _run(self):
        initial, bounds = self._read_params_table()
        if not initial:
            QMessageBox.warning(self, "Params", "Add at least one parameter.")
            return

        if self.dataset.count_matrix is None:
            QMessageBox.warning(self, "No data", "Load data first.")
            return

        idx = int(self.spn_scan.value())
        observations = self.dataset.count_matrix[idx, :]

        # Get instrument params from model tab if available
        if self.main_window and hasattr(self.main_window, "tab_model"):
            mdl = self.main_window.tab_model.model
            base_params = mdl.params if mdl else InstrumentParameters()
        else:
            base_params = InstrumentParameters()

        forward = self._build_forward_model(base_params)

        # Resize observations to match forward model output
        n_ch = base_params.n_channels
        if len(observations) != n_ch:
            x_old = np.linspace(0, 1, len(observations))
            x_new = np.linspace(0, 1, n_ch)
            observations = np.interp(x_new, x_old, observations)

        method_name = self.cmb_method.currentText()
        estimator_cls = ESTIMATION_METHODS[method_name]
        estimator = estimator_cls()

        try:
            result = estimator.fit(forward, observations, initial,
                                   bounds if "Differential" in method_name else None)
            self._log(
                f"Converged: {result.converged}  Iterations: {result.n_iterations}\n"
                f"Residual: {result.residual:.4e}\n"
                f"Estimated parameters:\n" +
                "\n".join(f"  {k} = {v:.6g}" for k, v in result.params_estimated.items())
            )
            self._plot_result(forward, observations, result, base_params.n_channels)
        except Exception as e:
            QMessageBox.critical(self, "Estimation error", str(e))

    def _plot_result(self, forward, observations, result, n_ch):
        fitted = forward(result.params_estimated)
        channels = np.arange(n_ch)

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("#181825")
        ax.plot(channels, observations, "o", color="#89b4fa", markersize=4,
                label="Observed counts")
        ax.plot(channels, fitted, "-", color="#f38ba8", linewidth=2,
                label="Model (fitted params)")
        ax.set_xlabel("Channel", color="#cdd6f4")
        ax.set_ylabel("Counts", color="#cdd6f4")
        ax.set_title("Parameter Estimation – Observed vs. Fitted", color="#cdd6f4")
        ax.tick_params(colors="#cdd6f4")
        for s in ax.spines.values(): s.set_edgecolor("#45475a")
        ax.legend(facecolor="#313244", edgecolor="#45475a", labelcolor="#cdd6f4")
        self.fig.tight_layout()
        self.canvas.draw()

    def _log(self, msg):
        self.log.append(msg)

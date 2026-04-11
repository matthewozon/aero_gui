"""
ui/tab_model.py
---------------
Tab 2: Measurement model configuration and visualisation.
"""

from matplotlib import pyplot as plt
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QComboBox, QLabel, QFileDialog,
    QGroupBox, QDoubleSpinBox, QSpinBox, QSplitter,
    QTextEdit, QMessageBox
)
from PyQt5.QtCore import Qt

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from modules.data_model import AerosolDataset
from modules.measurement_model import (
    MeasurementModel, InstrumentParameters,
    DMAParameters, CPCParameters
)


class ModelTab(QWidget):
    def __init__(self, dataset: AerosolDataset, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.main_window = parent
        self.model = None
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)

        # ---- Left: parameter controls ----
        left_w = QWidget()
        left_w.setFixedWidth(300)
        left = QVBoxLayout(left_w)
        left.setSpacing(8)

        # Model type
        grp_type = QGroupBox("Model Type")
        tl = QVBoxLayout(grp_type)
        self.cmb_model = QComboBox()
        self.cmb_model.addItems(list(MeasurementModel.MODELS.keys()))
        tl.addWidget(self.cmb_model)
        left.addWidget(grp_type)

        # DMA parameters
        grp_dma = QGroupBox("DMA Geometry & Flows")
        dl = QGridLayout(grp_dma)
        self._add_dspin(dl, 0, "Inner radius (cm):", "spn_r1", 0.1, 10, 0.937)
        self._add_dspin(dl, 1, "Outer radius (cm):", "spn_r2", 0.1, 100, 1.961)
        self._add_dspin(dl, 2, "Length (cm):", "spn_L", 1, 500, 44.369)
        self._add_dspin(dl, 3, "Sheath flow (L/min):", "spn_Qsh", 0.1, 500, 5.0)
        self._add_dspin(dl, 4, "Sample flow (L/min):", "spn_Qa", 0.01, 10, 0.5)
        self._add_dspin(dl, 5, "Temperature (K):", "spn_T", 250, 350, 298.15)
        self._add_dspin(dl, 6, "Pressure (Pa):", "spn_P", 50000, 120000, 101325)
        self._add_dspin(dl, 7, "Number of charges:", "spn_ncharges", -1, 1, 1)
        left.addWidget(grp_dma)

        # Scan parameters
        grp_scan = QGroupBox("Scan Settings")
        sl = QGridLayout(grp_scan)
        self._add_dspin(sl, 0, "V min (V):", "spn_Vmin", 0.1, 1000, 10)
        self._add_dspin(sl, 1, "V max (V):", "spn_Vmax", 100, 500000, 10000)
        self._add_spin(sl, 2, "Channels:", "spn_nch", 8, 512, 64)
        left.addWidget(grp_scan)

        # CPC parameters
        grp_cpc = QGroupBox("CPC Parameters")
        cl = QGridLayout(grp_cpc)
        self._add_dspin(cl, 0, "D₅₀ (nm):", "spn_d50", 1, 100, 10)
        self._add_dspin(cl, 1, "σ (nm):", "spn_sigma", 0.1, 20, 1.2)
        left.addWidget(grp_cpc)

        # Actions
        self.btn_compute = QPushButton("Compute Model Kernel")
        self.btn_compute.clicked.connect(self._compute_model)
        left.addWidget(self.btn_compute)

        grp_save = QGroupBox("Save Model")
        save_l = QHBoxLayout(grp_save)
        self.btn_save_csv = QPushButton("Save CSV")
        self.btn_save_csv.clicked.connect(lambda: self._save_model("csv"))
        self.btn_save_xlsx = QPushButton("Save XLSX")
        self.btn_save_xlsx.clicked.connect(lambda: self._save_model("xlsx"))
        save_l.addWidget(self.btn_save_csv)
        save_l.addWidget(self.btn_save_xlsx)
        left.addWidget(grp_save)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(100)
        left.addWidget(self.log)
        left.addStretch()

        # ---- Right: plots ----
        right_w = QWidget()
        right = QVBoxLayout(right_w)

        self.fig = Figure(figsize=(9, 7), facecolor="#1e1e2e")
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

    def _add_dspin(self, layout, row, label, attr, mn, mx, val):
        layout.addWidget(QLabel(label), row, 0)
        spn = QDoubleSpinBox()
        spn.setRange(mn, mx)
        spn.setValue(val)
        spn.setDecimals(3)
        layout.addWidget(spn, row, 1)
        setattr(self, attr, spn)

    def _add_spin(self, layout, row, label, attr, mn, mx, val):
        layout.addWidget(QLabel(label), row, 0)
        spn = QSpinBox()
        spn.setRange(mn, mx)
        spn.setValue(val)
        layout.addWidget(spn, row, 1)
        setattr(self, attr, spn)

    # ------------------------------------------------------------------

    def _build_params(self) -> InstrumentParameters:
        dma = DMAParameters(
            inner_radius=self.spn_r1.value() * 1e-2,
            outer_radius=self.spn_r2.value() * 1e-2,
            length=self.spn_L.value() * 1e-2,
            sheath_flow=self.spn_Qsh.value(),
            aerosol_flow=self.spn_Qa.value(),
            temperature=self.spn_T.value(),
            pressure=self.spn_P.value(),
            n_charges=self.spn_ncharges.value(),
        )
        cpc = CPCParameters(
            d50=self.spn_d50.value(),
            sigma=self.spn_sigma.value(),
        )
        return InstrumentParameters(
            dma=dma, cpc=cpc,
            v_min=self.spn_Vmin.value(),
            v_max=self.spn_Vmax.value(),
            n_channels=self.spn_nch.value(),
        )

    def _compute_model(self):
        try:
            params = self._build_params()
            model_key = self.cmb_model.currentText()
            model_type = MeasurementModel.MODELS[model_key]
            self.model = MeasurementModel(params, model_type)
            self.model.compute()
            self._log(
                f"Kernel computed: {self.model.kernel.shape[0]} channels × "
                f"{self.model.kernel.shape[1]} Dp bins\n"
                f"Dp range: {self.model.dp_grid[0]:.1f} – {self.model.dp_grid[-1]:.1f} nm"
            )
            self._plot_model()
            if self.main_window:
                self.main_window.set_status("Measurement model kernel computed.")
        except Exception as e:
            QMessageBox.critical(self, "Model error", str(e))

    def _plot_model(self):
        if self.model is None:
            return
        self.fig.clear()

        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212)
        for ax in (ax1, ax2):
            ax.set_facecolor("#181825")
            ax.tick_params(colors="#cdd6f4")
            for s in ax.spines.values():
                s.set_edgecolor("#45475a")

        # Kernel heatmap
        A = self.model.kernel
        print(f"self.model.dp_grid: {self.model.dp_grid}")
        print(f"self.model.diameters: {self.model.diameters}")
        im = ax1.pcolormesh(self.model.dp_grid, self.model.diameters, A,
                             cmap="plasma", shading="auto")
        cb = self.fig.colorbar(im, ax=ax1)
        cb.ax.tick_params(colors="#cdd6f4")
        ax1.set_xscale("log"); ax1.set_yscale("log")
        ax1.set_xlabel("True Dp (nm)", color="#cdd6f4")
        ax1.set_ylabel("Selected Dp (nm)", color="#cdd6f4")
        ax1.set_title("Transfer Kernel A  [channel × Dp]", color="#cdd6f4")

        # Transfer function for a few channels
        indices = np.linspace(0, A.shape[0] - 1, 6, dtype=int)
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(indices)))
        for k, idx in enumerate(indices):
            ax2.plot(self.model.dp_grid, A[idx, :],
                     color=colors[k],
                     label=f"Dp={self.model.diameters[idx]:.0f} nm")
        ax2.set_xscale("log")
        ax2.set_xlabel("True Dp (nm)", color="#cdd6f4")
        ax2.set_ylabel("Kernel weight", color="#cdd6f4")
        ax2.set_title("Selected transfer function rows", color="#cdd6f4")
        ax2.legend(fontsize=8, labelcolor="#cdd6f4",
                   facecolor="#313244", edgecolor="#45475a")

        self.fig.tight_layout()
        self.canvas.draw()

    def _save_model(self, fmt: str):
        if self.model is None:
            QMessageBox.warning(self, "Save", "Compute the model first.")
            return
        ext = "csv" if fmt == "csv" else "xlsx"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save model", f"kernel.{ext}",
            f"{'CSV' if fmt=='csv' else 'Excel'} (*.{ext})"
        )
        if not path:
            return
        try:
            self.model.save(path, fmt=fmt)
            self._log(f"Model saved → {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))

    def _log(self, msg):
        self.log.append(msg)

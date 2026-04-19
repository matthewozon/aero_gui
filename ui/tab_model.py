"""
ui/tab_model.py
---------------
Tab 2: Measurement model configuration and visualisation.

Changes in this version
------------------------
1.  Max charges (Nq) spin box — controls how many elementary charge
    states (1 … |Nq|) are summed in the transfer function.  Sign sets
    DMA polarity (negative = standard SMPS, positive = positive DMA).

2.  DMA transfer function model selector (triangle / NDTF / DTF),
    replacing the old "model type" combo that only switched the CPC model.

3.  Air viscosity and mean free path are now computed from T and P via
    _air_properties() from modules/algo/measurement and displayed in the
    log so the user can verify the operating-condition corrections.

4.  Kernel computation now calls MeasurementModel (which in turn calls
    smps3936_transfer_function from modules/algo/measurement), giving the
    full BAYROSOL signal chain.
"""

import numpy as np
import matplotlib.pyplot as plt
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
    DMAParameters, CPCParameters, ImpactorParameters,
)
from modules.algo.measurement import _air_properties


class ModelTab(QWidget):
    def __init__(self, dataset: AerosolDataset, parent=None):
        super().__init__(parent)
        self.dataset     = dataset
        self.main_window = parent
        self.model       = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QHBoxLayout(self)

        # ── Left: parameter controls ───────────────────────────────
        left_w = QWidget()
        left_w.setFixedWidth(310)
        left = QVBoxLayout(left_w)
        left.setSpacing(8)

        # ── DMA transfer function model ───────────────────────────
        grp_tf = QGroupBox("DMA Transfer Function Model")
        tfl = QVBoxLayout(grp_tf)
        self.cmb_tf = QComboBox()
        self.cmb_tf.addItems(list(MeasurementModel.MODELS.keys()))
        tfl.addWidget(self.cmb_tf)
        left.addWidget(grp_tf)

        # ── DMA geometry & flows ──────────────────────────────────
        grp_dma = QGroupBox("DMA Geometry & Flows")
        dl = QGridLayout(grp_dma)
        self._add_dspin(dl, 0, "Inner radius (cm):",  "spn_r1",  0.1,   10,     0.937)
        self._add_dspin(dl, 1, "Outer radius (cm):",  "spn_r2",  0.1,   10,     1.961)
        self._add_dspin(dl, 2, "Length (cm):",        "spn_L",   1,     200,    44.369)
        self._add_dspin(dl, 3, "Sheath flow (L/min):", "spn_Qsh", 0.1,  100,    5.0)
        self._add_dspin(dl, 4, "Sample flow (L/min):", "spn_Qa",  0.01, 10,     0.5)
        self._add_dspin(dl, 5, "Temperature (K):",    "spn_T",   200,   400,    298.15)
        self._add_dspin(dl, 6, "Pressure (Pa):",      "spn_P",   50000, 120000, 101325)
        # connect T and P changes to update the live viscosity display
        self.spn_T.valueChanged.connect(self._update_air_props)
        self.spn_P.valueChanged.connect(self._update_air_props)
        left.addWidget(grp_dma)

        # ── Live air-property display ─────────────────────────────
        grp_air = QGroupBox("Air Properties (computed from T, P)")
        al = QGridLayout(grp_air)
        al.addWidget(QLabel("η (Pa·s):"),  0, 0)
        self.lbl_eta  = QLabel("—")
        al.addWidget(self.lbl_eta,  0, 1)
        al.addWidget(QLabel("λ (nm):"),    1, 0)
        self.lbl_mfp  = QLabel("—")
        al.addWidget(self.lbl_mfp,  1, 1)
        left.addWidget(grp_air)
        self._update_air_props()   # populate on startup

        # ── Charge model ──────────────────────────────────────────
        grp_chg = QGroupBox("Charge Model")
        cl = QGridLayout(grp_chg)
        cl.addWidget(QLabel("Max charges Nq:"), 0, 0)
        self.spn_Nq = QSpinBox()
        self.spn_Nq.setRange(-12, 12)
        self.spn_Nq.setValue(-6)
        self.spn_Nq.setToolTip(
            "Number of charge states summed in the transfer function.\n"
            "Negative = negative-polarity DMA (standard SMPS).\n"
            "Positive = positive-polarity DMA.\n"
            "Absolute value = number of states included (1 … |Nq|)."
        )
        cl.addWidget(self.spn_Nq, 0, 1)
        cl.addWidget(QLabel("(−6 = neg. polarity, 6 states)"), 1, 0, 1, 2)
        left.addWidget(grp_chg)

        # ── Scan / voltage settings ───────────────────────────────
        grp_scan = QGroupBox("Scan Settings")
        sl = QGridLayout(grp_scan)
        self._add_dspin(sl, 0, "V min (V):", "spn_Vmin", 1,   1000,  10)
        self._add_dspin(sl, 1, "V max (V):", "spn_Vmax", 100, 50000, 10000)
        self._add_spin( sl, 2, "Channels:", "spn_nch",  8,   512,   64)
        left.addWidget(grp_scan)

        # ── CPC parameters ────────────────────────────────────────
        grp_cpc = QGroupBox("CPC Parameters")
        cpl = QGridLayout(grp_cpc)
        self._add_dspin(cpl, 0, "D₅₀ (nm):",    "spn_d50",   1,   100, 10)
        self._add_dspin(cpl, 1, "δ₅₀ (nm):",    "spn_dcpc",  0.1, 20,  1.0)
        left.addWidget(grp_cpc)

        # ── Impactor parameters ───────────────────────────────────
        grp_imp = QGroupBox("Impactor Parameters")
        ipl = QGridLayout(grp_imp)
        self._add_dspin(ipl, 0, "D₅₀ (nm):",  "spn_imp_d50",    100, 10000, 1000)
        self._add_dspin(ipl, 1, "δ₅₀ (nm):",  "spn_imp_delta",  10,  1000,  100)
        left.addWidget(grp_imp)

        # ── Actions ──────────────────────────────────────────────
        self.btn_compute = QPushButton("Compute Model Kernel")
        self.btn_compute.clicked.connect(self._compute_model)
        left.addWidget(self.btn_compute)

        grp_save = QGroupBox("Save Model")
        save_l = QHBoxLayout(grp_save)
        self.btn_save_csv  = QPushButton("Save CSV")
        self.btn_save_csv.clicked.connect(lambda: self._save_model("csv"))
        self.btn_save_xlsx = QPushButton("Save XLSX")
        self.btn_save_xlsx.clicked.connect(lambda: self._save_model("xlsx"))
        save_l.addWidget(self.btn_save_csv)
        save_l.addWidget(self.btn_save_xlsx)
        left.addWidget(grp_save)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(110)
        left.addWidget(self.log)
        left.addStretch()

        # ── Right: plots ──────────────────────────────────────────
        right_w = QWidget()
        right = QVBoxLayout(right_w)
        self.fig     = Figure(figsize=(9, 7), facecolor="#1e1e2e")
        self.canvas  = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        right.addWidget(self.toolbar)
        right.addWidget(self.canvas)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_w)
        splitter.addWidget(right_w)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter)

    # ------------------------------------------------------------------
    # Helper spin-box factories
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
    # Live air-property update
    # ------------------------------------------------------------------

    def _update_air_props(self):
        """Recompute eta and lambda from the current T/P spinboxes."""
        try:
            eta, lmfp = _air_properties(self.spn_T.value(), self.spn_P.value())
            self.lbl_eta.setText(f"{eta:.4e}")
            self.lbl_mfp.setText(f"{lmfp * 1e9:.2f}")   # display in nm
        except Exception:
            self.lbl_eta.setText("—")
            self.lbl_mfp.setText("—")

    # ------------------------------------------------------------------
    # Parameter assembly
    # ------------------------------------------------------------------

    def _build_params(self) -> InstrumentParameters:
        dma = DMAParameters(
            inner_radius = self.spn_r1.value()  * 1e-2,
            outer_radius = self.spn_r2.value()  * 1e-2,
            length       = self.spn_L.value()   * 1e-2,
            sheath_flow  = self.spn_Qsh.value(),
            aerosol_flow = self.spn_Qa.value(),
            temperature  = self.spn_T.value(),
            pressure     = self.spn_P.value(),
            max_charges  = self.spn_Nq.value(),
        )
        cpc = CPCParameters(
            d50    = self.spn_d50.value(),
            delta50 = self.spn_dcpc.value(),
        )
        imp = ImpactorParameters(
            d50    = self.spn_imp_d50.value(),
            delta50 = self.spn_imp_delta.value(),
        )
        tf_model = MeasurementModel.MODELS[self.cmb_tf.currentText()]
        return InstrumentParameters(
            dma       = dma,
            cpc       = cpc,
            impactor  = imp,
            v_min     = self.spn_Vmin.value(),
            v_max     = self.spn_Vmax.value(),
            n_channels = self.spn_nch.value(),
            tf_model  = tf_model,
        )

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def _compute_model(self):
        try:
            params   = self._build_params()
            tf_model = params.tf_model
            self.model = MeasurementModel(params, tf_model)
            self.model.compute()

            Nq  = params.dma.max_charges
            eta, lmfp = _air_properties(params.dma.temperature, params.dma.pressure)

            self._log(
                f"Kernel: {self.model.kernel.shape[0]} channels × "
                f"{self.model.kernel.shape[1]} Dp bins\n"
                f"Dp range: {self.model.dp_grid[0]:.1f} – "
                f"{self.model.dp_grid[-1]:.1f} nm\n"
                f"Nq = {Nq}  ({abs(Nq)} charge states, "
                f"{'negative' if Nq < 0 else 'positive'} polarity)\n"
                f"η = {eta:.4e} Pa·s   λ = {lmfp*1e9:.2f} nm  "
                f"(T={params.dma.temperature:.1f} K, "
                f"P={params.dma.pressure:.0f} Pa)"
            )
            self._plot_model()
            if self.main_window:
                self.main_window.set_status(
                    f"Kernel computed — {abs(Nq)} charge states, "
                    f"TF: {self.cmb_tf.currentText()}"
                )
        except Exception as e:
            QMessageBox.critical(self, "Model error", str(e))

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

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

        # ── Kernel heatmap ────────────────────────────────────────
        A  = self.model.kernel
        im = ax1.pcolormesh(self.model.dp_grid, self.model.diameters, A,
                             cmap="plasma", shading="auto")
        cb = self.fig.colorbar(im, ax=ax1)
        cb.ax.tick_params(colors="#cdd6f4")
        ax1.set_xscale("log"); ax1.set_yscale("log")
        ax1.set_xlabel("True Dp (nm)",     color="#cdd6f4")
        ax1.set_ylabel("Selected Dp (nm)", color="#cdd6f4")
        Nq = self.params_used.dma.max_charges if hasattr(self, 'params_used') else ""
        ax1.set_title("Transfer Kernel A  [channel × Dp]", color="#cdd6f4")

        # ── Transfer function rows for a few channels ─────────────
        indices = np.linspace(0, A.shape[0] - 1, 6, dtype=int)
        colors  = plt.cm.viridis(np.linspace(0.2, 0.9, len(indices)))
        for k, idx in enumerate(indices):
            ax2.plot(self.model.dp_grid, A[idx, :],
                     color=colors[k],
                     label=f"Dp={self.model.diameters[idx]:.0f} nm")
        ax2.set_xscale("log")
        ax2.set_xlabel("True Dp (nm)",   color="#cdd6f4")
        ax2.set_ylabel("Kernel weight",  color="#cdd6f4")
        ax2.set_title("Selected kernel rows", color="#cdd6f4")
        ax2.legend(fontsize=8, labelcolor="#cdd6f4",
                   facecolor="#313244", edgecolor="#45475a")

        self.fig.tight_layout()
        self.canvas.draw()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save_model(self, fmt: str):
        if self.model is None:
            QMessageBox.warning(self, "Save", "Compute the model first.")
            return
        ext = "csv" if fmt == "csv" else "xlsx"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save model", f"kernel.{ext}",
            f"{'CSV' if fmt == 'csv' else 'Excel'} (*.{ext})"
        )
        if not path:
            return
        try:
            self.model.save(path, fmt=fmt)
            self._log(f"Model saved → {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))

    def _log(self, msg: str):
        self.log.append(msg)

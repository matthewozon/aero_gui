"""
ui/tab_simulation.py
--------------------
Tab 5: Aerosol GDE simulation.
"""

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QComboBox, QLabel, QFileDialog,
    QGroupBox, QDoubleSpinBox, QSpinBox, QSplitter,
    QTextEdit, QMessageBox, QCheckBox, QTabWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from modules.simulation import GDESimulator, SimulationConfig


# Run simulation in a thread so GUI stays responsive
class SimWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, sim: GDESimulator, N0: np.ndarray):
        super().__init__()
        self.sim = sim
        self.N0 = N0

    def run(self):
        try:
            self.sim.run(self.N0)
            self.finished.emit(self.sim)
        except Exception as e:
            self.error.emit(str(e))


class SimulationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.sim = None
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)

        left_w = QWidget(); left_w.setFixedWidth(300)
        left = QVBoxLayout(left_w); left.setSpacing(8)

        # Size grid
        grp_grid = QGroupBox("Size Grid")
        gl = QGridLayout(grp_grid)
        gl.addWidget(QLabel("Dp min (nm):"), 0, 0)
        self.spn_dpmin = QDoubleSpinBox(); self.spn_dpmin.setRange(0.1, 100); self.spn_dpmin.setValue(3)
        gl.addWidget(self.spn_dpmin, 0, 1)
        gl.addWidget(QLabel("Dp max (nm):"), 1, 0)
        self.spn_dpmax = QDoubleSpinBox(); self.spn_dpmax.setRange(10, 10000); self.spn_dpmax.setValue(1000)
        gl.addWidget(self.spn_dpmax, 1, 1)
        gl.addWidget(QLabel("# bins:"), 2, 0)
        self.spn_nbins = QSpinBox(); self.spn_nbins.setRange(8, 256); self.spn_nbins.setValue(64)
        gl.addWidget(self.spn_nbins, 2, 1)
        left.addWidget(grp_grid)

        # Time settings
        grp_time = QGroupBox("Time Settings")
        tl = QGridLayout(grp_time)
        tl.addWidget(QLabel("Duration (s):"), 0, 0)
        self.spn_tend = QDoubleSpinBox(); self.spn_tend.setRange(60, 86400); self.spn_tend.setValue(3600)
        tl.addWidget(self.spn_tend, 0, 1)
        tl.addWidget(QLabel("Output interval (s):"), 1, 0)
        self.spn_dtout = QDoubleSpinBox(); self.spn_dtout.setRange(1, 3600); self.spn_dtout.setValue(60)
        tl.addWidget(self.spn_dtout, 1, 1)
        left.addWidget(grp_time)

        # Physical processes
        grp_proc = QGroupBox("Physical Processes")
        pl = QVBoxLayout(grp_proc)
        self.chk_coag = QCheckBox("Brownian coagulation"); self.chk_coag.setChecked(True)
        self.chk_cond = QCheckBox("Condensation")
        self.chk_nucl = QCheckBox("Nucleation (constant source)")
        self.chk_dep  = QCheckBox("Gravitational deposition")
        for w in [self.chk_coag, self.chk_cond, self.chk_nucl, self.chk_dep]:
            pl.addWidget(w)
        left.addWidget(grp_proc)

        # Initial distribution
        grp_init = QGroupBox("Initial Distribution")
        il = QGridLayout(grp_init)
        il.addWidget(QLabel("Mode (log-normal):"), 0, 0)
        il.addWidget(QLabel("N₀ [#/cm³]:"), 1, 0)
        self.spn_N0 = QDoubleSpinBox(); self.spn_N0.setRange(0, 1e8); self.spn_N0.setValue(1e4)
        il.addWidget(self.spn_N0, 1, 1)
        il.addWidget(QLabel("Dg (nm):"), 2, 0)
        self.spn_Dg = QDoubleSpinBox(); self.spn_Dg.setRange(1, 1000); self.spn_Dg.setValue(100)
        il.addWidget(self.spn_Dg, 2, 1)
        il.addWidget(QLabel("σg:"), 3, 0)
        self.spn_sg = QDoubleSpinBox(); self.spn_sg.setRange(1.01, 4); self.spn_sg.setValue(1.8)
        il.addWidget(self.spn_sg, 3, 1)
        left.addWidget(grp_init)

        self.btn_run = QPushButton("▶  Run Simulation")
        self.btn_run.clicked.connect(self._run_sim)
        left.addWidget(self.btn_run)

        self.btn_save = QPushButton("Save results…")
        self.btn_save.clicked.connect(self._save_results)
        left.addWidget(self.btn_save)

        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(100)
        left.addWidget(self.log)
        left.addStretch()

        # Right: plot tabs
        right_w = QWidget()
        right = QVBoxLayout(right_w)
        self.plot_tabs = QTabWidget()

        # Heatmap
        self.fig_heat = Figure(figsize=(9, 5), facecolor="#1e1e2e")
        self.canvas_heat = FigureCanvas(self.fig_heat)
        self.tb_heat = NavigationToolbar(self.canvas_heat, self)
        hw = QWidget(); hl = QVBoxLayout(hw)
        hl.addWidget(self.tb_heat); hl.addWidget(self.canvas_heat)
        self.plot_tabs.addTab(hw, "2D Heatmap")

        # Total number
        self.fig_N = Figure(figsize=(9, 4), facecolor="#1e1e2e")
        self.canvas_N = FigureCanvas(self.fig_N)
        self.tb_N = NavigationToolbar(self.canvas_N, self)
        nw = QWidget(); nl = QVBoxLayout(nw)
        nl.addWidget(self.tb_N); nl.addWidget(self.canvas_N)
        self.plot_tabs.addTab(nw, "Total N(t)")

        right.addWidget(self.plot_tabs)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_w)
        splitter.addWidget(right_w)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter)

    # ------------------------------------------------------------------

    def _make_initial_distribution(self, dp_grid: np.ndarray) -> np.ndarray:
        """Log-normal initial distribution."""
        N0 = self.spn_N0.value()
        Dg = self.spn_Dg.value()
        sg = self.spn_sg.value()
        log_dp = np.log(dp_grid)
        log_Dg = np.log(Dg)
        log_sg = np.log(sg)
        N = (N0 / (np.sqrt(2 * np.pi) * log_sg)) * \
            np.exp(-0.5 * ((log_dp - log_Dg) / log_sg) ** 2)
        # Normalise to total N0
        if N.sum() > 0:
            N = N / N.sum() * N0
        return N

    def _run_sim(self):
        dp_grid = np.logspace(
            np.log10(self.spn_dpmin.value()),
            np.log10(self.spn_dpmax.value()),
            self.spn_nbins.value()
        )
        cfg = SimulationConfig(
            dp_grid_nm=dp_grid,
            t_end=self.spn_tend.value(),
            dt_output=self.spn_dtout.value(),
            coagulation=self.chk_coag.isChecked(),
            condensation=self.chk_cond.isChecked(),
            nucleation=self.chk_nucl.isChecked(),
            deposition=self.chk_dep.isChecked(),
        )
        if self.chk_nucl.isChecked():
            cfg.nucleation_source = lambda t: 10.0   # 10 #/cm³/s constant
        if self.chk_dep.isChecked():
            from modules.simulation import gravitational_settling_velocity
            v = gravitational_settling_velocity(dp_grid, rho_p=1500)
            cfg.deposition_rate = v / 5.0  # assuming 5 m mixing height

        self.sim = GDESimulator(cfg)
        N0 = self._make_initial_distribution(dp_grid)

        self.btn_run.setEnabled(False)
        self.btn_run.setText("Running…")
        self._log("Starting simulation…")

        self._worker = SimWorker(self.sim, N0)
        self._worker.finished.connect(self._on_sim_done)
        self._worker.error.connect(self._on_sim_error)
        self._worker.start()

    def _on_sim_done(self, sim: GDESimulator):
        self.sim = sim
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶  Run Simulation")
        self._log(
            f"Done. {sim.result_N.shape[0]} time steps × {sim.result_N.shape[1]} bins."
        )
        self._plot_results()
        if self.main_window:
            self.main_window.set_status("Simulation complete.")

    def _on_sim_error(self, msg: str):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶  Run Simulation")
        QMessageBox.critical(self, "Simulation error", msg)

    def _plot_results(self):
        if self.sim is None or self.sim.result_N is None:
            return
        times, dp_grid, N = self.sim.get_heatmap_data()

        # Heatmap
        self.fig_heat.clear()
        ax = self.fig_heat.add_subplot(111)
        ax.set_facecolor("#181825")
        im = ax.pcolormesh(times / 60, dp_grid, N.T,
                            cmap="plasma", shading="auto")
        ax.set_yscale("log")
        cb = self.fig_heat.colorbar(im, ax=ax)
        cb.ax.tick_params(colors="#cdd6f4")
        cb.set_label("dN/dlogDp [#/cm³]", color="#cdd6f4")
        ax.set_xlabel("Time (min)", color="#cdd6f4")
        ax.set_ylabel("Dp (nm)", color="#cdd6f4")
        ax.set_title("Simulated Size Distribution Evolution", color="#cdd6f4")
        ax.tick_params(colors="#cdd6f4")
        for s in ax.spines.values(): s.set_edgecolor("#45475a")
        self.fig_heat.tight_layout()
        self.canvas_heat.draw()

        # Total N(t)
        Ntot = self.sim.get_total_number()
        self.fig_N.clear()
        ax2 = self.fig_N.add_subplot(111)
        ax2.set_facecolor("#181825")
        ax2.plot(times / 60, Ntot, color="#89b4fa", linewidth=2)
        ax2.fill_between(times / 60, Ntot, alpha=0.2, color="#89b4fa")
        ax2.set_xlabel("Time (min)", color="#cdd6f4")
        ax2.set_ylabel("Total N [#/cm³]", color="#cdd6f4")
        ax2.set_title("Total number concentration vs. time", color="#cdd6f4")
        ax2.tick_params(colors="#cdd6f4")
        for s in ax2.spines.values(): s.set_edgecolor("#45475a")
        self.fig_N.tight_layout()
        self.canvas_N.draw()

    def _save_results(self):
        if self.sim is None or self.sim.result_N is None:
            QMessageBox.warning(self, "Save", "Run a simulation first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save simulation", "sim_result.csv",
                                               "CSV (*.csv)")
        if not path:
            return
        import pandas as pd
        times, dp_grid, N = self.sim.get_heatmap_data()
        df = pd.DataFrame(N,
                           index=pd.Index(times, name="time_s"),
                           columns=pd.Index(dp_grid, name="Dp_nm"))
        df.to_csv(path)
        self._log(f"Saved → {path}")

    def _log(self, msg):
        self.log.append(msg)

"""
ui/tab_simulation.py
--------------------
Tab 5: Aerosol GDE simulation.

Supports two backends:
  - Legacy  : GDESimulator from modules/simulation.py
  - Direct  : algo/gde.py (AeroSys + simulate) — exposes full physics control
"""

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QComboBox, QLabel, QFileDialog,
    QGroupBox, QDoubleSpinBox, QSpinBox, QSplitter,
    QTextEdit, QMessageBox, QCheckBox, QTabWidget,
    QScrollArea, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from modules.simulation import GDESimulator, SimulationConfig


# ─── helpers ──────────────────────────────────────────────────────────────────

def _lognormal_cm3(dp_nm: np.ndarray, N0_cm3: float,
                   Dg_nm: float, sg: float) -> np.ndarray:
    """Log-normal size distribution [#/cm³] on a diameter grid [nm]."""
    log_sg = np.log(sg)
    N = (N0_cm3 / (np.sqrt(2 * np.pi) * log_sg)) * \
        np.exp(-0.5 * ((np.log(dp_nm) - np.log(Dg_nm)) / log_sg) ** 2)
    total = N.sum()
    if total > 0:
        N = N / total * N0_cm3
    return N


# ─── workers ──────────────────────────────────────────────────────────────────

class LegacySimWorker(QThread):
    finished = pyqtSignal(object)   # GDESimulator
    error    = pyqtSignal(str)

    def __init__(self, sim: GDESimulator, N0: np.ndarray):
        super().__init__()
        self.sim = sim
        self.N0  = N0

    def run(self):
        try:
            self.sim.run(self.N0)
            self.finished.emit(self.sim)
        except Exception as e:
            self.error.emit(str(e))


class GDEDirectWorker(QThread):
    """Run algo/gde.py directly with full physics control."""
    finished = pyqtSignal(object)   # dict with times, dp_nm, N
    error    = pyqtSignal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self):
        try:
            self._run()
        except Exception as e:
            import traceback
            self.error.emit(traceback.format_exc())

    def _run(self):
        from modules.algo.gde import (
            AeroSys, coagulation_coefficient, init_coagulation_indices,
            wall_deposition_rate, simulate,
        )
        p = self.params

        # ── size grid [m] ──────────────────────────────────────────────
        dp_m = np.logspace(
            np.log10(p['dp_min_nm'] * 1e-9),
            np.log10(p['dp_max_nm'] * 1e-9),
            p['n_bins'],
        )

        # ── characteristic scales ──────────────────────────────────────
        x_c = max(p['N0_cm3'] * 1e6, 1.0)    # [#/m³]
        t_c = float(p['t_end'])               # [s]

        GR_base_ms = p['growth_rate_nm_h'] * 1e-9 / 3600.0  # [m/s]
        GR_c = max(GR_base_ms, 1e-20)

        J_phys_m3 = p['nucleation_rate'] * 1e6   # [#/(m³·s)]
        J_c = max(J_phys_m3, 1.0)

        # ── build workspace (gamma_c placeholder = 1, updated below) ──
        ws = AeroSys.from_diameters(
            dp_m,
            beta_c=1e-15,
            GR_c=GR_c,
            J_c=J_c,
            gamma_c=1.0,
            t_c=t_c,
            x_c=x_c,
            scale='log',
        )
        ws.is_coa      = p['coagulation']
        ws.is_coa_gain = p['coagulation']
        ws.is_con      = p['condensation'] and (GR_base_ms > 0)
        ws.is_nuc      = p['nucleation']
        ws.is_los      = p['deposition']

        # ── coagulation ────────────────────────────────────────────────
        if ws.is_coa:
            coagulation_coefficient(ws,
                                    temperature=p['temperature'],
                                    pressure=p['pressure'],
                                    density=p['particle_density'])
            init_coagulation_indices(ws)

        # ── deposition profile ─────────────────────────────────────────
        xi_static = None
        if ws.is_los:
            wd_rate = wall_deposition_rate(ws)    # [1/s]
            gamma_c = max(float(np.mean(wd_rate)), 1e-30)
            ws.gamma0 = gamma_c
            xi_static = wd_rate / gamma_c         # dimensionless (0-1)

        # ── time axis ──────────────────────────────────────────────────
        n_t = max(2, int(round(p['t_end'] / max(p['dt_out'], 1.0))) + 1)
        t_samp = np.linspace(0.0, p['t_end'], n_t)

        xi_t = (np.tile(xi_static[:, None], (1, n_t))
                if xi_static is not None else None)

        # ── condensation / Kelvin growth rate ──────────────────────────
        if ws.is_con:
            if p['kelvin']:
                Rg    = 8.3144598               # J/(mol·K)
                M_c   = p['molar_mass_g_mol'] * 1e-3   # [kg/mol]
                sig_c = p['surface_tension_mN_m'] * 1e-3  # [N/m]
                rho_c = p['condensate_density']  # [kg/m³]
                T     = p['temperature']
                S     = p['supersaturation']
                # Kelvin equilibrium sat. ratio per bin
                Sk = np.exp(4.0 * M_c * sig_c / (rho_c * Rg * T * dp_m))
                GR_phys = GR_c * np.maximum(0.0, S - Sk)   # [m/s] per bin
                zeta_per_bin = GR_phys / GR_c               # dimensionless
                # Tile to (nbin, n_t) so simulate() sees per-bin profile
                zeta_t = np.tile(zeta_per_bin[:, None], (1, n_t))
            else:
                zeta_t = 1.0   # scalar: use full GR_c uniformly
        else:
            zeta_t = 0.0

        # ── nucleation profile ─────────────────────────────────────────
        if ws.is_nuc:
            if p['nuc_profile'] == 'gaussian':
                t_peak  = p['nuc_peak_min'] * 60.0
                t_width = max(p['nuc_width_min'] * 60.0, 1.0)
                j_t = np.exp(-0.5 * ((t_samp - t_peak) / t_width) ** 2)
            else:
                j_t = 1.0   # constant, scaled by J_c
        else:
            j_t = 0.0

        # ── initial distribution ───────────────────────────────────────
        N0_cm3 = _lognormal_cm3(dp_m * 1e9, p['N0_cm3'], p['Dg_nm'], p['sg'])
        N0_m3  = N0_cm3 * 1e6
        x0     = N0_m3 / x_c   # dimensionless

        # ── integrate ─────────────────────────────────────────────────
        X, _ = simulate(ws, x0, t_samp, zeta_t, j_t, xi_t)

        # X shape: (nbin, n_t) — dimensionless
        N_cm3 = X * x_c * 1e-6   # [#/cm³]

        self.finished.emit({
            'times':  t_samp,
            'dp_nm':  dp_m * 1e9,
            'N':      N_cm3,       # (nbin, n_t)
        })


# ─── main tab ─────────────────────────────────────────────────────────────────

class SimulationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.sim         = None   # GDESimulator or None
        self._gde_result = None   # dict from GDEDirectWorker or None
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)

        # ── scrollable left panel ──────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(320)
        scroll.setFrameShape(QFrame.NoFrame)

        left_inner = QWidget()
        left = QVBoxLayout(left_inner)
        left.setSpacing(6)
        scroll.setWidget(left_inner)

        # Backend selector
        grp_be = QGroupBox("Simulation Backend")
        bl = QVBoxLayout(grp_be)
        self.cmb_backend = QComboBox()
        self.cmb_backend.addItems([
            "Legacy (modules/simulation.py)",
            "Direct GDE (algo/gde.py)",
        ])
        self.cmb_backend.setCurrentIndex(1)
        bl.addWidget(self.cmb_backend)
        left.addWidget(grp_be)

        # Size grid
        grp_grid = QGroupBox("Size Grid")
        gl = QGridLayout(grp_grid)
        gl.addWidget(QLabel("Dp min (nm):"), 0, 0)
        self.spn_dpmin = QDoubleSpinBox()
        self.spn_dpmin.setRange(0.1, 100); self.spn_dpmin.setValue(3)
        gl.addWidget(self.spn_dpmin, 0, 1)
        gl.addWidget(QLabel("Dp max (nm):"), 1, 0)
        self.spn_dpmax = QDoubleSpinBox()
        self.spn_dpmax.setRange(10, 10000); self.spn_dpmax.setValue(1000)
        gl.addWidget(self.spn_dpmax, 1, 1)
        gl.addWidget(QLabel("# bins:"), 2, 0)
        self.spn_nbins = QSpinBox()
        self.spn_nbins.setRange(8, 512); self.spn_nbins.setValue(64)
        gl.addWidget(self.spn_nbins, 2, 1)
        left.addWidget(grp_grid)

        # Time settings
        grp_time = QGroupBox("Time Settings")
        tl = QGridLayout(grp_time)
        tl.addWidget(QLabel("Duration (s):"), 0, 0)
        self.spn_tend = QDoubleSpinBox()
        self.spn_tend.setRange(60, 86400); self.spn_tend.setValue(3600)
        tl.addWidget(self.spn_tend, 0, 1)
        tl.addWidget(QLabel("Output interval (s):"), 1, 0)
        self.spn_dtout = QDoubleSpinBox()
        self.spn_dtout.setRange(1, 3600); self.spn_dtout.setValue(60)
        tl.addWidget(self.spn_dtout, 1, 1)
        left.addWidget(grp_time)

        # Physical processes
        grp_proc = QGroupBox("Physical Processes")
        pl = QVBoxLayout(grp_proc)
        self.chk_coag = QCheckBox("Brownian coagulation")
        self.chk_coag.setChecked(True)
        self.chk_cond = QCheckBox("Condensation / growth")
        self.chk_nucl = QCheckBox("Nucleation")
        self.chk_dep  = QCheckBox("Wall deposition")
        for w in [self.chk_coag, self.chk_cond, self.chk_nucl, self.chk_dep]:
            pl.addWidget(w)
        left.addWidget(grp_proc)

        # Thermodynamics (used by direct GDE for coagulation coefficients)
        grp_thermo = QGroupBox("Thermodynamics")
        rl = QGridLayout(grp_thermo)
        rl.addWidget(QLabel("Temperature (K):"), 0, 0)
        self.spn_T = QDoubleSpinBox()
        self.spn_T.setRange(200, 400); self.spn_T.setValue(298.0)
        self.spn_T.setDecimals(1)
        rl.addWidget(self.spn_T, 0, 1)
        rl.addWidget(QLabel("Pressure (Pa):"), 1, 0)
        self.spn_P = QDoubleSpinBox()
        self.spn_P.setRange(50000, 110000); self.spn_P.setValue(101325)
        self.spn_P.setDecimals(0)
        rl.addWidget(self.spn_P, 1, 1)
        rl.addWidget(QLabel("Particle density (kg/m³):"), 2, 0)
        self.spn_rho = QDoubleSpinBox()
        self.spn_rho.setRange(100, 5000); self.spn_rho.setValue(1400)
        self.spn_rho.setDecimals(0)
        rl.addWidget(self.spn_rho, 2, 1)
        left.addWidget(grp_thermo)

        # Condensation / growth parameters
        grp_cond = QGroupBox("Condensation / Growth")
        cl = QGridLayout(grp_cond)
        cl.addWidget(QLabel("Growth rate (nm/h):"), 0, 0)
        self.spn_gr = QDoubleSpinBox()
        self.spn_gr.setRange(0.0, 1000.0); self.spn_gr.setValue(5.0)
        self.spn_gr.setDecimals(2)
        cl.addWidget(self.spn_gr, 0, 1)

        self.chk_kelvin = QCheckBox("Kelvin correction")
        cl.addWidget(self.chk_kelvin, 1, 0, 1, 2)

        # Kelvin parameters (always visible, only active when box checked)
        cl.addWidget(QLabel("Supersaturation S:"), 2, 0)
        self.spn_S = QDoubleSpinBox()
        self.spn_S.setRange(1.000, 10.0); self.spn_S.setValue(1.005)
        self.spn_S.setDecimals(4)
        cl.addWidget(self.spn_S, 2, 1)

        cl.addWidget(QLabel("Surface tension (mN/m):"), 3, 0)
        self.spn_sigma = QDoubleSpinBox()
        self.spn_sigma.setRange(1.0, 200.0); self.spn_sigma.setValue(72.0)
        self.spn_sigma.setDecimals(1)
        cl.addWidget(self.spn_sigma, 3, 1)

        cl.addWidget(QLabel("Condensate M (g/mol):"), 4, 0)
        self.spn_Mc = QDoubleSpinBox()
        self.spn_Mc.setRange(1.0, 500.0); self.spn_Mc.setValue(18.02)
        self.spn_Mc.setDecimals(2)
        cl.addWidget(self.spn_Mc, 4, 1)

        cl.addWidget(QLabel("Condensate density (kg/m³):"), 5, 0)
        self.spn_rho_c = QDoubleSpinBox()
        self.spn_rho_c.setRange(100, 5000); self.spn_rho_c.setValue(1000)
        self.spn_rho_c.setDecimals(0)
        cl.addWidget(self.spn_rho_c, 5, 1)
        left.addWidget(grp_cond)

        # Nucleation parameters
        grp_nuc = QGroupBox("Nucleation Rate")
        nl2 = QGridLayout(grp_nuc)
        nl2.addWidget(QLabel("Profile:"), 0, 0)
        self.cmb_nuc = QComboBox()
        self.cmb_nuc.addItems(["Constant", "Gaussian burst"])
        nl2.addWidget(self.cmb_nuc, 0, 1)

        nl2.addWidget(QLabel("Rate J (#/cm³/s):"), 1, 0)
        self.spn_J = QDoubleSpinBox()
        self.spn_J.setRange(0.0, 1e8); self.spn_J.setValue(1.0)
        self.spn_J.setDecimals(3)
        nl2.addWidget(self.spn_J, 1, 1)

        nl2.addWidget(QLabel("Peak time (min):"), 2, 0)
        self.spn_nuc_peak = QDoubleSpinBox()
        self.spn_nuc_peak.setRange(0, 1440); self.spn_nuc_peak.setValue(30)
        self.spn_nuc_peak.setDecimals(1)
        nl2.addWidget(self.spn_nuc_peak, 2, 1)

        nl2.addWidget(QLabel("Peak width (min):"), 3, 0)
        self.spn_nuc_width = QDoubleSpinBox()
        self.spn_nuc_width.setRange(0.1, 240); self.spn_nuc_width.setValue(10)
        self.spn_nuc_width.setDecimals(1)
        nl2.addWidget(self.spn_nuc_width, 3, 1)

        self.cmb_nuc.currentIndexChanged.connect(self._on_nuc_profile_changed)
        self._on_nuc_profile_changed(0)
        left.addWidget(grp_nuc)

        # Initial distribution
        grp_init = QGroupBox("Initial Distribution (log-normal)")
        il = QGridLayout(grp_init)
        il.addWidget(QLabel("N₀ [#/cm³]:"), 0, 0)
        self.spn_N0 = QDoubleSpinBox()
        self.spn_N0.setRange(0, 1e8); self.spn_N0.setValue(1e4)
        self.spn_N0.setDecimals(0)
        il.addWidget(self.spn_N0, 0, 1)
        il.addWidget(QLabel("Dg (nm):"), 1, 0)
        self.spn_Dg = QDoubleSpinBox()
        self.spn_Dg.setRange(1, 1000); self.spn_Dg.setValue(100)
        il.addWidget(self.spn_Dg, 1, 1)
        il.addWidget(QLabel("σg:"), 2, 0)
        self.spn_sg = QDoubleSpinBox()
        self.spn_sg.setRange(1.01, 4); self.spn_sg.setValue(1.8)
        self.spn_sg.setDecimals(2)
        il.addWidget(self.spn_sg, 2, 1)
        left.addWidget(grp_init)

        self.btn_run = QPushButton("▶  Run Simulation")
        self.btn_run.clicked.connect(self._run_sim)
        left.addWidget(self.btn_run)

        self.btn_save = QPushButton("Save results…")
        self.btn_save.clicked.connect(self._save_results)
        left.addWidget(self.btn_save)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(100)
        left.addWidget(self.log)
        left.addStretch()

        # ── right: plot tabs ───────────────────────────────────────────
        right_w = QWidget()
        right = QVBoxLayout(right_w)
        self.plot_tabs = QTabWidget()

        self.fig_heat = Figure(figsize=(9, 5), facecolor="#1e1e2e")
        self.canvas_heat = FigureCanvas(self.fig_heat)
        self.tb_heat = NavigationToolbar(self.canvas_heat, self)
        hw = QWidget(); hl = QVBoxLayout(hw)
        hl.addWidget(self.tb_heat); hl.addWidget(self.canvas_heat)
        self.plot_tabs.addTab(hw, "2D Heatmap")

        self.fig_N = Figure(figsize=(9, 4), facecolor="#1e1e2e")
        self.canvas_N = FigureCanvas(self.fig_N)
        self.tb_N = NavigationToolbar(self.canvas_N, self)
        nw = QWidget(); nl3 = QVBoxLayout(nw)
        nl3.addWidget(self.tb_N); nl3.addWidget(self.canvas_N)
        self.plot_tabs.addTab(nw, "Total N(t)")

        right.addWidget(self.plot_tabs)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(right_w)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter)

    def _on_nuc_profile_changed(self, index: int):
        is_gauss = (index == 1)
        self.spn_nuc_peak.setEnabled(is_gauss)
        self.spn_nuc_width.setEnabled(is_gauss)

    # ── simulation dispatch ────────────────────────────────────────────

    def _run_sim(self):
        if self.cmb_backend.currentIndex() == 0:
            self._run_legacy()
        else:
            self._run_direct()

    # ── legacy backend ─────────────────────────────────────────────────

    def _run_legacy(self):
        dp_grid = np.logspace(
            np.log10(self.spn_dpmin.value()),
            np.log10(self.spn_dpmax.value()),
            self.spn_nbins.value(),
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
            cfg.nucleation_source = lambda t: self.spn_J.value()
        if self.chk_dep.isChecked():
            from modules.simulation import gravitational_settling_velocity
            v = gravitational_settling_velocity(dp_grid, rho_p=1500)
            cfg.deposition_rate = v / 5.0

        self.sim = GDESimulator(cfg)
        N0 = _lognormal_cm3(dp_grid,
                            self.spn_N0.value(),
                            self.spn_Dg.value(),
                            self.spn_sg.value())

        self._set_running(True)
        self._log("Legacy backend: starting simulation…")

        self._worker = LegacySimWorker(self.sim, N0)
        self._worker.finished.connect(self._on_legacy_done)
        self._worker.error.connect(self._on_sim_error)
        self._worker.start()

    def _on_legacy_done(self, sim: GDESimulator):
        self.sim = sim
        self._set_running(False)
        self._log(
            f"Done. {sim.result_N.shape[0]} time steps × {sim.result_N.shape[1]} bins."
        )
        times, dp_grid, N = sim.get_heatmap_data()
        self._gde_result = {
            'times': times,
            'dp_nm': dp_grid,
            'N':     N.T,        # (nbin, n_t) to match direct GDE output
        }
        self._plot_results()
        if self.main_window:
            self.main_window.set_status("Simulation complete.")

    # ── direct GDE backend ─────────────────────────────────────────────

    def _run_direct(self):
        nuc_profile = ("gaussian" if self.cmb_nuc.currentIndex() == 1
                       else "constant")
        params = dict(
            dp_min_nm       = self.spn_dpmin.value(),
            dp_max_nm       = self.spn_dpmax.value(),
            n_bins          = self.spn_nbins.value(),
            t_end           = self.spn_tend.value(),
            dt_out          = self.spn_dtout.value(),
            coagulation     = self.chk_coag.isChecked(),
            condensation    = self.chk_cond.isChecked(),
            nucleation      = self.chk_nucl.isChecked(),
            deposition      = self.chk_dep.isChecked(),
            temperature     = self.spn_T.value(),
            pressure        = self.spn_P.value(),
            particle_density= self.spn_rho.value(),
            growth_rate_nm_h= self.spn_gr.value(),
            kelvin          = self.chk_kelvin.isChecked(),
            supersaturation = self.spn_S.value(),
            surface_tension_mN_m = self.spn_sigma.value(),
            molar_mass_g_mol     = self.spn_Mc.value(),
            condensate_density   = self.spn_rho_c.value(),
            nucleation_rate = self.spn_J.value(),
            nuc_profile     = nuc_profile,
            nuc_peak_min    = self.spn_nuc_peak.value(),
            nuc_width_min   = self.spn_nuc_width.value(),
            N0_cm3          = self.spn_N0.value(),
            Dg_nm           = self.spn_Dg.value(),
            sg              = self.spn_sg.value(),
        )

        self._set_running(True)
        self._log("Direct GDE backend: building AeroSys and running…")

        self._worker = GDEDirectWorker(params)
        self._worker.finished.connect(self._on_direct_done)
        self._worker.error.connect(self._on_sim_error)
        self._worker.start()

    def _on_direct_done(self, result: dict):
        self._gde_result = result
        self._set_running(False)
        n_t  = result['N'].shape[1]
        nbin = result['N'].shape[0]
        self._log(f"Done. {n_t} time steps × {nbin} bins.")
        self._plot_results()
        if self.main_window:
            self.main_window.set_status("Simulation complete.")

    # ── shared error ───────────────────────────────────────────────────

    def _on_sim_error(self, msg: str):
        self._set_running(False)
        QMessageBox.critical(self, "Simulation error", msg)

    # ── plotting ───────────────────────────────────────────────────────

    def _plot_results(self):
        if self._gde_result is None:
            return
        times = self._gde_result['times']     # (n_t,)
        dp_nm = self._gde_result['dp_nm']     # (nbin,)
        N     = self._gde_result['N']          # (nbin, n_t)

        t_min = times / 60.0

        # Heatmap
        self.fig_heat.clear()
        ax = self.fig_heat.add_subplot(111)
        ax.set_facecolor("#181825")
        im = ax.pcolormesh(t_min, dp_nm, N,
                           cmap="plasma", shading="auto")
        ax.set_yscale("log")
        cb = self.fig_heat.colorbar(im, ax=ax)
        cb.ax.tick_params(colors="#cdd6f4")
        cb.set_label("dN/dlogDp [#/cm³]", color="#cdd6f4")
        ax.set_xlabel("Time (min)", color="#cdd6f4")
        ax.set_ylabel("Dp (nm)", color="#cdd6f4")
        ax.set_title("Simulated Size Distribution Evolution", color="#cdd6f4")
        ax.tick_params(colors="#cdd6f4")
        for s in ax.spines.values():
            s.set_edgecolor("#45475a")
        self.fig_heat.tight_layout()
        self.canvas_heat.draw()

        # Total N(t)
        Ntot = N.sum(axis=0)   # sum over bins
        self.fig_N.clear()
        ax2 = self.fig_N.add_subplot(111)
        ax2.set_facecolor("#181825")
        ax2.plot(t_min, Ntot, color="#89b4fa", linewidth=2)
        ax2.fill_between(t_min, Ntot, alpha=0.2, color="#89b4fa")
        ax2.set_xlabel("Time (min)", color="#cdd6f4")
        ax2.set_ylabel("Total N [#/cm³]", color="#cdd6f4")
        ax2.set_title("Total number concentration vs. time", color="#cdd6f4")
        ax2.tick_params(colors="#cdd6f4")
        for s in ax2.spines.values():
            s.set_edgecolor("#45475a")
        self.fig_N.tight_layout()
        self.canvas_N.draw()

    # ── save ───────────────────────────────────────────────────────────

    def _save_results(self):
        if self._gde_result is None:
            QMessageBox.warning(self, "Save", "Run a simulation first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save simulation", "sim_result.csv", "CSV (*.csv)")
        if not path:
            return
        import pandas as pd
        times = self._gde_result['times']
        dp_nm = self._gde_result['dp_nm']
        N     = self._gde_result['N']   # (nbin, n_t)
        df = pd.DataFrame(
            N.T,
            index=pd.Index(times, name="time_s"),
            columns=pd.Index(dp_nm, name="Dp_nm"),
        )
        df.to_csv(path)
        self._log(f"Saved → {path}")

    # ── helpers ────────────────────────────────────────────────────────

    def _set_running(self, running: bool):
        self.btn_run.setEnabled(not running)
        self.btn_run.setText("Running…" if running else "▶  Run Simulation")

    def _log(self, msg: str):
        self.log.append(msg)

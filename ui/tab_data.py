"""
ui/tab_data.py
--------------
Tab 1: Data loading, visualisation (heatmap + time series), manipulation, export.

Supports two loading modes selected via the "File Format" combo:

  Generic (CSV / Excel)
      Original behaviour: open any tabular file, then manually assign
      which column is the time axis and which are the diameter/count bins.

  TSI SMPS (xlsx)
      Specialised loader for TSI AIM software exports.  The file structure
      is parsed automatically; the sheet to read is selected via a combo box
      that appears after the file is opened.  Instrument metadata extracted
      from the file can be forwarded to the Measurement Model tab with the
      "Seed Model Tab" button.
"""

import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QComboBox, QLabel, QFileDialog,
    QGroupBox, QCheckBox, QDoubleSpinBox,
    QSplitter, QListWidget, QTextEdit,
    QTabWidget, QMessageBox, QAbstractItemView,
)
from PyQt5.QtCore import Qt

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from modules.data_model import AerosolDataset


class DataTab(QWidget):
    def __init__(self, dataset: AerosolDataset, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.main_window = parent
        self._tsi_result = None   # last TSIResult (for seeding the model tab)
        self._build_ui()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QHBoxLayout(self)

        # ---- Left panel: controls ----
        left = QVBoxLayout()
        left.setSpacing(8)
        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setFixedWidth(290)

        # ── File format selector ──────────────────────────────────────
        grp_fmt = QGroupBox("File Format")
        fmt_l = QVBoxLayout(grp_fmt)
        self.cmb_format = QComboBox()
        self.cmb_format.addItems(["Generic (CSV / Excel)", "TSI SMPS (xlsx)"])
        self.cmb_format.currentIndexChanged.connect(self._on_format_changed)
        fmt_l.addWidget(self.cmb_format)
        left.addWidget(grp_fmt)

        # ── Data file – stacked for Generic / TSI ────────────────────
        grp_file = QGroupBox("Data File")
        file_l = QVBoxLayout(grp_file)

        # Generic open button
        self.btn_open_generic = QPushButton("Open CSV / Excel…")
        self.btn_open_generic.clicked.connect(self._open_generic)
        file_l.addWidget(self.btn_open_generic)

        # TSI open button + sheet selector
        self.btn_open_tsi = QPushButton("Open TSI SMPS xlsx…")
        self.btn_open_tsi.clicked.connect(self._open_tsi)
        self.btn_open_tsi.setVisible(False)
        file_l.addWidget(self.btn_open_tsi)

        self._lbl_sheet = QLabel("Sheet:")
        self._lbl_sheet.setVisible(False)
        self.cmb_sheet = QComboBox()
        self.cmb_sheet.setVisible(False)
        self.cmb_sheet.currentIndexChanged.connect(self._on_sheet_changed)
        sheet_row = QHBoxLayout()
        sheet_row.addWidget(self._lbl_sheet)
        sheet_row.addWidget(self.cmb_sheet)
        file_l.addLayout(sheet_row)

        self.lbl_file = QLabel("No file loaded.")
        self.lbl_file.setWordWrap(True)
        file_l.addWidget(self.lbl_file)

        # TSI-only: seed model button
        self.btn_seed_model = QPushButton("Seed Measurement Model →")
        self.btn_seed_model.setVisible(False)
        self.btn_seed_model.setToolTip(
            "Transfer instrument parameters extracted from the TSI file\n"
            "to the Measurement Model tab spin-boxes."
        )
        self.btn_seed_model.clicked.connect(self._seed_model_tab)
        file_l.addWidget(self.btn_seed_model)

        left.addWidget(grp_file)

        # ── Column assignment (Generic mode only) ─────────────────────
        self.grp_cols = QGroupBox("Column Roles")
        cl = QGridLayout(self.grp_cols)
        cl.addWidget(QLabel("Time column:"), 0, 0)
        self.cmb_time = QComboBox(); cl.addWidget(self.cmb_time, 0, 1)
        cl.addWidget(QLabel("Count columns:"), 1, 0)
        self.lst_count_cols = QListWidget()
        self.lst_count_cols.setSelectionMode(QAbstractItemView.MultiSelection)
        self.lst_count_cols.setMaximumHeight(100)
        cl.addWidget(self.lst_count_cols, 2, 0, 1, 2)
        self.btn_apply_cols = QPushButton("Apply column mapping")
        self.btn_apply_cols.clicked.connect(self._apply_column_mapping)
        cl.addWidget(self.btn_apply_cols, 3, 0, 1, 2)
        left.addWidget(self.grp_cols)

        # ── Visualisation options ─────────────────────────────────────
        grp_viz = QGroupBox("Visualisation")
        vl = QGridLayout(grp_viz)
        vl.addWidget(QLabel("Data source:"), 0, 0)
        self.cmb_data_src = QComboBox()
        self.cmb_data_src.addItems(["dN/dlogDp (concentration)", "Raw counts"])
        vl.addWidget(self.cmb_data_src, 0, 1)
        vl.addWidget(QLabel("Colour map:"), 1, 0)
        self.cmb_cmap = QComboBox()
        self.cmb_cmap.addItems(["viridis", "plasma", "inferno", "magma",
                                 "jet", "turbo", "RdYlBu_r"])
        vl.addWidget(self.cmb_cmap, 1, 1)
        self.chk_log_color = QCheckBox("Log colour scale")
        self.chk_log_color.setChecked(True)
        vl.addWidget(self.chk_log_color, 2, 0, 1, 2)
        self.chk_log_y = QCheckBox("Log Y axis (diameter)")
        self.chk_log_y.setChecked(True)
        vl.addWidget(self.chk_log_y, 3, 0, 1, 2)
        self.btn_plot_heat = QPushButton("Plot 2D Heatmap")
        self.btn_plot_heat.clicked.connect(self._plot_heatmap)
        vl.addWidget(self.btn_plot_heat, 4, 0, 1, 2)
        left.addWidget(grp_viz)

        # ── 1D plot ───────────────────────────────────────────────────
        grp_ts = QGroupBox("1D Plot")
        tsl = QGridLayout(grp_ts)

        tsl.addWidget(QLabel("Mode:"), 0, 0)
        self.cmb_ts_mode = QComboBox()
        self.cmb_ts_mode.addItems([
            "Sum over Dp range vs. scan",
            "Size distribution (one scan)",
            "Single Dp vs. scan",
        ])
        self.cmb_ts_mode.currentIndexChanged.connect(self._on_ts_mode_changed)
        tsl.addWidget(self.cmb_ts_mode, 0, 1)

        # Row 1 — label + widget A
        self._lbl_ts_a = QLabel("Diam min (nm):")
        tsl.addWidget(self._lbl_ts_a, 1, 0)
        self.spn_d_min = QDoubleSpinBox()
        self.spn_d_min.setRange(1, 1e6); self.spn_d_min.setValue(10)
        tsl.addWidget(self.spn_d_min, 1, 1)

        # Row 2 — label + widget B (hidden for some modes)
        self._lbl_ts_b = QLabel("Diam max (nm):")
        tsl.addWidget(self._lbl_ts_b, 2, 0)
        self.spn_d_max = QDoubleSpinBox()
        self.spn_d_max.setRange(1, 1e6); self.spn_d_max.setValue(1000)
        tsl.addWidget(self.spn_d_max, 2, 1)

        self.btn_plot_ts = QPushButton("Plot")
        self.btn_plot_ts.clicked.connect(self._plot_time_series)
        tsl.addWidget(self.btn_plot_ts, 3, 0, 1, 2)
        left.addWidget(grp_ts)

        # ── Export ────────────────────────────────────────────────────
        grp_exp = QGroupBox("Export Subset")
        el = QVBoxLayout(grp_exp)
        self.lst_export_cols = QListWidget()
        self.lst_export_cols.setSelectionMode(QAbstractItemView.MultiSelection)
        self.lst_export_cols.setMaximumHeight(80)
        el.addWidget(QLabel("Select columns to export:"))
        el.addWidget(self.lst_export_cols)
        self.btn_export = QPushButton("Save subset…")
        self.btn_export.clicked.connect(self._export_subset)
        el.addWidget(self.btn_export)
        left.addWidget(grp_exp)

        left.addStretch()

        # ---- Right panel: plots ----
        right = QVBoxLayout()
        right_w = QWidget()
        right_w.setLayout(right)

        self.plot_tabs = QTabWidget()

        # Heatmap figure
        self.fig_heat = Figure(figsize=(9, 5), facecolor="#1e1e2e")
        self.canvas_heat = FigureCanvas(self.fig_heat)
        self.toolbar_heat = NavigationToolbar(self.canvas_heat, self)
        heat_w = QWidget()
        heat_l = QVBoxLayout(heat_w)
        heat_l.addWidget(self.toolbar_heat)
        heat_l.addWidget(self.canvas_heat)
        self.plot_tabs.addTab(heat_w, "2D Heatmap")

        # Time series figure
        self.fig_ts = Figure(figsize=(9, 4), facecolor="#1e1e2e")
        self.canvas_ts = FigureCanvas(self.fig_ts)
        self.toolbar_ts = NavigationToolbar(self.canvas_ts, self)
        ts_w = QWidget()
        ts_l = QVBoxLayout(ts_w)
        ts_l.addWidget(self.toolbar_ts)
        ts_l.addWidget(self.canvas_ts)
        self.plot_tabs.addTab(ts_w, "1D Time Series")

        # Log / summary
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)

        right.addWidget(self.plot_tabs)
        right.addWidget(QLabel("Log:"))
        right.addWidget(self.log)

        # Combine
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_w)
        splitter.addWidget(right_w)
        splitter.setStretchFactor(1, 3)

        root.addWidget(splitter)

    # ------------------------------------------------------------------
    # Format switching
    # ------------------------------------------------------------------

    def _on_ts_mode_changed(self, index: int):
        """Show/hide the correct parameter rows for the selected 1D plot mode."""
        # 0 = Sum over Dp range  →  Diam min + Diam max
        # 1 = Size distribution  →  Scan index
        # 2 = Single Dp vs scan  →  Diameter (nm)
        labels = ["Diam min (nm):", "Scan index:", "Diameter (nm):"]
        self._lbl_ts_a.setText(labels[index])

        show_b = (index == 0)
        self._lbl_ts_b.setVisible(show_b)
        self.spn_d_max.setVisible(show_b)

        if index == 1:          # scan index: integer steps, reset range after data loaded
            self.spn_d_min.setDecimals(0)
            self.spn_d_min.setRange(0, 9999)
            self.spn_d_min.setValue(0)
            self.spn_d_min.setSingleStep(1)
        else:
            self.spn_d_min.setDecimals(2)
            self.spn_d_min.setRange(1, 1e6)
            self.spn_d_min.setSingleStep(1.0)
            if index == 0:
                self.spn_d_min.setValue(10)
            else:
                self.spn_d_min.setValue(100)

    def _on_format_changed(self, index: int):
        is_tsi = (index == 1)
        self.btn_open_generic.setVisible(not is_tsi)
        self.btn_open_tsi.setVisible(is_tsi)
        self._lbl_sheet.setVisible(False)
        self.cmb_sheet.setVisible(False)
        self.btn_seed_model.setVisible(False)
        self.grp_cols.setVisible(not is_tsi)

    # ------------------------------------------------------------------
    # Generic loading (original behaviour)
    # ------------------------------------------------------------------

    def open_file_dialog(self):
        """Called by the main-window File menu (always uses generic mode)."""
        self._open_generic()

    def _open_generic(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open data file", "",
            "Data files (*.csv *.xlsx *.xls);;All files (*)"
        )
        if not path:
            return
        try:
            self.dataset.load(path)
            self.lbl_file.setText(path.split("/")[-1])
            self._populate_column_lists()
            self._log(f"Loaded: {path}\nColumns: {', '.join(self.dataset.columns)}")
            if self.main_window:
                self.main_window.set_status(f"Loaded {path}")
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))

    def _populate_column_lists(self):
        cols = self.dataset.columns
        self.cmb_time.clear()
        self.cmb_time.addItems(cols)
        self.lst_count_cols.clear()
        for c in cols:
            self.lst_count_cols.addItem(c)
        self.lst_export_cols.clear()
        for c in cols:
            self.lst_export_cols.addItem(c)

    def _apply_column_mapping(self):
        time_col = self.cmb_time.currentText()
        count_cols = [item.text() for item in self.lst_count_cols.selectedItems()]
        if not count_cols:
            QMessageBox.warning(self, "Column mapping", "Select at least one count column.")
            return
        try:
            self.dataset.set_column_roles(time_col, count_cols)
            self._log(self.dataset.summary())
            if self.main_window:
                self.main_window.set_status(
                    f"Columns mapped: {len(count_cols)} count bins, time='{time_col}'"
                )
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # ------------------------------------------------------------------
    # TSI loading
    # ------------------------------------------------------------------

    def _open_tsi(self):
        from modules.tsi_loader import list_sheets

        path, _ = QFileDialog.getOpenFileName(
            self, "Open TSI SMPS file", "",
            "TSI Excel files (*.xlsx *.xls);;All files (*)"
        )
        if not path:
            return

        # Populate sheet selector
        try:
            sheets = list_sheets(path)
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            return

        self._tsi_path = path
        self.cmb_sheet.blockSignals(True)
        self.cmb_sheet.clear()
        self.cmb_sheet.addItems(sheets)
        self.cmb_sheet.blockSignals(False)

        self._lbl_sheet.setVisible(True)
        self.cmb_sheet.setVisible(True)
        self.lbl_file.setText(path.split("/")[-1])

        # Load first sheet immediately
        self._load_tsi_sheet(path, self.cmb_sheet.currentText())

    def _on_sheet_changed(self, _sheet_index: int):
        if hasattr(self, '_tsi_path') and self._tsi_path:
            self._load_tsi_sheet(self._tsi_path, self.cmb_sheet.currentText())

    def _load_tsi_sheet(self, path: str, sheet_name: str):
        try:
            tsi = self.dataset.load_tsi(path, sheet_name)
            self._tsi_result = tsi
        except Exception as e:
            QMessageBox.critical(self, "TSI load error", str(e))
            return

        # Auto-populate export column list from dataset columns
        self.lst_export_cols.clear()
        for c in self.dataset.columns:
            self.lst_export_cols.addItem(c)

        # Show seed button
        self.btn_seed_model.setVisible(True)

        # Log instrument summary
        self._log(f"TSI file: {path.split('/')[-1]}  [sheet: {sheet_name}]")
        self._log(tsi.summary())

        if self.main_window:
            self.main_window.set_status(
                f"TSI loaded — {tsi.n_scans} scans, "
                f"{len(tsi.diameters_dn)} dp bins  [{sheet_name}]"
            )

    # ------------------------------------------------------------------
    # Seed model tab
    # ------------------------------------------------------------------

    def _seed_model_tab(self):
        if self._tsi_result is None:
            return
        if self.main_window is None:
            return
        hints = self._tsi_result.instrument_hints()
        try:
            self.main_window.tab_model.seed_from_tsi(hints)
            self.main_window.tabs.setCurrentWidget(self.main_window.tab_model)
            self._log("Instrument parameters forwarded to Measurement Model tab.")
        except Exception as e:
            QMessageBox.critical(self, "Seed error", str(e))

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def _plot_heatmap(self):
        use_raw = self.cmb_data_src.currentIndex() == 1

        if use_raw:
            if self.dataset.raw_counts is None:
                QMessageBox.warning(self, "No raw data",
                                    "Raw counts are not available for this dataset.\n"
                                    "Load a TSI file that contains raw count data.")
                return
            self._plot_heatmap_raw()
        else:
            if self.dataset.count_matrix is None:
                QMessageBox.warning(self, "No data",
                                    "Load data and apply column mapping first.")
                return
            self._plot_heatmap_dn()

    def _plot_heatmap_dn(self):
        """2-D heatmap of the dN/dlogDp concentration grid."""
        C = self.dataset.get_counts(log_scale=self.chk_log_color.isChecked())
        times = np.arange(self.dataset.n_scans)
        dp = self.dataset.diameters

        self.fig_heat.clear()
        ax = self.fig_heat.add_subplot(111)
        ax.set_facecolor("#181825")

        T, D = np.meshgrid(times, dp)
        mesh = ax.pcolormesh(T, D, C.T,
                              cmap=self.cmb_cmap.currentText(),
                              shading="auto")
        cbar = self.fig_heat.colorbar(mesh, ax=ax)
        cbar.ax.yaxis.label.set_color("#cdd6f4")
        cbar.ax.tick_params(colors="#cdd6f4")
        clabel = "log₁₀(dN/dlogDp)" if self.chk_log_color.isChecked() else "dN/dlogDp"
        cbar.set_label(clabel, color="#cdd6f4")

        if self.chk_log_y.isChecked():
            ax.set_yscale("log")
        ax.set_xlabel("Scan index", color="#cdd6f4")
        ax.set_ylabel("Diameter (nm)", color="#cdd6f4")
        ax.set_title("dN/dlogDp – 2D Heatmap", color="#cdd6f4")
        ax.tick_params(colors="#cdd6f4")
        for spine in ax.spines.values():
            spine.set_edgecolor("#45475a")

        self.fig_heat.tight_layout()
        self.canvas_heat.draw()
        self.plot_tabs.setCurrentIndex(0)

    def _plot_heatmap_raw(self):
        """
        2-D heatmap of raw counts.

        X axis  : scan index
        Y axis  : diameter [nm] selected at each time step within the scan
                  (taken from the first scan; the voltage programme is identical
                  for all scans)
        Colour  : raw counts (optionally log-scaled)
        """
        C = self.dataset.raw_counts.copy()   # (n_scans, n_times)
        dp = self.dataset.raw_diameters[0]   # diameter per time step [nm]

        if self.chk_log_color.isChecked():
            C = C.astype(float)
            C[C <= 0] = np.nan
            C = np.log10(C)

        scans = np.arange(self.dataset.raw_counts.shape[0])

        self.fig_heat.clear()
        ax = self.fig_heat.add_subplot(111)
        ax.set_facecolor("#181825")

        # pcolormesh: x = scan index, y = diameter per time step, z = counts
        T, D = np.meshgrid(scans, dp)
        mesh = ax.pcolormesh(T, D, C.T,
                              cmap=self.cmb_cmap.currentText(),
                              shading="auto")
        cbar = self.fig_heat.colorbar(mesh, ax=ax)
        cbar.ax.yaxis.label.set_color("#cdd6f4")
        cbar.ax.tick_params(colors="#cdd6f4")
        clabel = "log₁₀(counts)" if self.chk_log_color.isChecked() else "Counts"
        cbar.set_label(clabel, color="#cdd6f4")

        if self.chk_log_y.isChecked():
            ax.set_yscale("log")
        ax.set_xlabel("Scan index", color="#cdd6f4")
        ax.set_ylabel("Diameter (nm)", color="#cdd6f4")
        ax.set_title("Raw counts – 2D Heatmap", color="#cdd6f4")
        ax.tick_params(colors="#cdd6f4")
        for spine in ax.spines.values():
            spine.set_edgecolor("#45475a")

        self.fig_heat.tight_layout()
        self.canvas_heat.draw()
        self.plot_tabs.setCurrentIndex(0)

    def _plot_time_series(self):
        mode    = self.cmb_ts_mode.currentIndex()
        use_raw = self.cmb_data_src.currentIndex() == 1
        {0: self._plot_ts_summed, 1: self._plot_ts_scan_slice,
         2: self._plot_ts_dp_slice}[mode](use_raw)

    # -- mode 0: sum over Dp range vs. scan --------------------------------

    def _plot_ts_summed(self, use_raw: bool):
        d_min = self.spn_d_min.value()
        d_max = self.spn_d_max.value()

        if use_raw:
            if self.dataset.raw_counts is None:
                QMessageBox.warning(self, "No raw data",
                                    "Raw counts are not available for this dataset.")
                return
            dp   = self.dataset.raw_diameters
            mask = (dp >= d_min) & (dp <= d_max)
            ts   = np.where(mask, self.dataset.raw_counts, 0).sum(axis=1)
            ylabel = "Total raw counts"
            title  = f"Raw counts {d_min:.0f}–{d_max:.0f} nm vs. scan"
        else:
            if self.dataset.count_matrix is None:
                QMessageBox.warning(self, "No data",
                                    "Load data and apply column mapping first.")
                return
            ts     = self.dataset.sum_over_diameter_range(d_min, d_max)
            ylabel = "Total dN/dlogDp"
            title  = f"dN/dlogDp {d_min:.0f}–{d_max:.0f} nm vs. scan"

        self._draw_ts_scan(np.arange(len(ts)), ts, "Scan index", ylabel, title)

    # -- mode 1: full size distribution for one scan -----------------------

    def _plot_ts_scan_slice(self, use_raw: bool):
        scan_idx = int(self.spn_d_min.value())

        if use_raw:
            if self.dataset.raw_counts is None:
                QMessageBox.warning(self, "No raw data",
                                    "Raw counts are not available for this dataset.")
                return
            n = self.dataset.raw_counts.shape[0]
            if not (0 <= scan_idx < n):
                QMessageBox.warning(self, "Index out of range",
                                    f"Scan index must be 0–{n - 1}.")
                return
            dp  = self.dataset.raw_diameters[scan_idx]   # (n_times,)
            y   = self.dataset.raw_counts[scan_idx]       # (n_times,)
            # sort by diameter so the line makes sense
            order = np.argsort(dp)
            dp, y = dp[order], y[order]
            ylabel = "Raw counts"
            title  = f"Raw counts – scan {scan_idx}"
        else:
            if self.dataset.count_matrix is None:
                QMessageBox.warning(self, "No data",
                                    "Load data and apply column mapping first.")
                return
            n = self.dataset.n_scans
            if not (0 <= scan_idx < n):
                QMessageBox.warning(self, "Index out of range",
                                    f"Scan index must be 0–{n - 1}.")
                return
            dp  = self.dataset.diameters
            y   = self.dataset.count_matrix[scan_idx]
            ylabel = "dN/dlogDp"
            title  = f"Size distribution – scan {scan_idx}"

        self.fig_ts.clear()
        ax = self.fig_ts.add_subplot(111)
        ax.set_facecolor("#181825")
        ax.plot(dp, y, color="#a6e3a1", linewidth=1.5)
        ax.fill_between(dp, y, alpha=0.2, color="#a6e3a1")
        if self.chk_log_y.isChecked():
            ax.set_xscale("log")
        ax.set_xlabel("Diameter (nm)", color="#cdd6f4")
        ax.set_ylabel(ylabel, color="#cdd6f4")
        ax.set_title(title, color="#cdd6f4")
        ax.tick_params(colors="#cdd6f4")
        for spine in ax.spines.values():
            spine.set_edgecolor("#45475a")
        self.fig_ts.tight_layout()
        self.canvas_ts.draw()
        self.plot_tabs.setCurrentIndex(1)

    # -- mode 2: single diameter vs. scan ----------------------------------

    def _plot_ts_dp_slice(self, use_raw: bool):
        dp_target = self.spn_d_min.value()

        if use_raw:
            if self.dataset.raw_counts is None:
                QMessageBox.warning(self, "No raw data",
                                    "Raw counts are not available for this dataset.")
                return
            # For each scan pick the time step whose diameter is closest to dp_target
            dp_arr = self.dataset.raw_diameters   # (n_scans, n_times)
            idx    = np.argmin(np.abs(dp_arr - dp_target), axis=1)
            y      = self.dataset.raw_counts[np.arange(len(idx)), idx]
            dp_actual = dp_arr[np.arange(len(idx)), idx].mean()
            ylabel = "Raw counts"
            title  = f"Raw counts at Dp ≈ {dp_actual:.1f} nm vs. scan"
        else:
            if self.dataset.count_matrix is None:
                QMessageBox.warning(self, "No data",
                                    "Load data and apply column mapping first.")
                return
            idx       = int(np.argmin(np.abs(self.dataset.diameters - dp_target)))
            dp_actual = self.dataset.diameters[idx]
            y         = self.dataset.count_matrix[:, idx]
            ylabel    = "dN/dlogDp"
            title     = f"dN/dlogDp at Dp = {dp_actual:.1f} nm vs. scan"

        self._draw_ts_scan(np.arange(len(y)), y, "Scan index", ylabel, title)

    # -- shared draw helper ------------------------------------------------

    def _draw_ts_scan(self, x, y, xlabel, ylabel, title):
        self.fig_ts.clear()
        ax = self.fig_ts.add_subplot(111)
        ax.set_facecolor("#181825")
        ax.plot(x, y, color="#89b4fa", linewidth=1.5)
        ax.fill_between(x, y, alpha=0.25, color="#89b4fa")
        ax.set_xlabel(xlabel, color="#cdd6f4")
        ax.set_ylabel(ylabel, color="#cdd6f4")
        ax.set_title(title, color="#cdd6f4")
        ax.tick_params(colors="#cdd6f4")
        for spine in ax.spines.values():
            spine.set_edgecolor("#45475a")
        self.fig_ts.tight_layout()
        self.canvas_ts.draw()
        self.plot_tabs.setCurrentIndex(1)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_subset(self):
        cols = [item.text() for item in self.lst_export_cols.selectedItems()]
        if not cols:
            QMessageBox.warning(self, "Export", "Select at least one column.")
            return
        path, filt = QFileDialog.getSaveFileName(
            self, "Save subset", "",
            "CSV (*.csv);;Excel (*.xlsx)"
        )
        if not path:
            return
        fmt = "xlsx" if path.endswith(".xlsx") else "csv"
        try:
            self.dataset.save_subset(path, columns=cols, fmt=fmt)
            self._log(f"Exported {len(cols)} columns → {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str):
        self.log.append(msg)

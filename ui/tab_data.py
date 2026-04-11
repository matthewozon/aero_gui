"""
ui/tab_data.py
--------------
Tab 1: Data loading, visualisation (heatmap + time series), manipulation, export.
"""

import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QComboBox, QLabel, QFileDialog,
    QGroupBox, QCheckBox, QLineEdit, QDoubleSpinBox,
    QSplitter, QListWidget, QListWidgetItem, QTextEdit,
    QTabWidget, QMessageBox, QAbstractItemView
)
from PyQt5.QtCore import Qt

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from modules.data_model import AerosolDataset


class DataTab(QWidget):
    def __init__(self, dataset: AerosolDataset, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.main_window = parent
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
        left_w.setFixedWidth(280)

        # File loading
        grp_file = QGroupBox("Data File")
        fl = QVBoxLayout(grp_file)
        self.btn_open = QPushButton("Open CSV / Excel…")
        self.btn_open.clicked.connect(self.open_file_dialog)
        self.lbl_file = QLabel("No file loaded.")
        self.lbl_file.setWordWrap(True)
        fl.addWidget(self.btn_open)
        fl.addWidget(self.lbl_file)
        left.addWidget(grp_file)

        # Column assignment
        grp_cols = QGroupBox("Column Roles")
        cl = QGridLayout(grp_cols)
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
        left.addWidget(grp_cols)

        # Visualisation options
        grp_viz = QGroupBox("Visualisation")
        vl = QGridLayout(grp_viz)
        vl.addWidget(QLabel("Colour map:"), 0, 0)
        self.cmb_cmap = QComboBox()
        self.cmb_cmap.addItems(["viridis", "plasma", "inferno", "magma",
                                 "jet", "turbo", "RdYlBu_r"])
        vl.addWidget(self.cmb_cmap, 0, 1)
        self.chk_log_color = QCheckBox("Log colour scale")
        self.chk_log_color.setChecked(True)
        vl.addWidget(self.chk_log_color, 1, 0, 1, 2)
        self.chk_log_y = QCheckBox("Log Y axis (diameter)")
        self.chk_log_y.setChecked(True)
        vl.addWidget(self.chk_log_y, 2, 0, 1, 2)
        self.btn_plot_heat = QPushButton("Plot 2D Heatmap")
        self.btn_plot_heat.clicked.connect(self._plot_heatmap)
        vl.addWidget(self.btn_plot_heat, 3, 0, 1, 2)
        left.addWidget(grp_viz)

        # 1D time series
        grp_ts = QGroupBox("1D Time Series")
        tsl = QGridLayout(grp_ts)
        tsl.addWidget(QLabel("Diam min (nm):"), 0, 0)
        self.spn_d_min = QDoubleSpinBox()
        self.spn_d_min.setRange(1, 1e6); self.spn_d_min.setValue(10)
        tsl.addWidget(self.spn_d_min, 0, 1)
        tsl.addWidget(QLabel("Diam max (nm):"), 1, 0)
        self.spn_d_max = QDoubleSpinBox()
        self.spn_d_max.setRange(1, 1e6); self.spn_d_max.setValue(1000)
        tsl.addWidget(self.spn_d_max, 1, 1)
        self.btn_plot_ts = QPushButton("Plot Time Series")
        self.btn_plot_ts.clicked.connect(self._plot_time_series)
        tsl.addWidget(self.btn_plot_ts, 2, 0, 1, 2)
        left.addWidget(grp_ts)

        # Export
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

        # Two plot areas in a sub-tab
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
        self.log.setMaximumHeight(120)

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
    # Slots
    # ------------------------------------------------------------------

    def open_file_dialog(self):
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
        # Time combo
        self.cmb_time.clear()
        self.cmb_time.addItems(cols)
        # Count column list
        self.lst_count_cols.clear()
        for c in cols:
            self.lst_count_cols.addItem(c)
        # Export columns
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

    def _plot_heatmap(self):
        if self.dataset.count_matrix is None:
            QMessageBox.warning(self, "No data", "Apply column mapping first.")
            return

        C = self.dataset.get_counts(log_scale=self.chk_log_color.isChecked())
        times = np.arange(self.dataset.n_scans)  # fallback: scan index
        dp = self.dataset.diameters

        self.fig_heat.clear()
        ax = self.fig_heat.add_subplot(111)
        ax.set_facecolor("#181825")

        # Pcolormesh: x=time, y=diameter, z=counts
        T, D = np.meshgrid(times, dp)
        mesh = ax.pcolormesh(T, D, C.T,
                              cmap=self.cmb_cmap.currentText(),
                              shading="auto")
        cbar = self.fig_heat.colorbar(mesh, ax=ax)
        cbar.ax.yaxis.label.set_color("#cdd6f4")
        cbar.ax.tick_params(colors="#cdd6f4")
        label = "log₁₀(counts)" if self.chk_log_color.isChecked() else "Counts"
        cbar.set_label(label, color="#cdd6f4")

        if self.chk_log_y.isChecked():
            ax.set_yscale("log")

        ax.set_xlabel("Scan index", color="#cdd6f4")
        ax.set_ylabel("Diameter (nm)", color="#cdd6f4")
        ax.set_title("Particle Size Distribution – 2D Heatmap", color="#cdd6f4")
        ax.tick_params(colors="#cdd6f4")
        for spine in ax.spines.values():
            spine.set_edgecolor("#45475a")

        self.fig_heat.tight_layout()
        self.canvas_heat.draw()
        self.plot_tabs.setCurrentIndex(0)

    def _plot_time_series(self):
        if self.dataset.count_matrix is None:
            QMessageBox.warning(self, "No data", "Apply column mapping first.")
            return

        d_min = self.spn_d_min.value()
        d_max = self.spn_d_max.value()
        ts = self.dataset.sum_over_diameter_range(d_min, d_max)
        times = np.arange(self.dataset.n_scans)

        self.fig_ts.clear()
        ax = self.fig_ts.add_subplot(111)
        ax.set_facecolor("#181825")
        ax.plot(times, ts, color="#89b4fa", linewidth=1.5)
        ax.fill_between(times, ts, alpha=0.25, color="#89b4fa")
        ax.set_xlabel("Scan index", color="#cdd6f4")
        ax.set_ylabel("Total counts", color="#cdd6f4")
        ax.set_title(
            f"Total counts {d_min:.0f}–{d_max:.0f} nm vs. scan index",
            color="#cdd6f4"
        )
        ax.tick_params(colors="#cdd6f4")
        for spine in ax.spines.values():
            spine.set_edgecolor("#45475a")

        self.fig_ts.tight_layout()
        self.canvas_ts.draw()
        self.plot_tabs.setCurrentIndex(1)

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

    def _log(self, msg: str):
        self.log.append(msg)

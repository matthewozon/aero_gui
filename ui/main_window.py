"""
ui/main_window.py
-----------------
Top-level application window.  Hosts a QTabWidget with one tab per module.
"""

from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QStatusBar,
    QAction, QFileDialog, QMessageBox, QWidget, QApplication
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QFont

from modules.data_model import AerosolDataset

from ui.tab_data import DataTab
from ui.tab_model import ModelTab
from ui.tab_inversion import InversionTab
from ui.tab_estimation import EstimationTab
from ui.tab_simulation import SimulationTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AeroGUI – Atmospheric Particle Analysis")
        self._scale = 1.0

        screen = QApplication.desktop().availableGeometry()
        self.resize(min(1280, screen.width() - 40), min(860, screen.height() - 60))

        # Shared dataset (passed by reference to all tabs)
        self.dataset = AerosolDataset()

        self._build_ui()
        self._build_menu()
        self._apply_stylesheet()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.setCentralWidget(self.tabs)

        # Instantiate all tabs; pass shared dataset so they all see same data
        self.tab_data       = DataTab(self.dataset, self)
        self.tab_model      = ModelTab(self.dataset, self)
        self.tab_inversion  = InversionTab(self.dataset, self)
        self.tab_estimation = EstimationTab(self.dataset, self)
        self.tab_simulation = SimulationTab(self)

        self.tabs.addTab(self.tab_data,       "📁  Data")
        self.tabs.addTab(self.tab_model,      "⚙️  Measurement Model")
        self.tabs.addTab(self.tab_inversion,  "🔄  Inversion")
        self.tabs.addTab(self.tab_estimation, "📐  Parameter Estimation")
        self.tabs.addTab(self.tab_simulation, "🌀  Simulation")

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready.  Load a data file to begin.")

    def _build_menu(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open data file…", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.tab_data.open_file_dialog)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # View menu
        view_menu = menubar.addMenu("View")

        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.setShortcut("Ctrl+=")
        zoom_in_action.triggered.connect(self._zoom_in)
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.triggered.connect(self._zoom_out)
        view_menu.addAction(zoom_out_action)

        zoom_reset_action = QAction("Reset Zoom", self)
        zoom_reset_action.setShortcut("Ctrl+0")
        zoom_reset_action.triggered.connect(self._zoom_reset)
        view_menu.addAction(zoom_reset_action)

        # Help menu
        help_menu = menubar.addMenu("Help")
        about_action = QAction("About AeroGUI", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _apply_stylesheet(self):
        s = self._scale
        f_base  = round(12 * s)
        f_small = round(11 * s)
        f_tab   = round(13 * s)

        font = QApplication.font()
        font.setPointSizeF(10 * s)
        QApplication.setFont(font)

        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: #1e1e2e;
            }}
            QTabWidget::pane {{
                border: 1px solid #45475a;
                background-color: #1e1e2e;
            }}
            QTabBar::tab {{
                background: #313244;
                color: #cdd6f4;
                padding: {round(8*s)}px {round(18*s)}px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-size: {f_tab}px;
            }}
            QTabBar::tab:selected {{
                background: #89b4fa;
                color: #1e1e2e;
                font-weight: bold;
            }}
            QTabBar::tab:hover:!selected {{
                background: #45475a;
            }}
            QGroupBox {{
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                font-size: {f_base}px;
                font-weight: bold;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                color: #89b4fa;
            }}
            QLabel {{
                color: #cdd6f4;
                font-size: {f_base}px;
            }}
            QPushButton {{
                background-color: #45475a;
                color: #cdd6f4;
                border: 1px solid #585b70;
                border-radius: 5px;
                padding: {round(6*s)}px {round(14*s)}px;
                font-size: {f_base}px;
            }}
            QPushButton:hover {{
                background-color: #585b70;
            }}
            QPushButton:pressed {{
                background-color: #89b4fa;
                color: #1e1e2e;
            }}
            QPushButton:disabled {{
                background-color: #313244;
                color: #6c7086;
            }}
            QComboBox {{
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: {f_base}px;
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QComboBox QAbstractItemView {{
                background-color: #313244;
                color: #cdd6f4;
                selection-background-color: #89b4fa;
                selection-color: #1e1e2e;
            }}
            QLineEdit, QSpinBox, QDoubleSpinBox {{
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: {f_base}px;
            }}
            QCheckBox {{
                color: #cdd6f4;
                font-size: {f_base}px;
            }}
            QCheckBox::indicator:checked {{
                background-color: #89b4fa;
                border: 1px solid #89b4fa;
            }}
            QTextEdit {{
                background-color: #181825;
                color: #a6e3a1;
                border: 1px solid #45475a;
                border-radius: 4px;
                font-family: monospace;
                font-size: {f_small}px;
            }}
            QStatusBar {{
                background-color: #181825;
                color: #a6e3a1;
                font-size: {f_small}px;
            }}
            QScrollBar:vertical {{
                background: #1e1e2e;
                width: 10px;
            }}
            QScrollBar::handle:vertical {{
                background: #45475a;
                border-radius: 4px;
            }}
        """)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _zoom_in(self):
        self._scale = min(2.0, round(self._scale + 0.1, 1))
        self._apply_stylesheet()

    def _zoom_out(self):
        self._scale = max(0.5, round(self._scale - 0.1, 1))
        self._apply_stylesheet()

    def _zoom_reset(self):
        self._scale = 1.0
        self._apply_stylesheet()

    def set_status(self, msg: str):
        self.status.showMessage(msg)

    def _show_about(self):
        QMessageBox.about(
            self,
            "About AeroGUI",
            "<b>AeroGUI</b> v0.1<br>"
            "Atmospheric Particle Data Analysis Platform<br><br>"
            "Modules: Data · Measurement Model · Inversion · "
            "Parameter Estimation · Simulation"
        )

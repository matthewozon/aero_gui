"""
AeroGUI - Atmospheric Science Data Analysis Platform
=====================================================
Entry point. Run with:  python main.py
"""

import sys
import os

# --- Ensure project root is on the path ---
sys.path.insert(0, os.path.dirname(__file__))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from ui.main_window import MainWindow


def main():
    # Enable high-DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("AeroGUI")
    app.setOrganizationName("AtmosphericScience")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

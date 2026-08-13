"""Application bootstrap."""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from gui import theme
from gui.views.main_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeMenuBar, True)
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("RLRedTeam Analyst")

    font = QFont(theme.FONT_SANS)
    font.setPointSizeF(theme.SIZE_BODY * 0.75)  # px -> pt at 96dpi
    app.setFont(font)
    app.setStyleSheet(theme.build_stylesheet())

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

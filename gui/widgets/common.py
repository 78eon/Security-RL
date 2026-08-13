"""Shared widgets.

Two design rules are enforced here rather than left to each view:

* state is never carried by colour alone -- every coloured dot sits beside a
  word, so the UI stays readable projected, printed, or to a colourblind viewer;
* numeric cells are mono, right-aligned, tabular, with a real minus sign.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from gui import theme


def label(text: str, object_name: str = "", *, wrap: bool = False) -> QLabel:
    widget = QLabel(text)
    if object_name:
        widget.setObjectName(object_name)
    widget.setWordWrap(wrap)
    return widget


def section_label(text: str) -> QLabel:
    """Uppercase mono eyebrow, 11.5px -- the design's section marker."""
    return label(text.upper(), "sectionLabel")


def mono(text: str) -> QLabel:
    return label(text, "mono")


def hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFixedHeight(1)
    line.setStyleSheet(f"background:{theme.RULE}; border:0;")
    return line


class StatusDot(QWidget):
    """A coloured dot that is always accompanied by its word."""

    def __init__(self, colour: str, size: int = 7) -> None:
        super().__init__()
        self._colour = QColor(colour)
        self._size = size
        self.setFixedSize(size, size)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(self._colour)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(0, 0, self._size, self._size)


def status_chip(text: str, colour: str) -> QWidget:
    """Dot plus word. Never the dot alone."""
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(theme.SP_S)
    row.addWidget(StatusDot(colour))
    text_label = QLabel(text)
    text_label.setStyleSheet(
        f"color:{colour}; font-family:'{theme.FONT_MONO}'; "
        f"font-size:{theme.SIZE_LABEL}px; background:transparent;"
    )
    row.addWidget(text_label)
    row.addStretch(1)
    return holder


class Banner(QFrame):
    """State message: 3px left border in the state colour over a tinted ground.

    Used for empty, failure and warning states. Carries a title, an explanation
    and optional actions -- an empty state that only says "no data" wastes the
    one moment the user is looking for what to do next.
    """

    def __init__(
        self,
        title: str,
        body: str = "",
        kind: str = "info",
        detail: str = "",
    ) -> None:
        super().__init__()
        colours = {
            "info": theme.ARM_1,
            "ok": theme.OK,
            "warn": theme.WARN,
            "error": theme.ERROR,
        }
        colour = colours.get(kind, theme.ARM_1)
        tint = theme.STATE_TINT.get(kind, theme.TINT_INFO)

        # Scoped by object name: QLabel subclasses QFrame, so an unscoped
        # "QFrame { border: ... }" would draw a box around every child label.
        self.setObjectName("bannerFrame")
        self.setStyleSheet(
            f"QFrame#bannerFrame {{ background:{tint}; "
            f"border:1px solid {theme.BORDER}; border-left:3px solid {colour}; }}"
            f"QFrame#bannerFrame QLabel {{ border:0; background:transparent; }}"
        )
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

        column = QVBoxLayout(self)
        column.setContentsMargins(theme.PANEL_PAD, theme.SP_M, theme.PANEL_PAD, theme.SP_M)
        column.setSpacing(theme.SP_S)

        heading = QLabel(title)
        heading.setStyleSheet(
            f"color:{theme.TEXT}; font-size:{theme.SIZE_HEADING}px; "
            "font-weight:600; background:transparent;"
        )
        heading.setWordWrap(True)
        column.addWidget(heading)

        if body:
            explanation = QLabel(body)
            explanation.setWordWrap(True)
            explanation.setStyleSheet(
                f"color:{theme.TEXT_SECONDARY}; background:transparent;"
            )
            column.addWidget(explanation)

        if detail:
            detail_label = QLabel(detail)
            detail_label.setWordWrap(True)
            detail_label.setStyleSheet(
                f"color:{theme.TEXT_TERTIARY}; font-family:'{theme.FONT_MONO}'; "
                f"font-size:{theme.SIZE_SMALL}px; background:transparent;"
            )
            column.addWidget(detail_label)

        self._actions = QHBoxLayout()
        self._actions.setSpacing(theme.SP_S)
        self._actions.setContentsMargins(0, 0, 0, 0)
        self._actions.addStretch(1)
        column.addLayout(self._actions)

    def add_action(self, button) -> None:
        self._actions.insertWidget(self._actions.count() - 1, button)


class MetricTile(QFrame):
    """Headline number: label, big tabular value, optional delta."""

    def __init__(self, caption: str, value: str = "—", delta: str = "") -> None:
        super().__init__()
        self.setObjectName("metricTile")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(96)

        column = QVBoxLayout(self)
        column.setContentsMargins(theme.PANEL_PAD, theme.SP_M, theme.PANEL_PAD, theme.SP_M)
        column.setSpacing(theme.SP_XS)

        self._caption = label(caption.upper(), "metricLabel")
        self._value = label(value, "metricValue")
        font = QFont(theme.FONT_MONO, theme.SIZE_METRIC)
        font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        self._value.setFont(font)
        self._delta = label(delta, "metricDelta")

        column.addWidget(self._caption)
        column.addWidget(self._value)
        column.addWidget(self._delta)
        column.addStretch(1)

    def set_value(self, value: str, delta: str = "") -> None:
        self._value.setText(value)
        self._delta.setText(delta)

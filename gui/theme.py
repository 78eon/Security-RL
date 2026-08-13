"""Design tokens, taken verbatim from the visual direction document.

Single source of truth: `theme.qss` is generated from these values at startup, so
a colour can never drift between the stylesheet and code that paints (charts,
QGraphicsScene items) which QSS cannot reach.
"""

from __future__ import annotations

from pathlib import Path

# -- surfaces ---------------------------------------------------------------
WINDOW = "#0d1013"  # window backdrop
SURFACE = "#14181d"  # base surface
PANEL = "#11151a"  # rail, side panel, footer
HEADER = "#1b2027"  # header row, button face
RAISED = "#1e242c"  # hover step above HEADER

# -- lines ------------------------------------------------------------------
BORDER = "#2b313b"  # outer borders
RULE = "#22272f"  # inner rules between rows
BORDER_STRONG = "#333a45"  # button border
BORDER_SOFT = "#3b424d"

# -- text -------------------------------------------------------------------
TEXT = "#e6e9ee"  # primary
TEXT_BRIGHT = "#d3d9e2"
TEXT_SECONDARY = "#8d96a5"
TEXT_TERTIARY = "#7d8797"  # contrast floor on SURFACE (4.6:1)
TEXT_MUTED = "#6b7482"

# -- series -----------------------------------------------------------------
# Fixed and validated colourblind-safe (CVD separation dE 24.7). ARM_2 is a
# series colour ONLY -- never a UI accent, or a chart legend stops being
# distinguishable from a button.
ARM_1 = "#2a78d6"  # sparse / native, also selection + primary action
ARM_2 = "#eb6834"  # shaped
ARM_1_LIGHT = "#8dbcf0"
ARM_2_LIGHT = "#f0a184"
SELECTION = "#1a2530"  # selected row fill -- not a blue wash
PRIMARY_FILL = "#1a2c42"  # primary button face

# -- state ------------------------------------------------------------------
OK = "#3f9e5a"
OK_TEXT = "#8fd0a3"
WARN = "#d9962b"
WARN_TEXT = "#e8c48f"
ERROR = "#b0242c"
ERROR_TEXT = "#e79a9a"

TINT_OK = "#131820"
TINT_WARN = "#2a1f16"
TINT_ERROR = "#1d1418"
TINT_INFO = "#131820"

STATE_TINT = {"ok": TINT_OK, "warn": TINT_WARN, "error": TINT_ERROR, "info": TINT_INFO}

# -- CVSS severity ----------------------------------------------------------
# Its own scale, deliberately distinct from ARM_1/ARM_2 so severity never reads
# as series identity.
SEVERITY = {
    "LOW": "#8fd0a3",
    "MEDIUM": "#a8b02e",
    "HIGH": "#d9962b",
    "CRITICAL": "#d9552b",
    "NONE": TEXT_MUTED,
}

# -- replay host states -----------------------------------------------------
HOST_STATE = {
    "undiscovered": ("#171b21", "#2b313b", TEXT_MUTED),
    "discovered": ("#1c2128", "#444c58", TEXT_SECONDARY),
    "user": ("#1a2c42", ARM_1, ARM_1_LIGHT),
    "root": ("#3d5a76", ARM_1_LIGHT, "#ffffff"),
    "crown": ("#2a1f16", WARN, WARN_TEXT),
}

# -- type -------------------------------------------------------------------
FONT_SANS = "IBM Plex Sans"
FONT_MONO = "IBM Plex Mono"

SIZE_TITLE = 19  # view title, weight 600
SIZE_HEADING = 15  # panel heading, weight 600
SIZE_BODY = 13.5  # body, table cells
SIZE_LABEL = 11.5  # section label, column header, uppercase
SIZE_MONO = 13  # run names, ids, all numerals
SIZE_METRIC = 27  # metric tile
SIZE_SMALL = 12

# -- spacing ----------------------------------------------------------------
# The whole scale. Nothing else.
SP_XS, SP_S, SP_M, SP_L = 4, 8, 12, 18
PANEL_PAD = 16
RAIL_WIDTH = 186
DETAIL_WIDTH = 352
ROW_HEIGHT = 40
ROW_HEIGHT_DENSE = 31

MINUS = "−"  # real minus sign, so negative numerals align on the decimal


def fmt_num(value: float | int | None, decimals: int = 0) -> str:
    """Format for a numeric column: thousands separated, true minus, em dash for null.

    A missing value renders as an em dash rather than 0, which would be a lie --
    'no hosts exploited' and 'mean CVSS of zero' are different facts.
    """
    if value is None:
        return "—"
    text = f"{value:,.{decimals}f}"
    return text.replace("-", MINUS)


def truncate_hash(value: str | None, head: int = 4, tail: int = 2) -> str:
    """Shorten a hash for a table cell. Full value belongs in the detail panel."""
    if not value:
        return "—"
    if len(value) <= head + tail + 1:
        return value
    return f"{value[:head]}…{value[-tail:]}"


def build_stylesheet() -> str:
    """Render theme.qss with the tokens substituted in."""
    qss = (Path(__file__).parent / "theme.qss").read_text()
    tokens = {
        k: v for k, v in globals().items() if k.isupper() and isinstance(v, str)
    }
    tokens.update(
        {
            "SIZE_TITLE": SIZE_TITLE, "SIZE_HEADING": SIZE_HEADING,
            "SIZE_BODY": SIZE_BODY, "SIZE_LABEL": SIZE_LABEL,
            "SIZE_MONO": SIZE_MONO, "SIZE_SMALL": SIZE_SMALL,
            "SP_XS": SP_XS, "SP_S": SP_S, "SP_M": SP_M, "SP_L": SP_L,
            "RAIL_WIDTH": RAIL_WIDTH, "ROW_HEIGHT": ROW_HEIGHT,
        }
    )
    for key, value in tokens.items():
        qss = qss.replace(f"@{key}@", str(value))
    return qss

"""CVSS v3.1 severity handling and the CVE severity -> reward weight mapping.

Pure functions only: no I/O, no RNG, no dependency on nasim/gymnasium/torch.
This is what makes the reward core testable before the environment is wired.

The weight produced here is the ``CVE/CVSS x weight`` term of the Essential
reward (see :mod:`rlredteam.reward`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

# CVSS v3.1 qualitative severity ratings, section 5.
# https://www.first.org/cvss/v3.1/specification-document
_BANDS: tuple[tuple[float, float, str], ...] = (
    (0.0, 0.0, "NONE"),
    (0.1, 3.9, "LOW"),
    (4.0, 6.9, "MEDIUM"),
    (7.0, 8.9, "HIGH"),
    (9.0, 10.0, "CRITICAL"),
)

MIN_SCORE = 0.0
MAX_SCORE = 10.0


class Severity(StrEnum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class WeightMode(StrEnum):
    """How a base score becomes a reward weight."""

    POWER = "power"
    LINEAR = "linear"
    BAND = "band"


def validate_base_score(score: float) -> float:
    """Return ``score`` if it is a valid CVSS base score, else raise.

    Rejects NaN explicitly: ``NaN`` fails every comparison silently, so a
    range check alone would let it through and poison the weight downstream.
    """
    if score is None or isinstance(score, bool):
        raise ValueError(f"base score must be a number, got {score!r}")
    value = float(score)
    if value != value:
        raise ValueError("base score must not be NaN")
    if not MIN_SCORE <= value <= MAX_SCORE:
        raise ValueError(f"base score {value} outside CVSS range [0.0, 10.0]")
    return value


def severity_band(score: float) -> Severity:
    """Map a CVSS v3.1 base score to its qualitative severity rating."""
    value = validate_base_score(score)
    for low, high, name in _BANDS:
        if low <= value <= high:
            return Severity(name)
    raise AssertionError(f"unreachable: no band matched {value}")


@dataclass(frozen=True, slots=True)
class WeightParams:
    """Parameters of the severity -> weight map.

    ``gamma`` is a severity-emphasis exponent. It exists because real CVEs for
    common services cluster in 7.0-9.8, where a linear map spans only ~1.37x --
    narrower than PPO's seed-to-seed noise at n=10, which would make any effect
    undetectable. gamma=2.0 roughly doubles that dynamic range.

    Chosen a priori, not tuned on results. See docs/BUILD_PLAN.md.
    """

    mode: WeightMode = WeightMode.POWER
    gamma: float = 2.0
    w_min: float = 0.25
    w_max: float = 1.0
    band_weights: dict[Severity, float] | None = None

    def __post_init__(self) -> None:
        if self.gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {self.gamma}")
        if not 0.0 <= self.w_min <= self.w_max:
            raise ValueError(f"require 0 <= w_min <= w_max, got {self.w_min}, {self.w_max}")
        if self.mode is WeightMode.BAND and not self.band_weights:
            raise ValueError("WeightMode.BAND requires band_weights")


_DEFAULT_BAND_WEIGHTS: dict[Severity, float] = {
    Severity.NONE: 0.0,
    Severity.LOW: 0.25,
    Severity.MEDIUM: 0.50,
    Severity.HIGH: 0.75,
    Severity.CRITICAL: 1.00,
}


def severity_weight(score: float, params: WeightParams | None = None) -> float:
    """Map a CVSS base score to a reward weight in ``[w_min, w_max]``.

    POWER (default):  w = w_min + (w_max - w_min) * (score/10) ** gamma
    LINEAR:           the same with gamma = 1 -- kept as an ablation arm.
    BAND:             a step function over the five severity ratings. Kept for a
                      robustness check only: it assigns identical weights within
                      a band, discarding exactly the within-band variance this
                      study depends on.
    """
    params = params or WeightParams()
    value = validate_base_score(score)

    if params.mode is WeightMode.BAND:
        weights = params.band_weights or _DEFAULT_BAND_WEIGHTS
        return float(weights[severity_band(value)])

    exponent = 1.0 if params.mode is WeightMode.LINEAR else params.gamma
    normalised = value / MAX_SCORE
    return params.w_min + (params.w_max - params.w_min) * (normalised**exponent)


def contrast_ratio(scores: list[float], params: WeightParams | None = None) -> float:
    """Ratio of the largest to smallest weight over ``scores``.

    Reported in the methodology as the design statistic showing the catalogue
    has enough dynamic range for an effect to be detectable. A catalogue whose
    CVEs all score alike yields ~1.0 and cannot support the ablation.
    """
    if not scores:
        raise ValueError("contrast_ratio needs at least one score")
    weights = [severity_weight(s, params) for s in scores]
    smallest = min(weights)
    if smallest <= 0.0:
        return float("inf")
    return max(weights) / smallest


# CVSS v3.x vector, spec section 6. Validated structurally; the base score is
# taken from NVD rather than recomputed here (see docs/BUILD_PLAN.md).
_VECTOR_RE = re.compile(
    r"^CVSS:(?P<version>3\.[01])"
    r"/AV:(?P<av>[NALP])/AC:(?P<ac>[LH])/PR:(?P<pr>[NLH])/UI:(?P<ui>[NR])"
    r"/S:(?P<s>[UC])/C:(?P<c>[NLH])/I:(?P<i>[NLH])/A:(?P<a>[NLH])$"
)


@dataclass(frozen=True, slots=True)
class CvssVector:
    version: str
    av: str
    ac: str
    pr: str
    ui: str
    s: str
    c: str
    i: str
    a: str
    raw: str


def parse_vector(vector: str) -> CvssVector:
    """Parse and structurally validate a CVSS v3.0/v3.1 base vector string."""
    if not isinstance(vector, str):
        raise ValueError(f"vector must be a string, got {type(vector).__name__}")
    match = _VECTOR_RE.match(vector.strip())
    if match is None:
        raise ValueError(f"malformed CVSS v3.x base vector: {vector!r}")
    return CvssVector(raw=vector.strip(), **match.groupdict())

"""CVSS band boundaries and the severity -> weight map."""

from __future__ import annotations

import math

import pytest

from rlredteam.cvss import (
    Severity,
    WeightMode,
    WeightParams,
    contrast_ratio,
    parse_vector,
    severity_band,
    severity_weight,
    validate_base_score,
)


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.0, Severity.NONE),
        # Band edges are where off-by-one errors live, so test each boundary.
        (0.1, Severity.LOW),
        (3.9, Severity.LOW),
        (4.0, Severity.MEDIUM),
        (6.9, Severity.MEDIUM),
        (7.0, Severity.HIGH),
        (8.9, Severity.HIGH),
        (9.0, Severity.CRITICAL),
        (9.8, Severity.CRITICAL),
        (10.0, Severity.CRITICAL),
    ],
)
def test_severity_band_boundaries(score: float, expected: Severity) -> None:
    assert severity_band(score) is expected


@pytest.mark.parametrize("bad", [-0.1, 10.1, 100.0, float("nan"), None, True])
def test_validate_base_score_rejects(bad: object) -> None:
    with pytest.raises(ValueError):
        validate_base_score(bad)  # type: ignore[arg-type]


def test_nan_is_rejected_not_silently_passed() -> None:
    # NaN fails every comparison, so a naive range check would let it through.
    with pytest.raises(ValueError, match="NaN"):
        validate_base_score(float("nan"))


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.0, 0.25),
        (10.0, 1.0),
        (5.0, 0.25 + 0.75 * 0.25),
        (7.8, 0.25 + 0.75 * 0.78**2),
        (9.8, 0.25 + 0.75 * 0.98**2),
    ],
)
def test_power_weight_values(score: float, expected: float) -> None:
    assert severity_weight(score) == pytest.approx(expected)


@pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0, 3.0, 4.0])
def test_weight_is_monotonic_and_bounded(gamma: float) -> None:
    params = WeightParams(gamma=gamma, w_min=0.25, w_max=1.0)
    previous = -math.inf
    for step in range(1001):
        score = step / 100.0
        weight = severity_weight(score, params)
        assert params.w_min - 1e-12 <= weight <= params.w_max + 1e-12
        assert weight >= previous - 1e-12, f"not monotonic at {score}"
        previous = weight


def test_linear_mode_is_gamma_one() -> None:
    linear = WeightParams(mode=WeightMode.LINEAR)
    power_one = WeightParams(mode=WeightMode.POWER, gamma=1.0)
    for score in (0.0, 3.7, 5.3, 8.1, 10.0):
        assert severity_weight(score, linear) == pytest.approx(
            severity_weight(score, power_one)
        )


def test_band_mode_collapses_within_band() -> None:
    # Documents exactly why BAND is not the default: it discards the
    # within-band variance the ablation depends on.
    params = WeightParams(
        mode=WeightMode.BAND,
        band_weights={s: i / 4 for i, s in enumerate(Severity)},
    )
    assert severity_weight(7.0, params) == severity_weight(8.9, params)


def test_band_mode_requires_weights() -> None:
    with pytest.raises(ValueError, match="band_weights"):
        WeightParams(mode=WeightMode.BAND)


def test_gamma_raises_contrast_over_linear() -> None:
    """The stated reason gamma=2 was chosen over a linear map."""
    scores = [7.0, 7.5, 7.8, 8.1, 8.8, 9.8, 10.0]
    linear = contrast_ratio(scores, WeightParams(mode=WeightMode.LINEAR))
    power = contrast_ratio(scores, WeightParams(gamma=2.0))
    assert power > linear


def test_contrast_ratio_of_uniform_scores_is_one() -> None:
    assert contrast_ratio([9.8, 9.8, 9.8]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "vector",
    [
        "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
        "CVSS:3.0/AV:L/AC:H/PR:L/UI:R/S:C/C:N/I:L/A:N",
    ],
)
def test_parse_vector_accepts_valid(vector: str) -> None:
    parsed = parse_vector(vector)
    assert parsed.raw == vector
    assert parsed.version in ("3.0", "3.1")


@pytest.mark.parametrize(
    "vector",
    [
        "CVSS:2.0/AV:N/AC:L/Au:N/C:P/I:P/A:P",  # v2 must not be accepted
        "AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",  # missing prefix
        "CVSS:3.1/AV:X/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",  # bad metric value
        "",
        "nonsense",
    ],
)
def test_parse_vector_rejects_invalid(vector: str) -> None:
    with pytest.raises(ValueError):
        parse_vector(vector)

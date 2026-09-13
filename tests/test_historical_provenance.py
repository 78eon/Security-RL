from __future__ import annotations

import copy
import json
from pathlib import Path

from rlredteam.enterprise.recurrent import RecurrentResearchConfig
from rlredteam.enterprise.recurrent_study import current_input_manifest
from rlredteam.historical_provenance import frozen_inputs_match

REPO_ROOT = Path(__file__).resolve().parents[1]


def recurrent_inputs() -> tuple[dict, dict]:
    current = current_input_manifest(RecurrentResearchConfig.from_yaml())
    frozen = json.loads((REPO_ROOT / "configs/frozen_recurrent_policy.json").read_text())
    return current, frozen


def test_later_source_changes_do_not_invalidate_historical_evidence() -> None:
    current, frozen = recurrent_inputs()
    assert current["source_sha256"] != frozen["source_sha256"]

    assert frozen_inputs_match(REPO_ROOT, current, frozen) == (True, "")


def test_current_scientific_configuration_drift_still_fails() -> None:
    current, frozen = recurrent_inputs()
    changed = copy.deepcopy(current)
    changed["experiment_config_sha256"] = "0" * 64

    matches, reason = frozen_inputs_match(REPO_ROOT, changed, frozen)

    assert not matches
    assert "non-source scientific inputs" in reason


def test_historical_source_tampering_still_fails() -> None:
    current, frozen = recurrent_inputs()
    changed = copy.deepcopy(frozen)
    source = next(iter(changed["source_sha256"]))
    changed["source_sha256"][source] = "0" * 64

    matches, reason = frozen_inputs_match(REPO_ROOT, current, changed)

    assert not matches
    assert "source hash mismatch" in reason

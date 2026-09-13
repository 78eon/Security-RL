"""Fail-closed completion checks for Phase 14 reports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rlredteam.attack_path_report import (
    CRITICALITY_METHOD,
    EvidenceValidationError,
    canonical_sha256,
    generate_report,
)


class Phase14VerificationError(RuntimeError):
    """The report is not exactly reconstructable from its recorded evidence."""


def verify_report(source_path: Path, report_path: Path) -> dict[str, Any]:
    try:
        stored = json.loads(Path(report_path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase14VerificationError(f"cannot read Phase 14 report: {exc}") from exc
    if not isinstance(stored, Mapping):
        raise Phase14VerificationError("Phase 14 report must be an object")
    try:
        provenance = stored["provenance"]
        rebuilt = generate_report(
            Path(source_path),
            mode=str(stored["report_mode"]),
            experiment_id=str(provenance["experiment_id"]),
            checkpoint_hashes=provenance.get("source_checkpoint_hashes") or {},
        ).to_dict()
    except (KeyError, TypeError, EvidenceValidationError) as exc:
        raise Phase14VerificationError(f"incomplete or invalid report provenance: {exc}") from exc
    if stored != rebuilt:
        raise Phase14VerificationError(
            "report facts are not exactly derivable from the recorded source evidence"
        )
    facts_hash = canonical_sha256(stored["facts"])
    if facts_hash != provenance["facts_payload_sha256"]:
        raise Phase14VerificationError("facts payload hash mismatch")
    methodology = stored.get("methodology") or {}
    if methodology.get("criticality_method") != CRITICALITY_METHOD:
        raise Phase14VerificationError(
            "observed path criticality is missing its non-counterfactual label"
        )
    forbidden_claims = ("would block", "all attacks", "true counterfactual")
    statements = [
        str(item.get("statement", "")).casefold()
        for item in stored.get("observed_path_criticality", [])
    ]
    if any(claim in statement for statement in statements for claim in forbidden_claims):
        raise Phase14VerificationError("path criticality contains a counterfactual claim")
    return {
        "status": "pass",
        "report_id": stored["report_id"],
        "report_mode": stored["report_mode"],
        "facts": len(stored["facts"]),
        "trajectories": stored["cross_episode_analysis"]["trajectory_count"],
        "provenance": "complete",
        "catalogue_only_mappings": "pass",
        "hidden_topology_leakage": "none",
        "criticality_label": CRITICALITY_METHOD,
    }

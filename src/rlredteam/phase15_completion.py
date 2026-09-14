"""Fail-closed completion checks for Phase 15 counterfactual reports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rlredteam.mitigation import (
    CRITICALITY_LABEL,
    EFFECT_LABEL,
    INTERVENTION_VERSION,
    REPORT_SCHEMA_VERSION,
    MitigationError,
    analyse_pairs,
    validate_report_identity,
)


class Phase15VerificationError(RuntimeError):
    """The paired report does not prove the registered intervention."""


def verify_report(report_path: Path) -> dict[str, Any]:
    try:
        report = json.loads(Path(report_path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase15VerificationError(f"cannot read Phase 15 report: {exc}") from exc
    if not isinstance(report, Mapping):
        raise Phase15VerificationError("Phase 15 report must be an object")
    try:
        validate_report_identity(report)
        if report["schema_version"] != REPORT_SCHEMA_VERSION:
            raise Phase15VerificationError("report schema version drift")
        intervention = report["intervention"]
        provenance = report["provenance"]
        boundaries = report["claim_boundaries"]
        pairs = report["pairs"]
        if intervention["intervention_version"] != INTERVENTION_VERSION:
            raise Phase15VerificationError("intervention version drift")
        if intervention["unselected_assignments_changed"] != 0:
            raise Phase15VerificationError("an unselected CVE assignment changed")
        if intervention["canonical_catalogue_mutated"] is not False:
            raise Phase15VerificationError("canonical catalogue mutation was reported")
        if boundaries["observed_path_criticality"] != CRITICALITY_LABEL:
            raise Phase15VerificationError("observed criticality claim boundary is missing")
        if boundaries["mitigation_effect"] != EFFECT_LABEL:
            raise Phase15VerificationError("mitigation effect claim boundary is missing")
        original_seeds = list(provenance["evaluation_seeds_original"])
        mitigated_seeds = list(provenance["evaluation_seeds_mitigated"])
        pair_seeds = [pair["evaluation_seed"] for pair in pairs]
        if original_seeds != mitigated_seeds or original_seeds != pair_seeds:
            raise Phase15VerificationError("evaluation seed pairing drift")
        policy_hashes = {
            provenance["policy_sha256_before"],
            provenance["policy_sha256_after_original"],
            provenance["policy_sha256_after_mitigated"],
        }
        if len(policy_hashes) != 1 or provenance["gradient_updates"] is not False:
            raise Phase15VerificationError("frozen policy invariant failed")
        if provenance.get("evaluation_git_dirty") is not False:
            raise Phase15VerificationError("evaluation was produced from a dirty/unknown tree")
        if provenance["protected_artifacts_before"] != provenance["protected_artifacts_after"]:
            raise Phase15VerificationError("protected frozen artefact digest changed")
        selected_cve = intervention["selected_cve"]
        if any(
            step.get("cve_id") == selected_cve
            for pair in pairs
            for step in pair["mitigated"]["observed_path"]
        ):
            raise Phase15VerificationError("disabled CVE succeeded in a mitigated path")
        if report["analysis"] != analyse_pairs(list(pairs)):
            raise Phase15VerificationError("paired statistics are not reconstructable")
    except (KeyError, TypeError, MitigationError) as exc:
        raise Phase15VerificationError(f"incomplete or invalid report: {exc}") from exc
    return {
        "status": "pass",
        "report_id": report["report_id"],
        "selected_cve": intervention["selected_cve"],
        "paired_seeds": len(pairs),
        "policy_immutable": "pass",
        "frozen_artifacts_immutable": "pass",
        "claim_boundaries": "pass",
        "postgres_reconstruction": "not_requested",
    }


__all__ = ["Phase15VerificationError", "verify_report"]

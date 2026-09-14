"""Phase 15 evaluation-only CVE mitigation counterfactuals.

The intervention preserves the reconstructed NASim scenario and action space.
It blocks execution only when the frozen policy selects an action whose existing
deterministic assignment names the selected CVE.  No training API is present.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
from scipy import stats

from rlredteam.assign import assign_cves
from rlredteam.catalogue import DEFAULT_DB, CVECatalogue
from rlredteam.evaluation import (
    EvaluationBundle,
    evaluate_policy,
    policy_digest,
    sha256_file,
)
from rlredteam.manifest import digest
from rlredteam.nasim_adapter import RewardWrapper
from rlredteam.provenance import ExperimentManifest, git_commit, git_dirty, topology_hash
from rlredteam.reward import RewardConfig
from rlredteam.topology import DEFAULT_TOPOLOGY_CONFIG, TopologyConfig, describe, make_env

REPORT_SCHEMA_VERSION = "security-rl-mitigation-counterfactual-v1"
INTERVENTION_VERSION = "cve-unavailable-overlay-v1"
EFFECT_LABEL = "mitigation_effect_measured_by_frozen_policy_rerun"
CRITICALITY_LABEL = "observed_path_criticality_derived_from_recorded_paths_only"
_CVE = re.compile(r"^CVE-\d{4}-\d{4,}$")


class MitigationError(RuntimeError):
    """The intervention or paired evaluation cannot support a valid claim."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class MitigationSpec:
    selected_cve: str
    intervention_version: str = INTERVENTION_VERSION
    application_stage: str = "after_environment_reconstruction_before_action_execution"
    behavior: str = "assigned_actions_remain_visible_but_cannot_change_simulator_state"

    @classmethod
    def create(cls, selected_cve: str, catalogue: CVECatalogue) -> MitigationSpec:
        value = str(selected_cve).strip().upper()
        if not _CVE.fullmatch(value):
            raise MitigationError(f"invalid CVE identifier: {selected_cve!r}")
        catalogue.lookup(value)
        return cls(selected_cve=value)

    def digest(self) -> str:
        return canonical_sha256(asdict(self))


class CveMitigationOverlay(gym.Wrapper):
    """Evaluation-only wrapper that disables one assigned CVE without remapping."""

    def __init__(
        self,
        env,
        *,
        catalogue: CVECatalogue,
        topology_seed: int,
        spec: MitigationSpec,
    ) -> None:
        super().__init__(env)
        scenario = env.scenario
        self.scenario = scenario
        self.intervention_spec = spec
        self.assignment = assign_cves(
            list(scenario.exploits), list(scenario.privescs), catalogue, topology_seed
        )
        self.affected_actions = tuple(
            sorted(
                name
                for name, cve_id in self.assignment.mapping.items()
                if cve_id == spec.selected_cve
            )
        )
        if not self.affected_actions:
            raise MitigationError(
                f"{spec.selected_cve} is not assigned in topology seed {topology_seed}"
            )
        self.assignment_sha256 = canonical_sha256(self.assignment.mapping)
        self._observation: Any = None
        self._logical_steps = 0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._observation = copy.deepcopy(observation)
        self._logical_steps = 0
        return observation, info

    def step(self, action):
        index = int(action)
        if not 0 <= index < int(self.action_space.n):
            raise ValueError(f"action {index} outside [0, {int(self.action_space.n)})")
        resolved = self.action_space.get_action(index)
        self._logical_steps += 1
        if resolved.name in self.affected_actions:
            if self._observation is None:
                raise MitigationError("mitigation overlay must be reset before step")
            truncated = self._logical_steps >= int(self.env.scenario.step_limit)
            info = {
                "success": False,
                "access": 0,
                "mitigation_blocked": True,
                "connection_error": False,
                "permission_error": False,
                "undefined_error": False,
            }
            return (
                copy.deepcopy(self._observation),
                -float(resolved.cost),
                False,
                truncated,
                info,
            )
        observation, reward, terminated, truncated, info = self.env.step(index)
        self._observation = copy.deepcopy(observation)
        if self._logical_steps >= int(self.env.scenario.step_limit):
            truncated = True
        return observation, reward, terminated, truncated, info


def promote_phase14_cve(report_path: Path, selected_cve: str | None = None) -> dict[str, Any]:
    """Select a genuinely observed path-critical CVE without claiming mitigation effect."""
    try:
        report = json.loads(Path(report_path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise MitigationError(f"cannot read Phase 14 report: {exc}") from exc
    if report.get("report_mode") != "simulation_report":
        raise MitigationError("mitigation experiments require a Phase 14 simulation_report")
    candidates = [
        item
        for item in report.get("observed_path_criticality", [])
        if float(item.get("observed_path_criticality_percentage", 0.0)) > 0.0
    ]
    if not candidates:
        raise MitigationError("Phase 14 report contains no observed path-critical CVE")
    chosen = str(selected_cve).upper() if selected_cve else str(candidates[0]["cve_id"])
    match = next((item for item in candidates if item.get("cve_id") == chosen), None)
    if match is None:
        raise MitigationError(f"{chosen} is not observed path-critical in the Phase 14 report")
    return {
        "source_report_id": str(report.get("report_id", "")),
        "source_report_sha256": sha256_file(Path(report_path)),
        "selected_cve": chosen,
        "observed_path_criticality_percentage": float(
            match["observed_path_criticality_percentage"]
        ),
        "observed_path_criticality_label": CRITICALITY_LABEL,
    }


def evaluate_checkpoint_mitigation(
    run_dir: Path,
    evaluation_seeds: list[int],
    selected_cve: str,
    output: Path,
    *,
    phase14_report: Path | None = None,
    postgres: bool = False,
    protected_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Rerun one frozen PPO policy under original and one-CVE-disabled conditions."""
    from stable_baselines3 import PPO

    if not evaluation_seeds or len(set(evaluation_seeds)) != len(evaluation_seeds):
        raise MitigationError("evaluation seeds must be non-empty and unique")
    if any(isinstance(seed, bool) for seed in evaluation_seeds):
        raise MitigationError("evaluation seeds must be integers")
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    checkpoint = run_dir / "model.zip"
    if not manifest_path.is_file() or not checkpoint.is_file():
        raise MitigationError("run directory must contain manifest.json and model.zip")
    manifest = ExperimentManifest.read(manifest_path)
    topology_config = TopologyConfig.from_yaml()
    catalogue = CVECatalogue.open_default()
    reward_config = RewardConfig.from_yaml(
        Path(__file__).resolve().parents[2] / "configs" / f"{manifest.reward_mode}.yaml"
    )
    base_description = describe(make_env(topology_config, topology_seed=manifest.topology_seed))
    checks = {
        "topology_config_hash": (manifest.topology_config_hash, topology_config.config_hash()),
        "topology_hash": (manifest.topology_hash, topology_hash(base_description)),
        "cve_catalogue_hash": (manifest.cve_manifest_sha256, digest(catalogue)),
        "reward_config_hash": (manifest.reward_config_hash, reward_config.hash()),
    }
    mismatches = [
        f"{name}: manifest={expected}, reconstructed={actual}"
        for name, (expected, actual) in checks.items()
        if expected != actual
    ]
    if mismatches:
        raise MitigationError(
            "frozen environment cannot be reconstructed:\n  " + "\n  ".join(mismatches)
        )

    phase14 = (
        promote_phase14_cve(Path(phase14_report), selected_cve)
        if phase14_report is not None
        else None
    )
    spec = MitigationSpec.create(selected_cve, catalogue)
    protected_inputs = [
        run_dir,
        Path(DEFAULT_DB),
        Path(DEFAULT_TOPOLOGY_CONFIG),
        Path(__file__).resolve().parents[2] / "configs" / f"{manifest.reward_mode}.yaml",
    ]
    if phase14_report is not None:
        protected_inputs.append(Path(phase14_report))
    protected = tuple(dict.fromkeys((*protected_inputs, *map(Path, protected_paths))))
    protected_before = {str(path): artifact_tree_digest(path) for path in protected}

    def original_factory():
        return RewardWrapper(
            make_env(topology_config, topology_seed=manifest.topology_seed),
            catalogue,
            topology_seed=manifest.topology_seed,
            reward_config=reward_config,
        )

    overlay_metadata: dict[str, Any] = {}

    def mitigated_factory():
        overlay = CveMitigationOverlay(
            make_env(topology_config, topology_seed=manifest.topology_seed),
            catalogue=catalogue,
            topology_seed=manifest.topology_seed,
            spec=spec,
        )
        overlay_metadata.update(
            {
                "affected_actions": list(overlay.affected_actions),
                "original_assignment_sha256": overlay.assignment_sha256,
            }
        )
        return RewardWrapper(
            overlay,
            catalogue,
            topology_seed=manifest.topology_seed,
            reward_config=reward_config,
        )

    model = PPO.load(checkpoint, device="cpu")
    policy_before = policy_digest(model)
    original = evaluate_policy(
        model,
        run_name=manifest.experiment_id,
        reward_mode=manifest.reward_mode,
        training_seed=manifest.training_seed,
        evaluation_seeds=list(evaluation_seeds),
        topology_seed=manifest.topology_seed,
        topology_config=topology_config,
        reward_config=reward_config,
        environment_factory=original_factory,
    )
    policy_after_original = policy_digest(model)
    mitigated = evaluate_policy(
        model,
        run_name=manifest.experiment_id,
        reward_mode=manifest.reward_mode,
        training_seed=manifest.training_seed,
        evaluation_seeds=list(evaluation_seeds),
        topology_seed=manifest.topology_seed,
        topology_config=topology_config,
        reward_config=reward_config,
        environment_factory=mitigated_factory,
    )
    policy_after_mitigated = policy_digest(model)
    if len({policy_before, policy_after_original, policy_after_mitigated}) != 1:
        raise MitigationError("policy parameters changed during evaluation")

    protected_after = {str(path): artifact_tree_digest(path) for path in protected}
    if protected_before != protected_after:
        raise MitigationError("a protected frozen artefact changed during evaluation")
    pairs = _pair_bundles(original, mitigated, evaluation_seeds)
    report_body = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "claim_boundaries": {
            "observed_path_criticality": CRITICALITY_LABEL,
            "mitigation_effect": EFFECT_LABEL,
            "causal_scope": (
                "Effect is measured for this frozen policy, topology, intervention, and seed set; "
                "it is not a claim that patching the CVE blocks all attacks."
            ),
        },
        "intervention": {
            **asdict(spec),
            "intervention_sha256": spec.digest(),
            **overlay_metadata,
            "unselected_assignments_changed": 0,
            "canonical_catalogue_mutated": False,
        },
        "phase14_promotion": phase14,
        "pairs": pairs,
        "analysis": analyse_pairs(pairs),
        "provenance": {
            "run_name": manifest.experiment_id,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "policy_sha256_before": policy_before,
            "policy_sha256_after_original": policy_after_original,
            "policy_sha256_after_mitigated": policy_after_mitigated,
            "gradient_updates": False,
            "action_selection": "stochastic_with_seed_reset_per_condition",
            "evaluation_seeds_original": list(evaluation_seeds),
            "evaluation_seeds_mitigated": list(evaluation_seeds),
            "training_seed": manifest.training_seed,
            "topology_seed": manifest.topology_seed,
            "topology_config_sha256": topology_config.config_hash(),
            "reconstructed_topology_sha256": topology_hash(base_description),
            "environment_config_sha256": manifest.environment_config_hash,
            "cve_catalogue_sha256": digest(catalogue),
            "reward_config_sha256": reward_config.hash(),
            "training_code_commit": manifest.git_commit,
            "evaluation_code_commit": git_commit(),
            "evaluation_git_dirty": git_dirty(),
            "dependency_lock_sha256": manifest.dependency_lock_hash,
            "container_image_digest": manifest.docker_image_digest,
            "source_manifest_sha256": sha256_file(manifest_path),
            "protected_artifacts_before": protected_before,
            "protected_artifacts_after": protected_after,
        },
    }
    report = {"report_id": canonical_sha256(report_body), **report_body}
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if postgres:
        import psycopg

        from rlredteam.storage.mitigation_store import MitigationStore
        from rlredteam.storage.postgres_logger import connection_string, ensure_schema

        with psycopg.connect(connection_string()) as connection:
            ensure_schema(connection)
            MitigationStore(connection).save(report)
    return report


def _pair_bundles(
    original: EvaluationBundle,
    mitigated: EvaluationBundle,
    expected_seeds: list[int],
) -> list[dict[str, Any]]:
    original_by_seed = {episode.evaluation_seed: episode for episode in original.episodes}
    mitigated_by_seed = {episode.evaluation_seed: episode for episode in mitigated.episodes}
    if list(original_by_seed) != expected_seeds or list(mitigated_by_seed) != expected_seeds:
        raise MitigationError("conditions did not evaluate the identical ordered seed set")
    original_steps = _steps_by_seed(original)
    mitigated_steps = _steps_by_seed(mitigated)
    pairs = []
    for seed in expected_seeds:
        left = _condition(original_by_seed[seed], original_steps.get(seed, []))
        right = _condition(mitigated_by_seed[seed], mitigated_steps.get(seed, []))
        pairs.append(
            {
                "evaluation_seed": seed,
                "original": left,
                "mitigated": right,
                "delta": {
                    "success": int(right["success"]) - int(left["success"]),
                    "steps": right["steps"] - left["steps"],
                    "reward": right["reward"] - left["reward"],
                    "crown_jewel_reach": int(right["crown_jewel_reach"])
                    - int(left["crown_jewel_reach"]),
                },
            }
        )
    return pairs


def _steps_by_seed(bundle: EvaluationBundle) -> dict[int, list[dict[str, Any]]]:
    output: dict[int, list[dict[str, Any]]] = {}
    for step in bundle.steps:
        output.setdefault(int(step["evaluation_seed"]), []).append(step)
    return output


def _condition(episode, steps: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [step for step in steps if step.get("success")]
    techniques = sorted(
        {
            (mapping["framework"], mapping["technique_id"], mapping["technique_name"])
            for step in steps
            for mapping in step.get("framework_mappings", [])
        }
    )
    return {
        "success": bool(episode.goal_reached),
        "steps": int(episode.length),
        "reward": float(episode.native_return),
        "policy_reward": float(episode.policy_return),
        "crown_jewel_reach": any(
            step.get("success") and step.get("is_crown_jewel") for step in steps
        ),
        "terminal_reason": episode.terminal_reason,
        "mitigation_blocked_attempts": sum(
            bool(step.get("mitigation_blocked")) for step in steps
        ),
        "mitre_techniques": [
            {"framework": item[0], "technique_id": item[1], "technique_name": item[2]}
            for item in techniques
        ],
        "observed_path": [
            {
                "step": int(step["step"]),
                "action": step["action"],
                "action_kind": step["action_kind"],
                "target": step.get("target"),
                "cve_id": step.get("cve_id"),
                "framework_mappings": step.get("framework_mappings", []),
            }
            for step in successful
        ],
    }


def analyse_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    if not pairs:
        raise MitigationError("paired analysis requires at least one pair")
    original_success = [int(pair["original"]["success"]) for pair in pairs]
    mitigated_success = [int(pair["mitigated"]["success"]) for pair in pairs]
    original_rate = sum(original_success) / len(pairs)
    mitigated_rate = sum(mitigated_success) / len(pairs)
    absolute = mitigated_rate - original_rate
    relative = absolute / original_rate if original_rate else None
    reward_deltas = [float(pair["delta"]["reward"]) for pair in pairs]
    step_deltas = [float(pair["delta"]["steps"]) for pair in pairs]
    success_deltas = [float(pair["delta"]["success"]) for pair in pairs]
    original_only = sum(left == 1 and right == 0 for left, right in zip(
        original_success, mitigated_success, strict=True
    ))
    mitigated_only = sum(left == 0 and right == 1 for left, right in zip(
        original_success, mitigated_success, strict=True
    ))
    discordant = original_only + mitigated_only
    mcnemar_p = (
        float(stats.binomtest(min(original_only, mitigated_only), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "episodes_per_condition": len(pairs),
        "original_success_rate": original_rate,
        "mitigated_success_rate": mitigated_rate,
        "original_success_rate_ci95_wilson": list(_wilson(sum(original_success), len(pairs))),
        "mitigated_success_rate_ci95_wilson": list(
            _wilson(sum(mitigated_success), len(pairs))
        ),
        "absolute_success_rate_change": absolute,
        "relative_success_rate_change": relative,
        "relative_success_rate_change_note": (
            "(mitigated-original)/original"
            if relative is not None
            else "undefined because original success rate is zero"
        ),
        "paired_success_change_ci95_bootstrap": list(_bootstrap_mean_ci(success_deltas)),
        "mcnemar_exact": {
            "original_success_mitigated_failure": original_only,
            "original_failure_mitigated_success": mitigated_only,
            "discordant_pairs": discordant,
            "p_value": mcnemar_p,
        },
        "reward_change": _paired_continuous(reward_deltas),
        "steps_change": _paired_continuous(step_deltas),
        "interpretation": (
            "Mitigation effect is measured by paired frozen-policy reruns; observed path "
            "criticality remains a separate trace-derived quantity."
        ),
    }


def _paired_continuous(deltas: list[float]) -> dict[str, Any]:
    mean = sum(deltas) / len(deltas)
    if all(math.isclose(value, deltas[0]) for value in deltas):
        t_p = 1.0 if math.isclose(deltas[0], 0.0) else 0.0
    else:
        t_p = float(stats.ttest_1samp(deltas, popmean=0.0).pvalue)
    nonzero = [value for value in deltas if not math.isclose(value, 0.0)]
    wilcoxon_p = float(stats.wilcoxon(nonzero).pvalue) if nonzero else 1.0
    return {
        "paired_mean_change": mean,
        "paired_mean_change_ci95_bootstrap": list(_bootstrap_mean_ci(deltas)),
        "paired_t_test_p_value": t_p,
        "wilcoxon_signed_rank_p_value": wilcoxon_p,
    }


def _bootstrap_mean_ci(values: list[float], *, resamples: int = 10_000) -> tuple[float, float]:
    import numpy as np

    sample = np.asarray(values, dtype=float)
    rng = np.random.default_rng(15015)
    means = rng.choice(sample, size=(resamples, len(sample)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _wilson(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def artifact_tree_digest(path: Path) -> str:
    """Content-address one protected file/tree without reading timestamps."""
    root = Path(path)
    if not root.exists():
        raise MitigationError(f"protected artefact does not exist: {root}")
    if root.is_file():
        return sha256_file(root)
    value = hashlib.sha256()
    for file_path in sorted(item for item in root.rglob("*") if item.is_file()):
        value.update(str(file_path.relative_to(root)).encode())
        value.update(bytes.fromhex(sha256_file(file_path)))
    return value.hexdigest()


def validate_report_identity(report: Mapping[str, Any]) -> None:
    report_id = str(report.get("report_id", ""))
    body = {key: value for key, value in report.items() if key != "report_id"}
    if report_id != canonical_sha256(body):
        raise MitigationError("mitigation report identity does not match its content")


__all__ = [
    "CRITICALITY_LABEL",
    "CveMitigationOverlay",
    "EFFECT_LABEL",
    "INTERVENTION_VERSION",
    "MitigationError",
    "MitigationSpec",
    "REPORT_SCHEMA_VERSION",
    "analyse_pairs",
    "artifact_tree_digest",
    "canonical_sha256",
    "evaluate_checkpoint_mitigation",
    "promote_phase14_cve",
    "validate_report_identity",
]

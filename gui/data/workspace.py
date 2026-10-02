"""Read-only workspace projection. No environments, policies or database writes.

Knowledge conversion deliberately accepts ONLY recorded event dictionaries, not
ScenarioSummary or TrueTopology. Analyst graph data has a separate loader.
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import replace
from pathlib import Path

from gui.data.baseline import PACKAGE, _checked, _number, _path, _read, _sha, load_baseline
from gui.data.confirmatory import METRICS, load_confirmatory
from gui.data.models import (
    AgentKnowledgeView,
    AttackTimelineEntry,
    AttackTrajectorySummary,
    ComparisonOption,
    CurrentActionView,
    ExperimentComparisonRow,
    ExperimentReport,
    ResearchEvidenceSummary,
    ScenarioSummary,
    WorkspaceEvidence,
)

FROZEN = "results/convergence_v2/first_stable.json"
SNAPSHOT = "runs/experiment_01-sparse-s42-t42/config.snapshot.json"
DIAGNOSTICS = (
    "approx_kl",
    "clip_fraction",
    "explained_variance",
    "entropy_loss",
    "policy_gradient_loss",
    "value_loss",
    "learning_rate",
    "mean_episode_reward",
    "mean_episode_length",
)
WITHHELD = "Withheld from scientific comparison due to discovery-counting defect."
ERRORS = (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError)


def _json(path):
    value = json.loads(_read(path))
    if not isinstance(value, dict):
        raise ValueError("expected evidence object")
    return value


def _csv(blob):
    rows = list(csv.DictReader(io.StringIO(blob.decode())))
    if len(rows) > 100000:
        raise ValueError("evidence exceeds desktop row limit")
    return rows


def numeric(value):
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if _number(value) else None
    except (TypeError, ValueError):
        return None


def target_name(value):
    if isinstance(value, tuple | list):
        return "(" + ", ".join(map(str, value)) + ")"
    return str(value) if value is not None else "Unknown"


def knowledge_view(record: dict | None) -> AgentKnowledgeView:
    """Whitelisted snapshot projection; no join against hidden node metadata."""
    if not isinstance(record, dict):
        return AgentKnowledgeView()
    raw_ids = record.get("nodes", record.get("discovered", []))
    if not isinstance(raw_ids, list) or len(raw_ids) > 4096:
        raise ValueError("invalid recorded knowledge nodes")
    ids = {str(value) for value in raw_ids if isinstance(value, str | int)}
    types, access = record.get("known_node_types", {}), record.get("access", {})
    services = record.get("known_services", {})
    nodes = tuple(
        {
            "id": node,
            "type": str(types.get(node, "unknown")),
            "attributes": {
                "access": str(access.get(node, "Unknown")),
                "services": list(services.get(node, [])),
                "source": "Recorded AgentKnowledge snapshot",
            },
        }
        for node in sorted(ids)
    )
    edges = tuple(
        {
            "source": str(edge["source"]),
            "target": str(edge["target"]),
            "type": str(edge["type"]),
            "evidence": "Recorded AgentKnowledge snapshot",
        }
        for edge in record.get("edges", [])
        if isinstance(edge, dict)
        and str(edge.get("source")) in ids
        and str(edge.get("target")) in ids
        and edge.get("type")
    )
    return AgentKnowledgeView(
        nodes=nodes,
        edges=edges,
        services=tuple(
            f"{node}: {value}" for node in sorted(ids) for value in services.get(node, [])
        ),
        vulnerabilities=tuple(map(str, record.get("known_vulnerabilities", []))),
        credentials=tuple(map(str, record.get("credentials", []))),
        available_actions=tuple(map(str, record.get("available_semantic_actions", []))),
        status="Recorded AgentKnowledge only; unspecified facts remain unknown",
    )


def timeline_entries(events, *, legacy=False, source="Recorded episode"):
    """Keep missing before-state unknown; never backfill it from later steps."""
    if not isinstance(events, list | tuple) or len(events) > 10000:
        raise ValueError("invalid or excessive episode length")
    entries, previous = [], AgentKnowledgeView()
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError("invalid trajectory event")
        before = (
            knowledge_view(event["knowledge_before"]) if "knowledge_before" in event else previous
        )
        after = knowledge_view(event.get("knowledge"))
        # A missing snapshot is not equivalent to empty knowledge and is not
        # reconstructed from chosen targets, CVE metadata or future events.
        delta = event.get("knowledge_delta")
        if delta is None and isinstance(event.get("knowledge"), dict):
            delta = {"recorded_outcomes": event.get("outcomes", []), "snapshot_available": True}
        mappings = tuple(
            f"{item.get('tactic_id', '')} {item.get('tactic_name', '')} / "
            f"{item.get('technique_id', '')} {item.get('technique_name', '')}"
            for item in event.get("framework_mappings", [])
            if isinstance(item, dict)
        )
        breakdown = event.get("reward_breakdown") or {
            key: value for key, value in event.items() if key.endswith("_term")
        }
        action = CurrentActionView(
            step=int(event.get("step", index)),
            kind=str(event.get("action_kind", "Unknown")),
            name=str(event.get("simulator_action") or event.get("action", "Unknown")),
            target=target_name(event.get("target", event.get("target_entity"))),
            cve=str(event.get("cve_id") or "Unknown / not recorded"),
            cvss=numeric(event.get("cvss_base", event.get("cvss_score"))),
            mitre=mappings,
            native_reward=numeric(event.get("native_reward")),
            shaped_reward=numeric(event.get("shaped_reward")),
            recorded_reward=numeric(event.get("policy_reward", event.get("reward"))),
            breakdown=tuple(
                (str(key), numeric(value))
                for key, value in breakdown.items()
                if numeric(value) is not None
            ),
            success=event.get("success") if type(event.get("success")) is bool else None,
            access_gained=str(event.get("access_gained", "Unknown")),
            newly_discovered=None if legacy else event.get("newly_discovered"),
            knowledge_delta=(
                WITHHELD + " Full knowledge snapshots are not reconstructed."
                if legacy
                else json.dumps(delta, sort_keys=True)
                if delta is not None
                else "Unknown: not recorded"
            ),
            source=f"{source} · step {event.get('step', index)}",
        )
        entries.append(
            AttackTimelineEntry(action, before, after, event.get("is_crown_jewel") is True)
        )
        previous = after
    return tuple(entries)


def compare_options(left: ComparisonOption, right: ComparisonOption):
    if not left.cohort or left.cohort != right.cohort:
        return ()  # Similar metric names do not establish comparable protocols.
    a, b = dict(left.metrics), dict(right.metrics)
    return tuple(
        ExperimentComparisonRow(
            metric,
            left.label,
            right.label,
            a.get(metric),
            b.get(metric),
            b[metric] - a[metric]
            if a.get(metric) is not None and b.get(metric) is not None
            else None,
        )
        for metric in sorted(METRICS)
        if metric in a and metric in b
    )


def _context(root):
    baseline = load_baseline(root)
    if baseline.state != "REVIEWED":
        raise ValueError(baseline.detail)
    frozen = _json(_path(root, FROZEN))
    candidate = frozen["candidate"]
    directory = _path(root, f"results/convergence_v2/{candidate}")
    manifest = _json(_path(root, f"{PACKAGE}/report/reviewed_manifest.json"))["files"]
    return baseline, frozen, directory, manifest


def _report_blob(root, manifest, name):
    return _checked(_path(root, f"{PACKAGE}/report/{name}"), manifest[name])


def _scenario(root, frozen):
    path = _path(root, SNAPSHOT)
    snap = _json(path)
    if snap["topology_hash"] != frozen["inputs"]["topology_hash"]:
        raise ValueError("scenario snapshot does not match frozen topology hash")
    topo = snap["topology"]
    subnets = topo.get("subnets", [])
    hosts = topo.get("hosts", [])
    if len(hosts) > 4096 or len(subnets) > 512:
        raise ValueError("scenario exceeds graph limits")
    jewels = tuple(sorted(topo.get("sensitive_hosts", {})))
    nodes = [
        {"id": f"subnet:{i}", "type": "network_segment", "attributes": {}}
        for i in range(len(subnets))
    ]
    edges = []
    for subnet, host in hosts:
        node_id = target_name([subnet, host])
        nodes.append(
            {
                "id": node_id,
                "type": "host",
                "attributes": {
                    "crown_jewel": node_id in jewels,
                    "access": "Unknown: no episode selected",
                },
            }
        )
        edges.append(
            {
                "source": f"subnet:{subnet}",
                "target": node_id,
                "type": "CONTAINS",
                "evidence": SNAPSHOT,
            }
        )
    for i, row in enumerate(topo.get("topology", [])):
        for j, connected in enumerate(row):
            if connected and i < j:
                edges.append(
                    {
                        "source": f"subnet:{i}",
                        "target": f"subnet:{j}",
                        "type": "CONNECTED_TO",
                        "evidence": SNAPSHOT,
                    }
                )
    return ScenarioSummary(
        title=str(topo.get("name", "Stored NASim scenario")),
        simulator="NASim",
        topology_seed=snap.get("topology_seed"),
        reward_mode=str(snap.get("reward_mode", "Unknown")),
        training_seed=snap.get("training_seed"),
        host_count=len(hosts),
        subnet_count=len(subnets),
        crown_jewels=jewels,
        catalogue=str(snap.get("cve_manifest_sha256", "Unknown")),
        provenance=tuple(
            (key, str(snap.get(key, "Unknown")))
            for key in (
                "topology_hash",
                "environment_config_hash",
                "git_commit",
                "cve_manifest_sha256",
                "experiment_id",
            )
        )
        + (("snapshot", SNAPSHOT), ("snapshot_sha256", _sha(_read(path)))),
        true_nodes=tuple(nodes),
        true_edges=tuple(edges),
        status="Stored scenario snapshot; matching frozen topology hash. Services per host "
        "were not persisted and are not invented. No environment was regenerated.",
    )


def load_workspace(root: Path) -> WorkspaceEvidence:
    result = WorkspaceEvidence()
    try:
        baseline, frozen, directory, manifest = _context(root)
        scenario = ScenarioSummary()
        try:
            scenario = _scenario(root, frozen)
        except ERRORS as exc:
            scenario = replace(scenario, status=f"Scenario not yet available: {exc}")
        choices, comparisons = [], []
        for arm in ("shaped", "sparse"):
            rows = _csv(_report_blob(root, manifest, f"{arm}_evaluation.csv"))
            choices.extend(
                (
                    f"Frozen {arm} · train {r['training_seed']} · eval {r['evaluation_seed']}",
                    arm,
                    int(r["training_seed"]),
                    int(r["evaluation_seed"]),
                )
                for r in rows
            )
            summary = baseline.summary[arm]
            comparisons.append(
                ComparisonOption(
                    f"Frozen {arm}",
                    "fixed-nasim-aggregate",
                    tuple(
                        (m, numeric(summary.get("steps_to_goal" if m == "steps_to_success" else m)))
                        for m in sorted(METRICS)
                    ),
                    "Frozen policies, held-out episodes; descriptive only",
                )
            )
        for row in _csv(_report_blob(root, manifest, "paired_policy_results.csv")):
            for arm in ("shaped", "sparse"):
                comparisons.append(
                    ComparisonOption(
                        f"Frozen {arm} · seed {row['training_seed']}",
                        "fixed-nasim-seed",
                        tuple(
                            (
                                m,
                                numeric(
                                    row.get(
                                        f"{arm}_{'steps_to_goal' if m == 'steps_to_success' else m}"
                                    )
                                ),
                            )
                            for m in sorted(METRICS)
                        ),
                        "Per-policy evaluation means; descriptive only",
                    )
                )
        corrected = load_confirmatory(root)
        if corrected.comparison_available:
            for arm in ("shaped", "sparse"):
                comparisons.append(
                    ComparisonOption(
                        f"Corrected {arm}",
                        "fixed-nasim-aggregate",
                        tuple(
                            (r.metric, r.corrected) for r in corrected.comparisons if r.arm == arm
                        ),
                        corrected.comparison_path,
                    )
                )
        result = WorkspaceEvidence(
            scenario=scenario,
            episode_choices=tuple(choices),
            comparisons=tuple(comparisons),
            status="Reviewed local evidence",
        )
        diagnostics, convergence = [], []
        assessment = json.loads(
            _checked(_path(directory, "assessment.json"), frozen["assessment_sha256"])
        )
        for seed, checkpoint in frozen["checkpoints"].items():
            blob = _checked(
                _path(directory, f"shaped-{int(seed)}/diagnostics.csv"),
                checkpoint["files"]["diagnostics.csv"],
            )
            rows = tuple(
                {key: numeric(row.get(key)) for key in ("timesteps", *DIAGNOSTICS)}
                for row in _csv(blob)
            )
            diagnostics.append((int(seed), rows))
            item = assessment["per_seed"][seed]
            convergence.append(
                (
                    seed,
                    "PASS" if item["passed"] else "FAIL",
                    "; ".join(item.get("reasons", [])) or "Recorded criterion passed",
                )
            )
        aggregate = _report_blob(root, manifest, "figures/shaped-training.png")
        research = ResearchEvidenceSummary(
            status="Hash-checked frozen shaped diagnostics; no new statistical analysis",
            diagnostics=tuple(diagnostics),
            convergence=tuple(convergence),
            images=(("Frozen shaped aggregate reward", aggregate),),
            criterion=json.dumps(assessment["criterion"], indent=2, sort_keys=True),
        )
        return replace(result, research=research)
    except ERRORS as exc:
        return replace(result, status=f"Workspace evidence not yet available: {exc}")


def load_workspace_episode(root: Path, arm: str, seed: int, evaluation_seed: int):
    try:
        if (
            arm not in {"shaped", "sparse"}
            or type(seed) is not int
            or type(evaluation_seed) is not int
        ):
            raise ValueError("invalid episode selection")
        _, frozen, directory, manifest = _context(root)
        rows = _csv(_report_blob(root, manifest, f"{arm}_evaluation.csv"))
        matches = [
            row
            for row in rows
            if int(row["training_seed"]) == seed and int(row["evaluation_seed"]) == evaluation_seed
        ]
        if len(matches) != 1:
            raise ValueError("episode absent or ambiguous")
        row = matches[0]
        matched = _json(_path(directory, "final-evaluation/matched_outcomes.json"))
        if matched["frozen_sha256"] != _sha(_read(_path(root, FROZEN))):
            raise ValueError("trajectory refers to a different frozen candidate")
        relative = f"{arm}-{seed}/steps.jsonl"
        blob = _checked(
            _path(directory, f"final-evaluation/{relative}"), matched["files"][relative]
        )
        events = [json.loads(line) for line in blob.decode().splitlines() if line.strip()]
        events = [e for e in events if e.get("evaluation_seed") == evaluation_seed]
        if len(events) != int(row["length"]):
            raise ValueError("trajectory length differs from recorded episode")
        if any(
            e.get("step") != index
            or e.get("training_seed") != seed
            or e.get("run_name") != row["run_name"]
            or type(e.get("success")) is not bool
            or numeric(e.get("native_reward")) is None
            for index, e in enumerate(events)
        ):
            raise ValueError("trajectory identity or step sequence differs from recorded episode")
        native_total = sum(float(e["native_reward"]) for e in events)
        if not math.isclose(native_total, float(row["native_return"]), abs_tol=1e-7):
            raise ValueError("trajectory native return differs from reviewed outcome")
        if sum(not e["success"] for e in events) != int(row["failed_actions"]):
            raise ValueError("trajectory failures differ from reviewed outcome")
        if arm == "shaped":
            events = [dict(e, shaped_reward=e.get("policy_reward")) for e in events]
        title = f"Frozen {arm} · training seed {seed} · evaluation seed {evaluation_seed}"
        metrics = tuple(
            (label, str(row.get(key) or "Unknown"))
            for label, key in (
                ("Native return", "native_return"),
                ("Failed actions", "failed_actions"),
                ("Mean CVSS", "mean_cvss_exploited"),
                ("Compromised hosts", "hosts_compromised"),
                ("MITRE tactics observed", "observed_mitre_tactics"),
            )
        )
        exploited = sorted(
            {
                str(e["cve_id"])
                for e in events
                if e.get("success") is True
                and e.get("action_kind") in {"exploit", "privesc"}
                and e.get("cve_id")
            }
        )
        success = row["goal_reached"] == "True"
        report = ExperimentReport(
            title=title,
            outcome="SUCCESS" if success else "FAILURE",
            metrics=(
                ("Steps to success", row["length"] if success else "Not reached"),
                ("Exploited CVEs", ", ".join(exploited) or "None recorded"),
            )
            + metrics,
            provenance=(
                ("Training seed", str(seed)),
                ("Evaluation seed", str(evaluation_seed)),
                ("Experiment ID", row["run_name"]),
                ("Policy hash", row.get("policy_sha256", "Unknown")),
                ("Trajectory SHA256", _sha(blob)),
                ("Source commit", frozen["inputs"]["git_commit"]),
            ),
            limitation=WITHHELD
            + " Full before/after AgentKnowledge was not persisted in this trace.",
        )
        return AttackTrajectorySummary(
            title,
            timeline_entries(events, legacy=True, source=relative),
            report,
            "Recorded frozen-policy evaluation; replay only",
        )
    except ERRORS as exc:
        return AttackTrajectorySummary(status=f"Episode not yet available: {exc}")


def simulation_workspace(result):
    """Project the existing demo result; never change its policy or reward."""
    entries = timeline_entries(result.events, source="In-memory offline feasibility demo")
    scenario = ScenarioSummary(
        title=result.topology_name,
        simulator="Enterprise graph feasibility simulator",
        topology_seed=result.topology_seed,
        reward_mode="Existing feasibility reward (not PPO)",
        true_nodes=tuple(result.nodes),
        true_edges=tuple(result.edges),
        host_count=sum(
            n.get("type") in {"host", "server", "legacy_host", "cloud_workload"}
            for n in result.nodes
        ),
        subnet_count=sum(
            n.get("type") in {"network_segment", "cloud_network"} for n in result.nodes
        ),
        provenance=(("topology_hash", result.topology_hash), ("policy", result.agent)),
        status="Transient feasibility demonstration. Not frozen PPO research evidence.",
    )
    report = ExperimentReport(
        title=result.topology_name,
        outcome="SUCCESS" if result.goal_reached else "FAILURE",
        scope="Simulation demonstration — deterministic feasibility agent, not PPO evaluation",
        metrics=(
            (
                "Steps to success",
                str(result.episode_steps) if result.goal_reached else "Not reached",
            ),
            ("Recorded demo reward", str(result.total_reward)),
            ("Native / shaped decomposition", "Unknown: not recorded by demo"),
        ),
        provenance=scenario.provenance,
        limitation="Do not compare demo rewards with NASim PPO returns.",
    )
    return scenario, AttackTrajectorySummary(result.topology_name, entries, report, report.scope)

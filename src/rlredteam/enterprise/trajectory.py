"""Dependency-light causal reconstruction for stored enterprise trajectories."""

from __future__ import annotations


def _normalise_atom(atom: str) -> set[str]:
    prefix, separator, entity = atom.partition(":")
    if not separator:
        return {atom}
    atoms = {atom}
    if prefix in {
        "known_host",
        "known_service",
        "known_component",
        "known_target",
        "known_identity",
    }:
        atoms.add(f"known:{entity}")
    if prefix in {"authenticated", "pivoted", "root_access", "user_access", "read_access"}:
        atoms.add(f"access:{entity}")
    if prefix == "discovered":
        atoms.update({f"known:{entity}", f"reachable:{entity}"})
    if prefix in {"access_target", "network"}:
        atoms.add(f"reachable:{entity}")
    if prefix == "vulnerability":
        atoms.add(f"known_vulnerability:{entity}")
    if prefix == "credential_source":
        atoms.add(f"known:{entity}")
    return atoms


def reconstruct_attack_path(episode_steps: list[dict]) -> list[dict]:
    """Back-chain a causal path from recorded prerequisites and outcomes only."""
    progress = [
        step for step in episode_steps if step.get("success") and step.get("state_changed")
    ]
    goals = [step for step in progress if step.get("goal_reached")]
    if not goals:
        return []
    final = goals[-1]
    selected = [final]
    required = {
        atom
        for item in final.get("prerequisites", [])
        for atom in _normalise_atom(str(item))
    }
    for step in reversed(progress[: progress.index(final)]):
        produced = {
            atom
            for item in step.get("outcomes", [])
            for atom in _normalise_atom(str(item))
        }
        if not produced & required:
            continue
        selected.append(step)
        required -= produced
        required.update(
            atom
            for item in step.get("prerequisites", [])
            for atom in _normalise_atom(str(item))
        )
    return list(reversed(selected))

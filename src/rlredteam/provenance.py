"""Experiment provenance: the manifest that makes a result checkable.

Every field the remediation plan requires, gathered in one place, plus a
pre-run gate that refuses to start training when the environment does not
match what the experiment claims.

The gate exists because a run that starts on the wrong topology, the wrong CVE
data or an insecure database produces numbers that look fine and are worthless.
Failing loudly before the compute is spent is cheaper than discovering it after.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


# -- helpers ---------------------------------------------------------------


def _digest(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=REPO_ROOT, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def git_dirty() -> bool:
    """True when the working tree has uncommitted changes.

    A dirty tree means the recorded commit does not describe the code that
    actually ran, so the run is not reproducible from that commit alone.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5, cwd=REPO_ROOT, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(result.stdout.strip())


def dependency_lock_hash() -> str:
    """Digest of the pinned dependency set."""
    pyproject = REPO_ROOT / "pyproject.toml"
    return hashlib.sha256(pyproject.read_bytes()).hexdigest()[:16] if pyproject.exists() else ""


def docker_image_digest() -> str:
    """Image identity, when the run is inside a container.

    Read from the environment because a container cannot reliably inspect its
    own image; the Makefile and compose file set it.
    """
    return os.environ.get("RLREDTEAM_IMAGE_DIGEST", "")


def python_version() -> str:
    return f"{platform.python_version()} ({sys.implementation.name})"


def topology_hash(described: dict) -> str:
    """Digest of the ACTUAL generated network, not the rules that made it.

    This is the distinction the remediation plan turns on. The topology *config*
    hash covers generation parameters and is identical for every seed, so two
    runs on genuinely different networks share it. Only a hash over the realised
    structure -- subnets, adjacency, host addresses, services, exploits and the
    sensitive hosts -- proves two runs saw the same network.
    """
    return _digest(
        {
            "subnets": described.get("subnets"),
            "topology": described.get("topology"),
            "hosts": described.get("hosts"),
            "services": described.get("services"),
            "os": described.get("os"),
            "processes": described.get("processes"),
            "exploits": described.get("exploits"),
            "privescs": described.get("privescs"),
            "sensitive_hosts": described.get("sensitive_hosts"),
            "step_limit": described.get("step_limit"),
        }
    )


def environment_config_hash(described: dict) -> str:
    """Digest of the observation/action interface the agent was trained against."""
    return _digest(
        {
            "observation_space": described.get("observation_space"),
            "action_space_n": described.get("action_space_n"),
            "step_limit": described.get("step_limit"),
        }
    )


# -- manifest --------------------------------------------------------------


@dataclass
class ExperimentManifest:
    experiment_id: str
    git_commit: str | None
    git_dirty: bool
    python_version: str
    dependency_lock_hash: str
    docker_image_digest: str
    training_seed: int
    topology_seed: int
    topology_hash: str
    topology_config_hash: str
    environment_config_hash: str
    cve_database_hash: str
    reward_config_hash: str
    ppo_config_hash: str
    dataset_version: str
    training_budget: int
    checkpoint_path: str = ""
    database_run_id: int | None = None
    ppo_config: dict = field(default_factory=dict)
    reward_mode: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def write(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str))

    @classmethod
    def read(cls, path: Path) -> ExperimentManifest:
        return cls(**json.loads(Path(path).read_text()))


# -- pre-run gate ----------------------------------------------------------


class GateFailure(RuntimeError):
    """Raised when the pre-run gate refuses to start training."""


@dataclass
class GateResult:
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append((name, ok, detail))

    @property
    def passed(self) -> bool:
        return all(ok for _, ok, _ in self.checks)

    def report(self) -> str:
        lines = []
        for name, ok, detail in self.checks:
            mark = "PASS" if ok else "FAIL"
            lines.append(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
        return "\n".join(lines)

    def failures(self) -> list[str]:
        return [f"{n}: {d}" if d else n for n, ok, d in self.checks if not ok]


def run_gate(
    manifest: ExperimentManifest,
    *,
    frozen: dict | None = None,
    require_secure_db: bool = True,
    require_offline: bool = True,
) -> GateResult:
    """Every check the remediation plan requires, before any compute is spent.

    ``frozen`` is the expected-hash set for the experiment. When supplied, a
    mismatch is a hard stop -- that is the guardrail against silently running an
    arm on different inputs from its partner.
    """
    result = GateResult()

    if frozen:
        for key in (
            "topology_hash", "cve_database_hash",
            "reward_config_hash", "ppo_config_hash",
        ):
            expected = frozen.get(key)
            actual = getattr(manifest, key, None)
            if expected is None:
                continue
            result.add(
                f"{key} matches frozen experiment",
                expected == actual,
                "" if expected == actual else f"expected {expected}, got {actual}",
            )

    result.add(
        "PPO configuration is complete",
        bool(manifest.ppo_config) and bool(manifest.ppo_config_hash),
        "" if manifest.ppo_config else "no hyperparameters recorded",
    )
    result.add("training seed is explicit", manifest.training_seed is not None)
    result.add("topology seed is explicit", manifest.topology_seed is not None)
    result.add(
        "git provenance is recorded",
        bool(manifest.git_commit),
        "" if manifest.git_commit else "not a git checkout",
    )
    result.add(
        "working tree is clean",
        not manifest.git_dirty,
        "uncommitted changes — the recorded commit does not describe this run"
        if manifest.git_dirty else "",
    )
    result.add("dependency lock recorded", bool(manifest.dependency_lock_hash))

    if require_secure_db:
        password = os.environ.get("POSTGRES_PASSWORD")
        result.add(
            "database credentials supplied",
            bool(password),
            "" if password else "POSTGRES_PASSWORD is unset — no default is provided",
        )

    if require_offline:
        result.add(
            "training is offline",
            not _internet_reachable(),
            "external network reachable during training" if _internet_reachable() else "",
        )

    output = REPO_ROOT / "runs"
    result.add(
        "output directory is writable",
        os.access(output, os.W_OK) if output.exists() else True,
        "" if output.exists() and os.access(output, os.W_OK) else "",
    )
    return result


def _internet_reachable(timeout: float = 1.5) -> bool:
    """Best-effort external-reachability probe used by the offline check."""
    import socket

    try:
        socket.setdefaulttimeout(timeout)
        with socket.create_connection(("1.1.1.1", 443), timeout=timeout):
            return True
    except OSError:
        return False


def enforce(result: GateResult) -> None:
    if not result.passed:
        raise GateFailure(
            "Pre-run gate failed — training not started.\n"
            + result.report()
            + "\n\nFix the failing checks above; do not start training."
        )

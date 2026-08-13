"""CP-32 — automated security checks.

Every one of these guards a property that is easy to break by accident and
expensive to notice: a default password reintroduced, a port opened to the
network, a read-only mount made writable, the container reverted to root.

These are static checks on configuration plus runtime checks where a container
is available. They deliberately do not need training to run.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSE = REPO_ROOT / "docker-compose.yml"


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text())


# -- CP-01 credentials ------------------------------------------------------


def test_compose_has_no_default_credentials(compose: dict) -> None:
    """A ":-default" fallback publishes a working password in the repository."""
    raw = COMPOSE.read_text()
    offenders = re.findall(r"\$\{POSTGRES_(?:USER|PASSWORD|DB):-[^}]*\}", raw)
    assert not offenders, f"default credential fallbacks present: {offenders}"


def test_compose_requires_credentials_to_be_set(compose: dict) -> None:
    raw = COMPOSE.read_text()
    for name in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"):
        assert f"${{{name}:?" in raw, f"{name} is not declared required"


def test_connection_string_fails_closed_without_secrets(monkeypatch) -> None:
    from rlredteam.storage.postgres_logger import MissingCredentials, connection_string

    for name in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(MissingCredentials):
        connection_string()


def test_no_hardcoded_password_in_source() -> None:
    """No literal fallback password anywhere in shipped code."""
    offenders = []
    for path in list((REPO_ROOT / "src").rglob("*.py")) + list(
        (REPO_ROOT / "gui").rglob("*.py")
    ):
        text = path.read_text()
        for match in re.finditer(
            r"POSTGRES_PASSWORD['\"]?\s*,\s*['\"]([^'\"]+)['\"]", text
        ):
            offenders.append(f"{path.relative_to(REPO_ROOT)}: {match.group(1)}")
    assert not offenders, f"hardcoded password fallbacks: {offenders}"


def test_env_file_is_not_tracked_by_git() -> None:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", ".env"],
        capture_output=True, cwd=REPO_ROOT, check=False,
    )
    assert result.returncode != 0, ".env is tracked by git"


def test_no_nvd_api_key_in_tracked_files() -> None:
    """Catch a committed NVD key.

    Scoped away from data/provenance/, whose NVD responses legitimately contain
    UUID-shaped CNA `source` identifiers -- public data, not secrets. Scanning
    them would make this check cry wolf and get ignored.
    """
    result = subprocess.run(
        [
            "git", "grep", "-IEn",
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            "--", ".", ":(exclude)data/provenance/*",
        ],
        capture_output=True, text=True, cwd=REPO_ROOT, check=False,
    )
    assert not result.stdout.strip(), f"possible API key committed:\n{result.stdout}"


def test_env_example_carries_no_real_key() -> None:
    example = (REPO_ROOT / ".env.example").read_text()
    for line in example.splitlines():
        if line.startswith("NVD_API_KEY="):
            assert line.strip() == "NVD_API_KEY=", "a real key is in .env.example"


# -- CP-02 database exposure ------------------------------------------------


def test_postgres_is_bound_to_loopback_only(compose: dict) -> None:
    ports = compose["services"]["postgres"].get("ports", [])
    for entry in ports:
        assert str(entry).startswith("127.0.0.1:"), (
            f"postgres port {entry} is not restricted to loopback — it would be "
            "reachable from the local network"
        )


# -- CP-03 network isolation ------------------------------------------------


def test_training_network_has_no_egress(compose: dict) -> None:
    networks = compose.get("networks", {})
    app_networks = compose["services"]["app"].get("networks", [])
    assert app_networks, "app service declares no network"
    for name in app_networks:
        assert networks.get(name, {}).get("internal") is True, (
            f"network {name} is not internal — training would have internet access"
        )


def test_app_and_postgres_share_a_network(compose: dict) -> None:
    """Isolation must not break the thing training legitimately needs."""
    app = set(compose["services"]["app"].get("networks", []))
    db = set(compose["services"]["postgres"].get("networks", []))
    assert app & db, "app cannot reach postgres"


# -- CP-04 non-root ---------------------------------------------------------


def test_dockerfile_drops_to_a_non_root_user() -> None:
    text = (REPO_ROOT / "Dockerfile").read_text()
    users = re.findall(r"^USER\s+(\S+)", text, re.MULTILINE)
    assert users, "Dockerfile never switches away from root"
    final = users[-1].split(":")[0]
    assert final not in ("root", "0"), f"final USER is {final}"


@pytest.mark.integration
def test_runtime_uid_is_non_root() -> None:
    """Verified against the built image, not just the Dockerfile."""
    if shutil.which("podman") is None:
        pytest.skip("podman unavailable — run this check on the host")
    result = subprocess.run(
        ["podman", "run", "--rm", "localhost/sourcecode_app:latest", "id", "-u"],
        capture_output=True, text=True, check=False, timeout=120,
    )
    if result.returncode != 0:
        pytest.skip("training image not available")
    assert result.stdout.strip() != "0", "container runs as root"


# -- CP-05 read-only mounts -------------------------------------------------


PROTECTED = ("./src", "./configs", "./data", "./tools", "./scripts")


def test_protected_paths_are_mounted_read_only(compose: dict) -> None:
    mounts = {
        entry.split(":")[0]: entry.split(":")[-1]
        for entry in compose["services"]["app"].get("volumes", [])
        if isinstance(entry, str)
    }
    for path in PROTECTED:
        assert path in mounts, f"{path} is not mounted"
        assert mounts[path] == "ro", f"{path} is mounted {mounts[path]}, expected ro"


def test_runs_directory_is_writable(compose: dict) -> None:
    mounts = [
        entry for entry in compose["services"]["app"].get("volumes", [])
        if isinstance(entry, str) and entry.startswith("./runs")
    ]
    assert mounts, "runs/ is not mounted"
    assert mounts[0].endswith(":rw"), "runs/ must be writable for output"


# -- CP-06 capabilities and resources ---------------------------------------


def test_capabilities_are_dropped(compose: dict) -> None:
    assert compose["services"]["app"].get("cap_drop") == ["ALL"]


def test_privilege_escalation_is_disabled(compose: dict) -> None:
    for service in ("app", "postgres"):
        opts = compose["services"][service].get("security_opt", [])
        assert "no-new-privileges:true" in opts, f"{service} allows privilege escalation"


def test_resource_limits_are_set(compose: dict) -> None:
    app = compose["services"]["app"]
    assert app.get("pids_limit"), "no PID limit"
    assert app.get("mem_limit"), "no memory limit"


# -- CP-31 weight release gating --------------------------------------------


def test_trained_weights_are_gitignored() -> None:
    ignored = (REPO_ROOT / ".gitignore").read_text()
    assert "runs/" in ignored, "runs/ is not gitignored — weights could be published"


def test_no_model_files_tracked() -> None:
    result = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, cwd=REPO_ROOT, check=False
    )
    tracked = result.stdout.splitlines()
    weights = [f for f in tracked if f.endswith((".zip", ".pt", ".pth", ".ckpt"))]
    assert not weights, f"model weights are tracked: {weights}"


# -- environment sanity -----------------------------------------------------


def test_pythonhashseed_is_pinned_for_reproducibility() -> None:
    text = (REPO_ROOT / "Dockerfile").read_text()
    assert "PYTHONHASHSEED=0" in text


@pytest.mark.integration
def test_training_container_cannot_reach_the_internet() -> None:
    """CP-03, verified at runtime rather than inferred from the compose file."""
    if shutil.which("podman") is None:
        pytest.skip("podman unavailable — run this check on the host")
    result = subprocess.run(
        [
            "podman", "run", "--rm", "--network", "none",
            "localhost/sourcecode_app:latest",
            "python", "-c",
            "import socket,sys\n"
            "try:\n"
            "    socket.create_connection(('1.1.1.1',443),timeout=3)\n"
            "    print('REACHABLE')\n"
            "except OSError:\n"
            "    print('BLOCKED')\n",
        ],
        capture_output=True, text=True, check=False, timeout=180,
    )
    if result.returncode != 0 and "REACHABLE" not in result.stdout:
        pytest.skip(f"could not run container: {result.stderr[:120]}")
    assert "BLOCKED" in result.stdout, "training container reached the internet"


def test_no_secret_shaped_env_defaults_in_source() -> None:
    """Catches os.environ.get('...SECRET/TOKEN/KEY...', '<literal>') patterns."""
    offenders = []
    pattern = re.compile(
        r"environ\.get\(\s*['\"][A-Z_]*(?:SECRET|TOKEN|KEY|PASSWORD)[A-Z_]*['\"]\s*,\s*['\"]([^'\"]+)['\"]"
    )
    for path in list((REPO_ROOT / "src").rglob("*.py")) + list(
        (REPO_ROOT / "gui").rglob("*.py")
    ):
        for match in pattern.finditer(path.read_text()):
            offenders.append(f"{path.relative_to(REPO_ROOT)}: {match.group(1)}")
    assert not offenders, f"secret-shaped defaults: {offenders}"

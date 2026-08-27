"""Qt-free training request model shared by tests and the desktop worker."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass


def host_repo_root() -> str:
    """Return the repository path as seen by the host container engine."""
    return os.environ.get("RLREDTEAM_HOST_REPO", os.getcwd())


def training_network() -> str:
    """Return the isolated compose network used by training."""
    return os.environ.get("RLREDTEAM_NETWORK", "sourcecode_rlredteam-internal")


@dataclass(frozen=True)
class TrainRequest:
    seed: int = 42
    topology_seed: int = 42
    timesteps: int = 200_000
    reward_mode: str = "sparse"
    postgres: bool = True
    log_steps: bool = True

    @property
    def run_name(self) -> str:
        return f"{self.reward_mode}-s{self.seed}-t{self.topology_seed}"

    def argv(self) -> list[str]:
        args = [
            "podman",
            "run",
            "--rm",
            "--network",
            training_network(),
            "-v",
            f"{host_repo_root()}:/app:z",
            "-w",
            "/app",
            "-e",
            f"POSTGRES_USER={os.environ.get('POSTGRES_USER', '')}",
            "-e",
            f"POSTGRES_PASSWORD={os.environ.get('POSTGRES_PASSWORD', '')}",
            "-e",
            f"POSTGRES_DB={os.environ.get('POSTGRES_DB', '')}",
            "-e",
            "POSTGRES_HOST=postgres",
            "-e",
            "POSTGRES_PORT=5432",
            "-e",
            f"RLREDTEAM_GIT_DIRTY={os.environ.get('RLREDTEAM_GIT_DIRTY', '0')}",
            "localhost/sourcecode_app:latest",
            "python",
            "-m",
            "rlredteam.train",
            "--seed",
            str(self.seed),
            "--topology-seed",
            str(self.topology_seed),
            "--timesteps",
            str(self.timesteps),
            "--reward-config",
            f"configs/{self.reward_mode}.yaml",
        ]
        if self.postgres:
            args.append("--postgres")
        if self.log_steps:
            args.append("--log-steps")
        return args

    @property
    def command_line(self) -> str:
        """Return a display-safe command with the password redacted."""
        redacted = [
            "POSTGRES_PASSWORD=********"
            if arg.startswith("POSTGRES_PASSWORD=")
            else arg
            for arg in self.argv()
        ]
        return " ".join(shlex.quote(arg) for arg in redacted)

"""Launches training as a subprocess and streams its output.

Never in-process. The GUI image has no torch and no nasim by design, a 200k-step
run would block the Qt event loop and freeze the window, and an in-process crash
would take the whole app down. QProcess is already asynchronous, so this needs
no thread.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass

from PySide6.QtCore import QObject, QProcess, Signal


def host_repo_root() -> str:
    """The repository path as the HOST sees it.

    The training container is started by the host's engine, so its bind mount
    must use a host path. Inside the GUI container the repo appears at /app,
    which the host does not have -- mounting that would silently give training
    an empty directory. The launcher passes the real path in.
    """
    return os.environ.get("RLREDTEAM_HOST_REPO", os.getcwd())


def training_network() -> str:
    """The isolated network training must run on (CP-03).

    Not `host`. Host networking would reach postgres, but it also reaches the
    internet, and the pre-run gate refuses to train with an open network. The
    compose project's internal network gives the database and nothing else.
    """
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
        # `podman run`, deliberately not `podman compose run`. Compose rebuilds
        # the project network, which severs the database connection of anything
        # already running -- that is how an earlier 20-run grid was killed.
        args = [
            "podman", "run", "--rm",
            "--network", training_network(),
            "-v", f"{host_repo_root()}:/app:z",
            "-w", "/app",
            "-e", f"POSTGRES_USER={os.environ.get('POSTGRES_USER', '')}",
            "-e", f"POSTGRES_PASSWORD={os.environ.get('POSTGRES_PASSWORD', '')}",
            "-e", f"POSTGRES_DB={os.environ.get('POSTGRES_DB', '')}",
            "-e", "POSTGRES_HOST=postgres",
            "-e", "POSTGRES_PORT=5432",
            "-e", f"RLREDTEAM_GIT_DIRTY={os.environ.get('RLREDTEAM_GIT_DIRTY', '0')}",
            "localhost/sourcecode_app:latest",
            "python", "-m", "rlredteam.train",
            "--seed", str(self.seed),
            "--topology-seed", str(self.topology_seed),
            "--timesteps", str(self.timesteps),
            "--reward-config", f"configs/{self.reward_mode}.yaml",
        ]
        if self.postgres:
            args.append("--postgres")
        if self.log_steps:
            args.append("--log-steps")
        return args

    @property
    def command_line(self) -> str:
        """The command, with secrets redacted.

        This string is shown on screen and gets projected during demos, so the
        database password must not appear in it. The value passed to the
        process itself is unaffected.
        """
        redacted = []
        for arg in self.argv():
            if arg.startswith("POSTGRES_PASSWORD="):
                redacted.append("POSTGRES_PASSWORD=********")
            else:
                redacted.append(arg)
        return " ".join(shlex.quote(a) for a in redacted)


class Trainer(QObject):
    """Wraps one QProcess. Emits output lines, progress and completion."""

    output = Signal(str)
    started = Signal(str)  # run name
    finished = Signal(int, bool)  # exit code, was_stopped_by_user
    failed_to_start = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._process: QProcess | None = None
        self._request: TrainRequest | None = None
        self._stopping = False

    @property
    def request(self) -> TrainRequest | None:
        return self._request

    def is_running(self) -> bool:
        return (
            self._process is not None
            and self._process.state() != QProcess.ProcessState.NotRunning
        )

    def start(self, request: TrainRequest) -> None:
        if self.is_running():
            raise RuntimeError("a training run is already active")

        self._request = request
        self._stopping = False

        process = QProcess(self)
        # Merge stderr into stdout: SB3 and Python tracebacks go to stderr, and
        # a log that omits them would hide exactly the failures worth seeing.
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._drain)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_error)

        argv = request.argv()
        self._process = process
        process.start(argv[0], argv[1:])
        if not process.waitForStarted(5000):
            self.failed_to_start.emit(
                f"Could not start {argv[0]} — is podman installed and on PATH?"
            )
            self._process = None
            return
        self.started.emit(request.run_name)

    def stop(self) -> None:
        """Terminate, then kill if it does not go quietly."""
        if not self.is_running():
            return
        self._stopping = True
        self._process.terminate()
        if not self._process.waitForFinished(3000):
            self._process.kill()

    def _drain(self) -> None:
        if self._process is None:
            return
        raw = bytes(self._process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        for line in raw.splitlines():
            if line.strip():
                self.output.emit(line.rstrip())

    def _on_finished(self, code: int, _status) -> None:
        self.finished.emit(code, self._stopping)
        self._process = None

    def _on_error(self, error) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self.failed_to_start.emit(
                "Training process failed to start — check that podman is available."
            )

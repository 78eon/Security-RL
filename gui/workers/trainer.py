"""Launches training as a subprocess and streams its output.

Never in-process. The GUI image has no torch and no nasim by design, a 200k-step
run would block the Qt event loop and freeze the window, and an in-process crash
would take the whole app down. QProcess is already asynchronous, so this needs
no thread.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass

from PySide6.QtCore import QObject, QProcess, Signal


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
            "podman", "compose", "run", "--rm", "app",
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
        return " ".join(shlex.quote(a) for a in self.argv())


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

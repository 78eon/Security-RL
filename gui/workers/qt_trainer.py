"""Asynchronous Qt process controller for containerized training."""

from __future__ import annotations

from PySide6.QtCore import QObject, QProcess, Signal

from gui.workers.train_request import TrainRequest


class Trainer(QObject):
    """Wrap one training process and stream its output without blocking Qt."""

    output = Signal(str)
    started = Signal(str)
    finished = Signal(int, bool)
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

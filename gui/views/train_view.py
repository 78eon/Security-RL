"""Train — launch and monitor.

QProcess out to podman; the live curve is polled from episodes.csv, which
train.py flushes after every episode. Polling that file is more robust across a
bind mount than QFileSystemWatcher, and it works whether or not --postgres was
passed.
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui import theme
from gui.data import runs as runs_folder
from gui.data.repository import Repository
from gui.widgets.common import Banner, label, section_label
from gui.workers.trainer import Trainer, TrainRequest


class TrainView(QWidget):
    run_completed = Signal()

    def __init__(self, repository: Repository) -> None:
        super().__init__()
        self.repository = repository
        self.trainer = Trainer(self)
        self._episode_count = 0

        self._poll = QTimer(self)
        self._poll.setInterval(1000)
        self._poll.timeout.connect(self._poll_csv)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(theme.SP_L, theme.SP_L, theme.SP_L, theme.SP_L)
        outer.setSpacing(theme.SP_M)

        head = QVBoxLayout()
        head.setSpacing(2)
        head.addWidget(label("Train", "viewTitle"))
        head.addWidget(
            label("QProcess out to podman, live curve from episodes.csv", "viewSubtitle")
        )
        outer.addLayout(head)

        self.banner_slot = QVBoxLayout()
        outer.addLayout(self.banner_slot)

        body = QHBoxLayout()
        body.setSpacing(theme.SP_M)
        body.addWidget(self._build_form())
        body.addWidget(self._build_monitor(), 1)
        outer.addLayout(body, 1)

        self.trainer.output.connect(self._on_output)
        self.trainer.started.connect(self._on_started)
        self.trainer.finished.connect(self._on_finished)
        self.trainer.failed_to_start.connect(self._on_failed_to_start)

    # -- construction ----------------------------------------------------

    def _build_form(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("panel")
        panel.setFixedWidth(theme.DETAIL_WIDTH)
        column = QVBoxLayout(panel)
        column.setContentsMargins(
            theme.PANEL_PAD, theme.PANEL_PAD, theme.PANEL_PAD, theme.PANEL_PAD
        )
        column.setSpacing(theme.SP_M)

        column.addWidget(label("Launch a run", "panelHeading"))

        form = QFormLayout()
        form.setSpacing(theme.SP_S)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        self.reward_box = QComboBox()
        self.reward_box.addItems(["sparse", "shaped"])
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 99999)
        self.seed_spin.setValue(42)
        self.topology_spin = QSpinBox()
        self.topology_spin.setRange(0, 99999)
        self.topology_spin.setValue(42)
        self.timesteps_spin = QSpinBox()
        self.timesteps_spin.setRange(1000, 5_000_000)
        self.timesteps_spin.setSingleStep(10_000)
        self.timesteps_spin.setValue(200_000)
        self.timesteps_spin.setGroupSeparatorShown(True)

        form.addRow("Reward mode", self.reward_box)
        form.addRow("Seed", self.seed_spin)
        form.addRow("Topology seed", self.topology_spin)
        form.addRow("Timesteps", self.timesteps_spin)
        column.addLayout(form)

        self.postgres_check = QCheckBox("--postgres")
        self.postgres_check.setChecked(True)
        self.log_steps_check = QCheckBox("--log-steps")
        self.log_steps_check.setChecked(True)
        self.log_steps_check.setToolTip("Required for the Replay view")
        column.addWidget(self.postgres_check)
        column.addWidget(self.log_steps_check)
        column.addWidget(label("--log-steps is needed for replay", "sectionLabel"))

        column.addWidget(section_label("Command"))
        self.command_preview = QPlainTextEdit()
        self.command_preview.setReadOnly(True)
        self.command_preview.setFixedHeight(88)
        column.addWidget(self.command_preview)

        buttons = QHBoxLayout()
        buttons.setSpacing(theme.SP_S)
        self.launch_button = QPushButton("Launch")
        self.launch_button.setProperty("primary", True)
        self.launch_button.clicked.connect(self._launch)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setProperty("danger", True)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        buttons.addWidget(self.launch_button)
        buttons.addWidget(self.stop_button)
        column.addLayout(buttons)
        column.addStretch(1)

        for widget in (self.reward_box, self.seed_spin, self.topology_spin,
                       self.timesteps_spin):
            signal = getattr(widget, "valueChanged", None) or widget.currentIndexChanged
            signal.connect(self._update_command)
        self.postgres_check.toggled.connect(self._update_command)
        self.log_steps_check.toggled.connect(self._update_command)
        self._update_command()
        return panel

    def _build_monitor(self) -> QWidget:
        holder = QWidget()
        holder.setObjectName("surface")
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(theme.SP_S)

        self.status_label = label("No run active", "panelHeading")
        column.addWidget(self.status_label)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 100)
        column.addWidget(self.progress)

        column.addWidget(section_label("Live native return"))
        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(180)
        self.plot.showGrid(x=True, y=True, alpha=0.12)
        self.plot.setMenuEnabled(False)
        self.curve = self.plot.plot([], [], pen=pg.mkPen(theme.ARM_1, width=2))
        column.addWidget(self.plot, 1)

        column.addWidget(section_label("Process output · stdout + stderr"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        column.addWidget(self.log, 1)
        return holder

    # -- request ---------------------------------------------------------

    def _request(self) -> TrainRequest:
        return TrainRequest(
            seed=self.seed_spin.value(),
            topology_seed=self.topology_spin.value(),
            timesteps=self.timesteps_spin.value(),
            reward_mode=self.reward_box.currentText(),
            postgres=self.postgres_check.isChecked(),
            log_steps=self.log_steps_check.isChecked(),
        )

    def _update_command(self) -> None:
        self.command_preview.setPlainText(self._request().command_line)

    # -- lifecycle -------------------------------------------------------

    def is_running(self) -> bool:
        return self.trainer.is_running()

    def _launch(self) -> None:
        self._clear_banners()
        self.log.clear()
        self.curve.setData([], [])
        self._episode_count = 0
        try:
            self.trainer.start(self._request())
        except RuntimeError as exc:
            self._show_banner(Banner(str(exc), "", kind="warn"))

    def stop(self) -> None:
        self.trainer.stop()

    def _on_started(self, run_name: str) -> None:
        self.status_label.setText(f"{run_name} · running")
        self.launch_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress.setRange(0, 0)  # indeterminate until the first episode
        self._poll.start()

    def _on_output(self, line: str) -> None:
        self.log.appendPlainText(line)

    def _on_finished(self, code: int, stopped_by_user: bool) -> None:
        self._poll.stop()
        self._poll_csv()
        self.launch_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(100 if code == 0 else 0)

        request = self.trainer.request
        name = request.run_name if request else "run"
        if stopped_by_user:
            self.status_label.setText(f"{name} · stopped")
            self._show_banner(
                Banner(
                    "Run stopped",
                    "Episodes completed before the stop are still recorded and "
                    "remain valid.",
                    kind="warn",
                )
            )
        elif code == 0:
            self.status_label.setText(f"{name} · complete")
            self._show_banner(Banner("Run complete", name, kind="ok"))
        else:
            self.status_label.setText(f"{name} · failed")
            tail = self.log.toPlainText().strip().splitlines()[-3:]
            self._show_banner(
                Banner(
                    f"Training failed with exit code {code}",
                    "Any episodes that completed before the failure are kept.",
                    kind="error",
                    detail="\n".join(tail),
                )
            )
        self.run_completed.emit()

    def _on_failed_to_start(self, message: str) -> None:
        self.launch_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._poll.stop()
        self._show_banner(
            Banner(
                message,
                "The GUI shells out to the training container rather than "
                "training in-process, so podman must be reachable from here.",
                kind="error",
            )
        )

    # -- live curve ------------------------------------------------------

    def _poll_csv(self) -> None:
        request = self.trainer.request
        if request is None:
            return
        rows = runs_folder.read_episode_csv(request.run_name)
        if not rows:
            return
        self._episode_count = len(rows)
        xs, ys = [], []
        for row in rows:
            try:
                xs.append(int(row["episode_idx"]))
                ys.append(float(row["native_return"]))
            except (KeyError, ValueError):
                continue
        if xs:
            self.curve.setData(xs, ys)

        try:
            steps = int(rows[-1]["timesteps"])
            total = request.timesteps
            self.progress.setRange(0, 100)
            self.progress.setValue(min(100, int(100 * steps / total)))
            self.status_label.setText(
                f"{request.run_name} · running · episode "
                f"{theme.fmt_num(self._episode_count)} · "
                f"{theme.fmt_num(steps)} / {theme.fmt_num(total)} steps"
            )
        except (KeyError, ValueError, ZeroDivisionError):
            pass

    # -- banners ---------------------------------------------------------

    def _show_banner(self, banner: Banner) -> None:
        self._clear_banners()
        self.banner_slot.addWidget(banner)

    def _clear_banners(self) -> None:
        while self.banner_slot.count():
            item = self.banner_slot.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

"""Compatibility facade for training requests and the native Qt worker."""

from __future__ import annotations

from gui.workers.train_request import TrainRequest, host_repo_root, training_network

__all__ = ["TrainRequest", "host_repo_root", "training_network"]


def __getattr__(name: str):
    """Import Qt only when the process controller itself is requested."""
    if name == "Trainer":
        from gui.workers.qt_trainer import Trainer

        return Trainer
    raise AttributeError(name)

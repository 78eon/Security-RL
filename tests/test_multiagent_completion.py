from __future__ import annotations

from pathlib import Path

import pytest

from rlredteam.enterprise.multiagent_completion import (
    MultiAgentCompletionError,
    require,
    verify_multiagent_completion,
)


def test_completion_require_fails_closed() -> None:
    require(True, "unused")
    with pytest.raises(MultiAgentCompletionError, match="required evidence"):
        require(False, "required evidence")


def test_completion_rejects_repository_without_frozen_evidence(tmp_path: Path) -> None:
    with pytest.raises((MultiAgentCompletionError, OSError), match="missing|No such file"):
        verify_multiagent_completion(tmp_path)

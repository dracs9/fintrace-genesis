"""Suite-wide fixtures."""

import pytest


@pytest.fixture(autouse=True)
def _no_privilege_drop(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Docker image sets these for the real runner. In-process CLI calls in tests must not
    switch the pytest process to another UID, so tests opt in explicitly (see test_trajectory_cli)."""
    monkeypatch.delenv("FINTRACE_GRADER_USER", raising=False)
    monkeypatch.delenv("FINTRACE_AGENT_USER", raising=False)

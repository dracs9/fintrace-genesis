"""Integration gates that become runnable when Role 2 supplies fintrace.env.core."""

from pathlib import Path
from typing import Any, Iterator

import pytest

from fintrace.grader import GraderClient


@pytest.fixture
def environment(tmp_path: Path) -> Iterator[Any]:
    core = pytest.importorskip("fintrace.env.core", reason="requires the Role 2 environment",
                               exc_type=ModuleNotFoundError)
    with GraderClient(tmp_path, log_dir=tmp_path / "logs") as client:
        env = core.AuditEnv(data_root=tmp_path, mode="train", grader=client)
        try:
            yield env
        finally:
            env.close()


def test_second_submit_is_episode_done(environment: Any) -> None:
    _, info = environment.reset(90001, "medium")
    empty = {"episode_id": info["episode_id"], "findings": []}
    action = {"tool": "submit", "args": {"findings": empty}}
    _, _, terminated, truncated, _ = environment.step(action)
    assert terminated and not truncated
    obs, _, _, _, _ = environment.step(action)
    assert obs["error"] == {"code": "EPISODE_DONE", "message": "episode is not active"}


def test_timeout_uses_all_inspection_steps(environment: Any) -> None:
    environment.reset(90002, "medium")
    for i in range(100):
        _, reward, terminated, truncated, info = environment.step(
            {"tool": "calculator", "args": {"expression": "1+1"}})
        assert not terminated and truncated == (i == 99)
        if not truncated:
            assert reward == 0
    grade = info["grade"]
    assert grade["valid"] and grade["tp"] == grade["fp"] == 0
    assert reward == (0.8 if grade["fn"] == 0 else -1.0)

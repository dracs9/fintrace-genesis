"""Gymnasium wrapper passes env_checker (SPEC §8.6)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

gymnasium = pytest.importorskip("gymnasium")
from gymnasium.utils.env_checker import check_env  # noqa: E402

from env_helpers import FakeGraderClient  # noqa: E402
from auditgym.env.gym_adapter import AuditGymEnv  # noqa: E402


def test_check_env(tmp_path: Path) -> None:
    env = AuditGymEnv(tmp_path, config="easy", grader=FakeGraderClient())
    check_env(env, skip_render_check=True)


def test_json_round_trip_and_spaces(tmp_path: Path) -> None:
    env = AuditGymEnv(tmp_path, config="easy", grader=FakeGraderClient())
    obs, info = env.reset(seed=90_020)
    assert env.observation_space.contains(obs)
    episode_id = json.loads(obs)["result"]["manifest"]["episode_id"]
    assert info["episode_id"] == episode_id
    obs, reward, term, trunc, _ = env.step(json.dumps({"tool": "list_records", "args": {"table": "fx"}}))
    assert env.observation_space.contains(obs) and json.loads(obs)["ok"] is True
    obs, *_ = env.step("{not json")
    assert json.loads(obs)["error"]["code"] == "BAD_ARGS"
    submit = {"tool": "submit", "args": {"findings": {"episode_id": episode_id, "findings": []}}}
    obs, reward, term, trunc, info = env.step(json.dumps(submit))
    assert term is True and reward == pytest.approx(1 - 0.004) and "grade" in info


def test_same_seed_same_observation(tmp_path: Path) -> None:
    env = AuditGymEnv(tmp_path)
    assert env.reset(seed=5)[0] == env.reset(seed=5)[0]
    assert env.reset(seed=5)[0] != env.reset(seed=6)[0]

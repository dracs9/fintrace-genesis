"""AuditEnv on episodes produced by Role 1's generator; isolation and leakage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from env_helpers import FakeGraderClient, ScriptedPolicy, check_info, check_obs, leaks
from auditgym.env import AuditEnv
from auditgym.env.core import expected_episode_id, load_config

SEEDS = (90_001, 90_002)


@pytest.fixture(scope="module")
def data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("episodes")


@pytest.mark.parametrize("preset", ["easy", "medium", "hard", "default"])
def test_reset_generates_and_ids_agree(data_root: Path, preset: str) -> None:
    env = AuditEnv(data_root)
    obs, info = env.reset(SEEDS[0], preset)
    check_obs(obs)
    check_info(info)
    cfg = load_config(preset)
    assert info["episode_id"] == expected_episode_id(SEEDS[0], cfg) == obs["result"]["manifest"]["episode_id"]
    assert info["config_version"] == cfg["config_version"]
    assert obs["result"]["tables"] == obs["result"]["manifest"]["counts"]
    assert (data_root / info["episode_id"] / "open").is_dir()


def test_reset_accepts_dict_config(data_root: Path) -> None:
    cfg = load_config("easy")
    a = AuditEnv(data_root).reset(SEEDS[0], cfg)
    b = AuditEnv(data_root).reset(SEEDS[0], "easy")
    assert a == b


def test_unknown_preset_rejected(data_root: Path) -> None:
    for bad in ("nightmare", "../default", ""):
        with pytest.raises(ValueError):
            AuditEnv(data_root).reset(1, bad)


def test_two_resets_do_not_share_state(data_root: Path) -> None:
    env = AuditEnv(data_root, grader=FakeGraderClient())
    _, info_a = env.reset(SEEDS[0], "easy")
    obs, *_ = env.step({"tool": "list_records", "args": {"table": "ledger", "limit": 5}})
    cursor_a = obs["result"]["next_cursor"]
    first_a = obs["result"]["items"]
    obs_b, info_b = env.reset(SEEDS[1], "easy")
    assert info_a["episode_id"] != info_b["episode_id"]
    assert obs_b["steps_used"] == 0
    obs, *_ = env.step({"tool": "list_records", "args": {"table": "ledger", "cursor": cursor_a, "limit": 5}})
    assert obs["error"]["code"] == "BAD_CURSOR" and obs["steps_used"] == 1
    obs, *_ = env.step({"tool": "list_records", "args": {"table": "ledger", "limit": 5}})
    assert obs["result"]["items"] != first_a
    # going back gives exactly the original view
    env.reset(SEEDS[0], "easy")
    obs, *_ = env.step({"tool": "list_records", "args": {"table": "ledger", "limit": 5}})
    assert obs["result"]["items"] == first_a and obs["result"]["next_cursor"] == cursor_a


def _explore(env: AuditEnv, seed: int) -> list[object]:
    """Touch every tool and error path; return everything the agent saw."""
    seen: list[object] = list(env.reset(seed, "hard"))
    policy = ScriptedPolicy()
    policy.reset(seen[0], seed=seed)  # type: ignore[arg-type]
    extra = [
        {"tool": "list_records", "args": {"table": "ledger"}},
        {"tool": "list_records", "args": {"table": "fx", "cursor": "bad"}},
        {"tool": "get_record", "args": {"table": "ledger", "id": "R-000000"}},
        {"tool": "get_document", "args": {"doc_id": "D-000000"}},
        {"tool": "calculator", "args": {"expression": "__import__('os').getcwd()"}},
        {"tool": "nope"},
    ]
    obs = seen[0]
    for action in extra:
        out = env.step(action)
        seen.extend(out)
    while True:
        out = env.step(policy.act(obs))  # type: ignore[arg-type]
        seen.extend(out)
        obs = out[0]
        check_obs(obs)
        if out[2] or out[3]:
            break
    seen.extend(env.step({"tool": "submit", "args": {"findings": {}}}))
    return seen


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_no_leakage_in_observations_info_errors(data_root: Path, mode: str) -> None:
    for seed in SEEDS:
        env = AuditEnv(data_root, mode=mode, grader=FakeGraderClient())  # type: ignore[arg-type]
        seen = _explore(env, seed)
        episode_id = seen[1]["episode_id"]  # type: ignore[index]
        truth = json.loads((data_root / episode_id / "private/truth.json").read_text("utf-8"))
        issue_ids = tuple(i["issue_id"] for i in truth["issues"])
        assert leaks(seen, extra=(str(data_root), data_root.name, *issue_ids)) == []

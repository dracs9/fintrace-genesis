"""AuditEnv reset/step contract (SPEC §8.1, §8.2, §8.5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from env_helpers import TINY_ID, FakeGraderClient, check_info, check_obs
from auditgym.env import AuditEnv
from auditgym.env import core
from auditgym.env.core import FINDING_TYPES, TASK, canonical_json

LIST_INV = {"tool": "list_records", "args": {"table": "invoices"}}


def _submit(findings: list | None = None) -> dict:
    return {"tool": "submit", "args": {"findings": {"episode_id": TINY_ID, "findings": findings or []}}}


def test_first_observation_matches_spec(env: AuditEnv, tiny_dir: Path) -> None:
    obs, info = env.reset_from_dir(tiny_dir)
    check_obs(obs)
    check_info(info)
    assert obs["ok"] is True and obs["error"] is None
    assert obs["steps_used"] == 0 and obs["steps_left"] == 100
    result = obs["result"]
    assert sorted(result) == ["finding_types", "manifest", "policy", "tables", "task", "tools"]
    assert result["task"] == TASK
    assert result["manifest"] == json.loads((tiny_dir / "open/manifest.json").read_text("utf-8"))
    assert result["policy"] == json.loads((tiny_dir / "open/policy.json").read_text("utf-8"))
    assert result["tables"] == {"ledger": 9, "invoices": 3, "payments": 1, "fx": 60}
    assert result["tools"] == ["list_records", "get_record", "get_document", "calculator", "submit"]
    assert result["finding_types"] == list(FINDING_TYPES)
    assert info == {"episode_id": TINY_ID, "schema_version": "1.0", "config_version": "tiny-1.0"}


def test_env_holds_no_hidden_dir_path(started: AuditEnv) -> None:
    def walk(value: object, seen: set[int]) -> None:
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, (str, Path)):
            parts = Path(str(value)).parts
            assert "private" not in parts and not str(value).endswith(("truth.json", "provenance.json"))
        elif isinstance(value, dict) or hasattr(value, "items"):
            for k, v in value.items():
                walk(k, seen)
                walk(v, seen)
        elif isinstance(value, (list, tuple)):
            for v in value:
                walk(v, seen)
        elif hasattr(value, "__dict__") or hasattr(value, "__dataclass_fields__"):
            for name in getattr(value, "__dataclass_fields__", None) or vars(value):
                walk(getattr(value, name), seen)

    walk(started, set())


def test_steps_count_and_rewards_zero_midway(started: AuditEnv) -> None:
    for n in range(1, 4):
        obs, reward, term, trunc, info = started.step(LIST_INV)
        check_obs(obs)
        assert (obs["steps_used"], obs["steps_left"]) == (n, 100 - n)
        assert (reward, term, trunc, info) == (0.0, False, False, {})


def test_submit_terminates_train(started: AuditEnv, grader: FakeGraderClient) -> None:
    started.step(LIST_INV)
    started.step({"tool": "nope"})
    obs, reward, term, trunc, info = started.step(_submit())
    check_obs(obs)
    assert obs["result"] == {"accepted": True}
    assert term is True and trunc is False
    assert grader.calls == [(TINY_ID, {"episode_id": TINY_ID, "findings": []}, 2)]
    assert reward == pytest.approx(1.0 - 0.004)
    assert sorted(info) == ["grade"]
    assert sorted(info["grade"]) == ["f1", "fn", "fp", "precision", "recall", "reward", "tp", "valid"]
    assert info["grade"]["reward"] == reward
    check_info(info)


def test_eval_mode_hides_reward(tiny_dir: Path) -> None:
    grader = FakeGraderClient()
    env = AuditEnv(tiny_dir.parent, mode="eval", grader=grader)
    env.reset_from_dir(tiny_dir)
    _, reward, term, _, info = env.step(_submit())
    assert term is True and reward == 0.0 and info == {}
    assert len(grader.calls) == 1  # the grader still sees the submission


def test_eval_mode_truncation_hides_reward(tiny_dir: Path) -> None:
    env = AuditEnv(tiny_dir.parent, mode="eval", grader=FakeGraderClient(), max_steps=2)
    env.reset_from_dir(tiny_dir)
    env.step(LIST_INV)
    _, reward, _, trunc, info = env.step(LIST_INV)
    assert trunc is True and reward == 0.0 and info == {}


def test_99_steps_then_submit_terminates(started: AuditEnv, grader: FakeGraderClient) -> None:
    for _ in range(99):
        _, _, term, trunc, _ = started.step(LIST_INV)
        assert not term and not trunc
    obs, _, term, trunc, _ = started.step(_submit())
    assert term is True and trunc is False
    assert obs["steps_used"] == 100 and obs["steps_left"] == 0
    assert grader.calls[0][2] == 99


def test_wall_timeout_scores_empty_once_and_reset_starts_a_new_clock(
    tiny_dir: Path, grader: FakeGraderClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 100.0
    monkeypatch.setattr(core, "monotonic", lambda: now)
    env = AuditEnv(tiny_dir.parent, grader=grader)
    env.reset_from_dir(tiny_dir)
    now = 159.9
    assert env.step(LIST_INV)[3] is False
    now = 160.0
    obs, _, terminated, truncated, _ = env.step(_submit())
    check_obs(obs)
    assert (terminated, truncated, obs["steps_used"]) == (False, True, 1)
    assert grader.calls == [(TINY_ID, None, 1)]
    assert env.step(_submit())[0]["error"]["code"] == "EPISODE_DONE"
    assert len(grader.calls) == 1
    env.reset_from_dir(tiny_dir)
    assert env.step(_submit())[2:4] == (True, False)


def test_step_budget_cannot_exceed_case_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="between 1 and 100"):
        AuditEnv(tmp_path, max_steps=101)


def test_100_steps_without_submit_truncates(started: AuditEnv, grader: FakeGraderClient) -> None:
    for _ in range(99):
        started.step(LIST_INV)
    obs, reward, term, trunc, info = started.step(LIST_INV)
    check_obs(obs)
    assert obs["ok"] is True and obs["steps_left"] == 0
    assert term is False and trunc is True
    assert grader.calls == [(TINY_ID, None, 100)]
    assert reward == -1.0 and info["grade"]["reward"] == -1.0


def test_failed_submit_on_last_step_truncates(tiny_dir: Path) -> None:
    grader = FakeGraderClient()
    env = AuditEnv(tiny_dir.parent, grader=grader, max_steps=1)
    env.reset_from_dir(tiny_dir)
    obs, _, term, trunc, _ = env.step({"tool": "submit", "args": {}})
    assert obs["error"]["code"] == "BAD_ARGS"
    assert (term, trunc) == (False, True)
    assert grader.calls == [(TINY_ID, None, 1)]


def test_error_steps_count_as_inspection(started: AuditEnv, grader: FakeGraderClient) -> None:
    started.step({"tool": "list_records", "args": {"table": "nope"}})
    started.step({"tool": "get_record", "args": {"table": "ledger", "id": "R-zzzzzz"}})
    started.step({"tool": "calculator", "args": {"expression": "2**8"}})
    started.step(_submit())
    assert grader.calls[0][2] == 3


def test_step_before_reset_is_episode_done(env: AuditEnv) -> None:
    obs, reward, term, trunc, info = env.step(LIST_INV)
    check_obs(obs)
    assert obs["error"] == {"code": "EPISODE_DONE", "message": "episode is not active"}
    assert obs["result"] is None and obs["steps_used"] == 0
    assert (reward, term, trunc, info) == (0.0, False, False, {})


@pytest.mark.parametrize("finisher", ["submit", "truncate"])
def test_step_after_end_is_episode_done(tiny_dir: Path, finisher: str) -> None:
    grader = FakeGraderClient()
    env = AuditEnv(tiny_dir.parent, grader=grader, max_steps=3)
    env.reset_from_dir(tiny_dir)
    if finisher == "submit":
        env.step(_submit())
    else:
        for _ in range(3):
            env.step(LIST_INV)
    for action in (LIST_INV, _submit()):  # includes a repeated submit
        obs, reward, term, trunc, info = env.step(action)
        check_obs(obs)
        assert obs["error"]["code"] == "EPISODE_DONE"
        assert reward == 0.0 and info == {}
        assert term or trunc
    assert len(grader.calls) == 1


def test_close_deactivates(started: AuditEnv) -> None:
    started.close()
    obs, *_ = started.step(LIST_INV)
    assert obs["error"]["code"] == "EPISODE_DONE"


def test_no_grader_gives_zero_terminal_reward(tiny_dir: Path) -> None:
    env = AuditEnv(tiny_dir.parent)
    env.reset_from_dir(tiny_dir)
    _, reward, term, _, info = env.step(_submit())
    assert term and reward == 0.0 and info == {}


def test_observation_is_a_copy(started: AuditEnv) -> None:
    obs, *_ = started.step({"tool": "get_document", "args": {"doc_id": "D-7q2x9a"}})
    obs["result"]["document"]["total_minor"] = -1
    again, *_ = started.step({"tool": "get_document", "args": {"doc_id": "D-7q2x9a"}})
    assert again["result"]["document"]["total_minor"] == 100000


def test_action_mutation_after_submit_does_not_change_findings(started: AuditEnv,
                                                               grader: FakeGraderClient) -> None:
    action = _submit([{"type": "duplicate_invoice"}])
    started.step(action)
    action["args"]["findings"]["findings"].clear()
    assert grader.calls[0][1]["findings"] == [{"type": "duplicate_invoice"}]


def test_reset_clears_state(started: AuditEnv, tiny_dir: Path) -> None:
    started.step(LIST_INV)
    started.step(_submit())
    obs, _ = started.reset_from_dir(tiny_dir)
    assert obs["steps_used"] == 0
    obs, _, term, _, _ = started.step(LIST_INV)
    assert obs["ok"] and obs["steps_used"] == 1 and not term


def _run(env: AuditEnv, tiny_dir: Path, actions: list[dict]) -> list[str]:
    obs, info = env.reset_from_dir(tiny_dir)
    trace = [canonical_json([obs, info])]
    for action in actions:
        trace.append(canonical_json(env.step(action)))
    return trace


def test_r2_same_actions_same_observations(tiny_dir: Path) -> None:
    first = list_page = {"tool": "list_records", "args": {"table": "fx", "limit": 7}}
    actions = [first, list_page, {"tool": "calculator", "args": {"expression": "1/3"}},
               {"tool": "get_record", "args": {"table": "fx", "id": "2026-03-02:USD:KZT"}},
               {"tool": "bogus"}, _submit()]
    a = _run(AuditEnv(tiny_dir.parent, grader=FakeGraderClient()), tiny_dir, actions)
    b = _run(AuditEnv(tiny_dir.parent, grader=FakeGraderClient()), tiny_dir, actions)
    assert a == b


def test_constructor_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        AuditEnv(tmp_path, mode="test")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        AuditEnv(tmp_path, max_steps=0)

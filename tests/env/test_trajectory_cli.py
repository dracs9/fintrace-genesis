"""Trajectory logging (SPEC §11) and the CLI (SPEC §15)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from env_helpers import FakeGraderClient, ScriptedPolicy, check_obs, leaks
from fintrace import cli
from fintrace.env import AuditEnv
from fintrace.env import core
from fintrace.logging.trajectory import SUMMARY_FIELDS, TrajectoryWriter, read_jsonl

TRAJ_FIELDS = sorted(["episode_id", "policy", "step", "action", "observation", "reward", "terminated",
                      "truncated", "schema_version", "config_version"])


class _SpyPolicy(ScriptedPolicy):
    def reset(self, first_obs: dict, *, seed: int) -> None:
        self.seen = [json.dumps(first_obs, sort_keys=True)]
        super().reset(first_obs, seed=seed)

    def act(self, obs: dict) -> dict:
        if len(self.seen) == 0 or json.dumps(obs, sort_keys=True) != self.seen[-1]:
            self.seen.append(json.dumps(obs, sort_keys=True))
        return super().act(obs)


def test_trajectory_matches_what_policy_saw(tmp_path: Path) -> None:
    fake = FakeGraderClient()
    grader = cli._RecordingGrader(fake)
    env = AuditEnv(tmp_path / "data", grader=grader)
    policy = _SpyPolicy()
    path = tmp_path / "runs" / "r1" / "trajectories.jsonl"
    with TrajectoryWriter(path, "scripted") as writer:
        summary = cli.run_policy_episode(env, policy, 90_010, "easy", writer, grader)
    lines = path.read_text("utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    for line, record in zip(lines, records):
        assert sorted(record) == TRAJ_FIELDS
        assert line == json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        check_obs(record["observation"])
    assert records[0]["step"] == 0 and records[0]["action"] is None and records[0]["reward"] == 0.0
    assert [r["step"] for r in records] == list(range(len(records)))
    assert records[-1]["terminated"] is True and records[-1]["action"]["tool"] == "submit"
    assert records[-1]["reward"] == summary.reward == fake.calls[0][2] * -0.002 + 1.0
    seen = [json.dumps(r["observation"], sort_keys=True) for r in records[:-1]]
    assert seen == policy.seen
    assert {r["episode_id"] for r in records} == {summary.episode_id}
    assert records[0]["config_version"] == "easy-1.0" and records[0]["schema_version"] == "1.0"
    assert leaks(records, extra=("grade",)) == []
    assert summary.steps == len(records) - 1 and summary.f1 == 1.0


def test_writer_requires_begin(tmp_path: Path) -> None:
    with TrajectoryWriter(tmp_path / "t.jsonl", "p") as writer:
        with pytest.raises(RuntimeError):
            writer.step(1, {}, {}, 0.0, False, False)


def test_trajectory_records_wall_timeout_without_logging_the_ignored_action(
    tiny_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter((100.0, 160.0))
    monkeypatch.setattr(core, "monotonic", lambda: next(ticks))
    grader = cli._RecordingGrader(FakeGraderClient())
    env = AuditEnv(tiny_dir.parent, grader=grader)
    obs, info = env.reset_from_dir(tiny_dir)
    path = tmp_path / "trajectory.jsonl"
    with TrajectoryWriter(path, "scripted") as writer:
        summary = cli._run_observation(env, ScriptedPolicy(), writer, grader, obs, info, 0.0)
    reset, terminal = read_jsonl(path)
    assert (reset["step"], terminal["step"]) == (0, 1)
    assert terminal["truncated"] and terminal["action"] is None
    assert summary.steps == terminal["observation"]["steps_used"] == 0
    assert grader.inner.calls == [(info["episode_id"], None, 0)]


def test_cli_run_with_external_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    code = cli.main(["run", "--policy", "env_helpers:ScriptedPolicy", "--split", "dev", "--n", "2",
                     "--config", "easy", "--run-id", "smoke"])
    assert code == 0
    run_dir = tmp_path / "runs" / "smoke"
    rows = read_jsonl(run_dir / "summary.jsonl")
    assert [sorted(r) for r in rows] == [sorted(SUMMARY_FIELDS)] * 2
    assert rows[0]["episode_id"] != rows[1]["episode_id"]
    trajectory = read_jsonl(run_dir / "trajectories.jsonl")
    assert sum(1 for r in trajectory if r["step"] == 0) == 2
    assert sorted(p.name for p in (tmp_path / "data" / "dev").iterdir()) == sorted(r["episode_id"] for r in rows)
    # same run again -> identical trajectories (R2 through the CLI)
    before = (run_dir / "trajectories.jsonl").read_bytes()
    assert cli.main(["run", "--policy", "env_helpers:ScriptedPolicy", "--split", "dev", "--n", "2",
                     "--config", "easy", "--run-id", "smoke"]) == 0
    assert (run_dir / "trajectories.jsonl").read_bytes() == before


def test_cli_generate_is_reproducible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                      capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    assert cli.main(["generate", "--split", "train", "--n", "2", "--config", "easy", "--out", "a"]) == 0
    assert cli.main(["generate", "--split", "train", "--n", "2", "--config", "easy", "--out", "b"]) == 0
    files_a = sorted(p.relative_to(tmp_path / "a") for p in (tmp_path / "a").rglob("*") if p.is_file())
    files_b = sorted(p.relative_to(tmp_path / "b") for p in (tmp_path / "b").rglob("*") if p.is_file())
    assert files_a == files_b and len(files_a) == 16
    for rel in files_a:
        assert (tmp_path / "a" / rel).read_bytes() == (tmp_path / "b" / rel).read_bytes()


@pytest.mark.parametrize("argv", [
    ["validate", "--data", "data"],
    ["transform", "--src", "x", "--dst", "y", "--permute-seed", "1", "--id-salt", "s"],
    ["report", "--runs", "runs"],
])
def test_missing_role_commands_report_not_implemented(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                      capsys: pytest.CaptureFixture[str], argv: list[str]) -> None:
    monkeypatch.chdir(tmp_path)
    try:
        cli._resolve(cli.ENTRY_POINTS[argv[0]])
    except cli.NotImplementedYet:
        pass
    else:
        pytest.skip("implemented by its owner")
    assert cli.main(argv) == cli.NOT_IMPLEMENTED
    assert "not implemented yet" in capsys.readouterr().err


def test_run_starts_grader_as_grader_user_then_drops_to_agent(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SPEC §9.5 order: grader worker first (with its account), then the runner drops to agent."""
    import fintrace.grader.privileges as privileges

    events: list[tuple[str, object]] = []

    class Grader(FakeGraderClient):
        def __init__(self, data_root: Path, *, run_as: str | None = None) -> None:
            super().__init__()
            events.append(("grader", run_as))

        def close(self) -> None:
            events.append(("close", None))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(cli.GRADER_USER_ENV, "grader")
    monkeypatch.setenv(cli.AGENT_USER_ENV, "agent")
    monkeypatch.setattr(cli, "_resolve", lambda spec: Grader if spec == cli.ENTRY_POINTS["grader"]
                        else pytest.fail(spec))
    monkeypatch.setattr(privileges, "drop_to", lambda user: events.append(("drop", user)))
    monkeypatch.setattr(cli, "load_policy", lambda spec: ScriptedPolicy())
    assert cli.main(["run", "--policy", "x", "--split", "dev", "--n", "1", "--config", "easy"]) == 0
    assert events == [("grader", "grader"), ("drop", "agent"), ("close", None)]


def test_no_privilege_env_means_no_drop(monkeypatch: pytest.MonkeyPatch) -> None:
    import fintrace.grader.privileges as privileges

    monkeypatch.delenv(cli.AGENT_USER_ENV, raising=False)
    monkeypatch.setattr(privileges, "drop_to", lambda user: pytest.fail("dropped"))
    cli.drop_agent_privileges()


def test_all_skips_unimplemented_policy_before_spawning(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(spec: str) -> None:
        raise cli.NotImplementedYet(spec)

    monkeypatch.setattr(cli, "load_policy", missing)
    monkeypatch.setattr(cli.subprocess, "call", lambda *a, **k: pytest.fail("spawned"))
    with pytest.raises(cli.NotImplementedYet):
        cli._run_subprocess("oracle", cli.build_parser().parse_args(["all"]))


def test_unknown_policy(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run", "--policy", "nope", "--split", "dev", "--n", "1"]) == 1
    assert cli.main(["run", "--policy", "no_such_pkg.mod:Cls", "--split", "dev", "--n", "1"]) == 1

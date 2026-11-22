"""Check R1 and R2 across the CLI public boundary with a nonempty audit key."""

import hashlib
import json
from pathlib import Path

import pytest

from auditgym.cli import main
from auditgym.env.core import load_config
from auditgym.episode.serialize import generate_episode
from auditgym.logging.trajectory import read_jsonl


def _tree_digest(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def test_r1_salted_episode_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config("hard")
    monkeypatch.setenv("AUDITGYM_SALT", "reproducibility-test")
    first = generate_episode(90_925, config, tmp_path / "first")
    second = generate_episode(90_925, config, tmp_path / "second")
    assert _tree_digest(first) == _tree_digest(second)


def test_r2_oracle_trajectory_and_reward(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AUDITGYM_SALT", "replay-oracle-test")
    command = ["run", "--policy", "oracle", "--split", "dev", "--config", "medium", "--n", "4"]
    assert main(command) == 0
    run_dir = tmp_path / "runs/oracle-dev-train"
    path = run_dir / "trajectories.jsonl"
    first = path.read_bytes()
    episodes = sorted((tmp_path / "data/dev").iterdir())
    assert any(json.loads((p / "private/truth.json").read_bytes())["issues"] for p in episodes)
    assert main(command) == 0
    assert path.read_bytes() == first
    for row in read_jsonl(run_dir / "summary.jsonl"):
        assert row["f1"] == 1
        assert row["reward"] == pytest.approx(1 - .002 * (row["steps"] - 1))
        assert 1 <= row["steps"] <= 100


def test_external_policy_gets_public_seed_and_no_salt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AUDITGYM_SALT", "external-policy-test")
    (tmp_path / "public_seed_policy.py").write_text(
        "import os\n"
        "assert 'AUDITGYM_SALT' not in os.environ\n"
        "class External:\n"
        "    name = 'external:v1'\n"
        "    def reset(self, first_obs: dict, *, seed: int) -> None:\n"
        "        assert seed == 0\n"
        "        self.ep = first_obs['result']['manifest']['episode_id']\n"
        "    def act(self, obs: dict) -> dict:\n"
        "        return {'tool': 'submit', 'args': {'findings': {'episode_id': self.ep, 'findings': []}}}\n",
        encoding="utf-8",
    )
    assert main(["run", "--policy", "public_seed_policy:External", "--split", "dev",
                 "--config", "easy", "--n", "1"]) == 0
    run = tmp_path / "runs/external_v1-dev-train"
    assert read_jsonl(run / "summary.jsonl")[0]["steps"] == 1
    assert {r["policy"] for r in read_jsonl(run / "trajectories.jsonl")} == {"external:v1"}
    assert main(["report", "--runs", "runs", "--data", "data"]) == 0


def test_eval_keeps_real_grade_in_the_grader_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["run", "--policy", "flag_all", "--mode", "eval", "--split", "dev", "--n", "1"]) == 0
    run = tmp_path / "runs/flag_all-dev-eval"
    summary, = read_jsonl(run / "summary.jsonl")
    assert all(summary[key] is None for key in ("precision", "recall", "f1", "reward"))
    assert all(r["reward"] == 0.0 for r in read_jsonl(run / "trajectories.jsonl"))
    grade, = read_jsonl(tmp_path / "logs/grader.jsonl")
    assert grade["episode_id"] == summary["episode_id"]
    assert grade["reward"] == -1.0

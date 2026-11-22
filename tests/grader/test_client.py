import json
from pathlib import Path
from typing import Any, IO

import pytest

from fintrace.grader import GraderClient, grade


def test_spawn_log_and_public_projection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    episode_id = "ep_111111aaaaaa"
    episode = tmp_path / episode_id
    (episode / "private").mkdir(parents=True)
    (episode / "open").mkdir()
    (episode / "open/manifest.json").write_text(json.dumps({"episode_id": episode_id}))
    row = {"type": "cutoff_error", "currency": "KZT", "affected_ids": ["doc:D-111111"]}
    issue = dict(row, issue_id="iss_only_here", expected_amount_minor=42, evidence_sets=[row["affected_ids"]])
    truth = episode / "private/truth.json"
    truth.write_text(json.dumps({"episode_id": episode_id, "issues": [issue]}))
    finding = dict(row, claimed_amount_minor=42, evidence_ids=row["affected_ids"], explanation="")
    envelope = {"episode_id": episode_id, "findings": [finding]}
    assert grade(episode, envelope, 10).matches == (("iss_only_here", 0),)
    # Child processes spawned with a fresh interpreter do not inherit this hook.
    original = Path.open

    def guard(path: Path, *args: Any, **kwargs: Any) -> IO:
        assert "private" not in path.parts, "parent tried to open the key"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard)
    logs = tmp_path / "logs"
    with GraderClient(tmp_path, log_dir=logs) as client:
        assert client._worker._start_method == "spawn"
        result = client.submit(episode_id, envelope, 10)
        assert (result.tp, result.reward, result.matches) == (1, 0.98, ())
        invalid = client.submit(episode_id, {}, 0)
        assert (invalid.valid, invalid.reward, invalid.error) == (False, -1, "invalid arguments")
        with pytest.raises(RuntimeError, match="^invalid arguments$"):
            client.submit("../private", None, 0)
    client.close()
    assert not client._worker.is_alive()
    entries = [json.loads(line) for line in (logs / "grader.jsonl").read_text().splitlines()]
    assert entries[0]["matches"] == [["iss_only_here", 0]]
    assert entries[1]["error"] == entries[2]["error"] == "invalid arguments"


def test_non_json_values_cannot_be_coerced_into_valid_submissions(tmp_path: Path) -> None:
    episode = "ep_000000000001"
    (tmp_path / "private").mkdir()
    (tmp_path / "open").mkdir()
    (tmp_path / "open/manifest.json").write_text(json.dumps({"episode_id": episode}))
    (tmp_path / "private/truth.json").write_text(json.dumps({"episode_id": episode, "issues": []}))
    with GraderClient(tmp_path, log_dir=tmp_path / "logs") as client:
        for malformed in ((), float("nan"), object()):
            result = client.submit(episode, {"episode_id": episode, "findings": malformed}, 0)
            assert not result.valid and result.reward == -1.0


def test_broken_key_does_not_disclose_paths(tmp_path: Path) -> None:
    with GraderClient(tmp_path, log_dir=tmp_path / "logs") as client:
        with pytest.raises(RuntimeError, match="^invalid arguments$"):
            client.submit("ep_aaaaaaaaaaaa", None, 100)

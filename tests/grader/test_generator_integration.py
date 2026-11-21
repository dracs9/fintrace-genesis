"""Offline cross-repository checks using Role 1's generator on PYTHONPATH."""

import csv
import json
from pathlib import Path

import pytest
import yaml

from auditgym.grader import GraderClient
from auditgym.grader.leakage import LeakIndex


def test_oracle_on_50_generated_episodes_and_eval_salt(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    generator = pytest.importorskip("auditgym.episode.serialize", reason="requires Role 1",
                                     exc_type=ModuleNotFoundError)
    config_path = Path(generator.__file__).resolve().parents[3] / "configs/default.yaml"
    config = yaml.safe_load(config_path.read_bytes())
    seen_types: set[str] = set()
    monkeypatch.setenv("AUDITGYM_SALT", "test-eval-salt")
    with GraderClient(tmp_path, log_dir=tmp_path / "logs") as client:
        for seed in range(90000, 90050):
            episode_dir = tmp_path / f"generated_{seed}"
            generator.generate_episode(seed, config, episode_dir)
            truth = json.loads((episode_dir / "private/truth.json").read_bytes())
            episode_id = truth["episode_id"]
            episode_dir = episode_dir.rename(tmp_path / episode_id)
            rows = []
            steps = 0
            for item in truth["issues"]:
                seen_types.add(item["type"])
                evidence = item["evidence_sets"][0]
                rows.append({"type": item["type"], "claimed_amount_minor": item["expected_amount_minor"],
                             "affected_ids": item["affected_ids"], "currency": item["currency"],
                             "evidence_ids": evidence, "explanation": ""})
                steps += len(evidence)
            result = client.submit(episode_id, {"episode_id": episode_id, "findings": rows}, steps)
            assert (result.tp, result.fp, result.fn, result.reward) == (len(rows), 0, 0, 1 - 0.002 * steps)
            index = LeakIndex.from_episode(episode_dir)
            for path in sorted((episode_dir / "open").iterdir()):
                with path.open(encoding="utf-8", newline="") as handle:
                    if path.suffix == ".csv":
                        index.check(list(csv.DictReader(handle)))
                    elif path.suffix == ".jsonl":
                        index.check([json.loads(line) for line in handle])
                    else:
                        index.check(json.load(handle))
    assert seen_types == {"duplicate_invoice", "payment_misallocation", "fx_rate_error", "cutoff_error"}
    monkeypatch.setenv("AUDITGYM_SALT", "second-test-salt")
    replay = generator.generate_episode(90049, config, tmp_path / "salted")
    salted = (replay / "open/manifest.json").read_bytes()
    assert json.loads(salted)["episode_id"] != episode_id
    replay = generator.generate_episode(90049, config, tmp_path / "replay")
    assert (replay / "open/manifest.json").read_bytes() == salted

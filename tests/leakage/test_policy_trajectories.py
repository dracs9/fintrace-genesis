"""Release check over recorded runs exported by the environment and four policies."""

import json
import os
from collections import defaultdict
from pathlib import Path

import pytest

from auditgym.grader.leakage import LeakIndex


def test_four_policy_trajectories_cover_50_episodes() -> None:
    data = os.environ.get("AUDITGYM_TEST_EPISODES")
    runs = os.environ.get("AUDITGYM_TEST_RUNS")
    if not data and not runs:
        pytest.skip("requires recorded Role 2/4 trajectories and their episodes")
    assert data and runs, "supply both fixture roots"
    episodes = {}
    for manifest in sorted(Path(data).rglob("open/manifest.json")):
        key = json.loads(manifest.read_bytes())["episode_id"]
        episodes[key] = manifest.parent.parent
    policies = {"oracle", "random_explorer", "flag_all", "reconciler"}
    complete: dict[str, set[str]] = defaultdict(set)
    started: set[tuple[str, str]] = set()
    indexes: dict[str, LeakIndex] = {}
    for path in sorted(Path(runs).rglob("trajectories.jsonl")):
        with path.open(encoding="utf-8") as records:
            for line in records:
                record = json.loads(line)
                episode_id, policy = record["episode_id"], record["policy"]
                if episode_id not in indexes:
                    indexes[episode_id] = LeakIndex.from_episode(episodes[episode_id])
                index = indexes[episode_id]
                index.check(record["observation"])
                for field in ("obs", "info", "error"):
                    if field in record:
                        index.check(record[field])
                pair = (episode_id, policy)
                if record["step"] == 0:
                    started.add(pair)
                if record["terminated"] or record["truncated"]:
                    assert pair in started
                    complete[episode_id].add(policy)
                    started.remove(pair)
    assert not started
    assert sum(policies <= complete[k] for k in sorted(complete)) >= 50

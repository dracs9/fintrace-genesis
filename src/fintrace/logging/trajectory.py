"""Trajectory and run-summary logs (SPEC §11).

``trajectories.jsonl``: one canonical-JSON line per step; step 0 is the reset with
``action: null``. Observations are written exactly as the policy saw them; ``info`` is
never written (it may hold the train-mode grade). Nothing from ``private/`` is read here.
``summary.jsonl``: one line per episode. Deterministic except ``wall_time_s``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Any

SUMMARY_FIELDS = ("episode_id", "policy", "precision", "recall", "f1", "reward", "steps", "wall_time_s")


def canonical_line(record: object) -> str:
    """Canonical JSON (SPEC §1) followed by ``\\n``."""
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False) + "\n"


class TrajectoryWriter:
    """Append-only writer of ``trajectories.jsonl`` for one run (many episodes)."""

    def __init__(self, path: Path, policy: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._policy = policy
        self._fh: IO[str] = self.path.open("w", encoding="utf-8", newline="\n")
        self._episode: dict[str, str] | None = None

    def begin(self, info: dict[str, Any], observation: dict[str, Any]) -> None:
        """Write step 0 of an episode from the reset ``info`` and first observation."""
        self._episode = {"episode_id": info["episode_id"], "schema_version": info["schema_version"],
                         "config_version": info["config_version"]}
        self._write(0, None, observation, 0.0, False, False)

    def step(self, step: int, action: Any, observation: dict[str, Any], reward: float,
             terminated: bool, truncated: bool) -> None:
        """Write one step exactly as the policy saw it."""
        if self._episode is None:
            raise RuntimeError("begin() must be called first")
        self._write(step, action, observation, reward, terminated, truncated)

    def _write(self, step: int, action: Any, observation: dict[str, Any], reward: float,
               terminated: bool, truncated: bool) -> None:
        assert self._episode is not None
        record = {**self._episode, "policy": self._policy, "step": step, "action": action,
                  "observation": observation, "reward": float(reward),
                  "terminated": bool(terminated), "truncated": bool(truncated)}
        try:
            line = canonical_line(record)
        except (TypeError, ValueError):  # a policy emitted a non-JSON action
            record["action"] = {"unserialisable": True}
            line = canonical_line(record)
        self._fh.write(line)

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "TrajectoryWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass(frozen=True)
class EpisodeSummary:
    """One row of ``summary.jsonl``. Metrics are ``None`` when no grader was available."""

    episode_id: str
    policy: str
    precision: float | None
    recall: float | None
    f1: float | None
    reward: float | None
    steps: int
    wall_time_s: float


def write_summary(path: Path, rows: list[EpisodeSummary]) -> Path:
    """Write ``summary.jsonl`` (one canonical line per episode, in run order)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(canonical_line(asdict(row)))
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file written by this module."""
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]

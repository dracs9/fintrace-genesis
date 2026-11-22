"""Test doubles and checks for Role 2 tests (stand-ins for Role 3/4 interfaces)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from fintrace.env.tools import load_schema

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TINY = FIXTURES / "ep_tiny"
TINY_ID = "ep_0123456789ab"

_OBS_VALIDATOR = Draft202012Validator(load_schema("observation.schema.json"))
FORBIDDEN = ("private", "truth", "provenance", "iss_", "inject", "seed", "causal", "mutation")


@dataclass(frozen=True)
class FakeGrade:
    """Same fields as SPEC §9.6 GradeResult."""

    episode_id: str
    valid: bool = True
    error: str | None = None
    tp: int = 1
    fp: int = 0
    fn: int = 0
    precision: float = 1.0
    recall: float = 1.0
    f1: float = 1.0
    inspection_steps: int = 0
    reward: float = 0.5
    matches: tuple[tuple[str, int], ...] = ()


@dataclass
class FakeGraderClient:
    """Records every ``submit`` call; reward = 1 - 0.002 * inspection_steps (or -1 for None)."""

    calls: list[tuple[str, Any, int]] = field(default_factory=list)

    def submit(self, episode_id: str, findings: dict | None, inspection_steps: int) -> FakeGrade:
        self.calls.append((episode_id, findings, inspection_steps))
        reward = -1.0 if findings is None else 1.0 - 0.002 * inspection_steps
        return FakeGrade(episode_id=episode_id, inspection_steps=inspection_steps, reward=reward)


class ScriptedPolicy:
    """Deterministic policy: lists ledger pages, opens every document seen, then submits."""

    name = "scripted"

    def reset(self, first_obs: dict, *, seed: int) -> None:
        self._episode_id = first_obs["result"]["manifest"]["episode_id"]
        self._queue: list[dict] = [{"tool": "list_records", "args": {"table": "invoices"}}]
        self._docs: list[str] = []

    def act(self, obs: dict) -> dict:
        result = obs.get("result") or {}
        for item in result.get("items", []):
            if "doc_id" in item and item["doc_id"] not in self._docs:
                self._docs.append(item["doc_id"])
                self._queue.append({"tool": "get_document", "args": {"doc_id": item["doc_id"]}})
        if self._queue:
            return self._queue.pop(0)
        findings = {"episode_id": self._episode_id, "findings": []}
        return {"tool": "submit", "args": {"findings": findings}}


def check_obs(obs: dict) -> None:
    """Every observation round-trips through JSON and matches the observation schema."""
    assert json.loads(json.dumps(obs)) == obs
    _OBS_VALIDATOR.validate(obs)
    assert obs["steps_used"] + obs["steps_left"] >= 0


def check_info(info: dict) -> None:
    assert isinstance(info, dict)
    assert json.loads(json.dumps(info)) == info


def leaks(value: Any, extra: tuple[str, ...] = ()) -> list[str]:
    """Return forbidden fragments found anywhere (keys and values) in ``value``."""
    text = json.dumps(value, ensure_ascii=False).lower()
    found = [w for w in FORBIDDEN + tuple(e.lower() for e in extra) if w in text]
    if "\\\\" in text or ":/" in text.replace("://", ""):
        found.append("<path>")
    return found

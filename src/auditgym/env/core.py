"""AuditEnv: Gymnasium-style JSON environment (SPEC §8).

Invariants:
* ``reset`` replaces the whole episode state; only ``open/`` is loaded and no path to
  ``private/`` is kept on the object;
* every observation is ``{"ok","result","error","steps_used","steps_left"}`` and is a fresh
  canonical-JSON round-tripped value, so callers cannot mutate env state through it;
* agent-visible errors are only the fixed codes/messages of SPEC §8.4; unexpected internal
  exceptions become ``BAD_ARGS`` and are logged to ``auditgym.env`` (never shown to the agent);
* intermediate rewards are 0.0; the terminal reward comes from the grader in ``train`` and is
  0.0 in ``eval``.
Deterministic: same episode + same actions -> identical observations and rewards (R2).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol

import yaml

from .errors import EnvError
from .tools import READ_TOOLS, EpisodeData, validate_action

GENERATOR_VERSION = "1.0.1"  # must match episode/serialize.py; verified against manifest in reset
FINDING_TYPES = ("duplicate_invoice", "payment_misallocation", "fx_rate_error", "cutoff_error")
TOOL_NAMES = ("list_records", "get_record", "get_document", "calculator", "submit")
GRADE_FIELDS = ("tp", "fp", "fn", "precision", "recall", "f1", "reward", "valid")
TASK = ("Audit the books for the period. Find errors of types [" + ", ".join(FINDING_TYPES)
        + "]. Submit findings with evidence.")
CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"

_log = logging.getLogger("auditgym.env")


class GraderClient(Protocol):
    """Interface of the out-of-process grader (SPEC §9.5-9.6)."""

    def submit(self, episode_id: str, findings: dict | None, inspection_steps: int) -> Any: ...


def _plain(value: object) -> object:
    if isinstance(value, MappingProxyType):
        return dict(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError("not JSON serialisable")


def canonical_json(value: object) -> str:
    """Canonical JSON used for observations and logs (SPEC §1)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False, default=_plain)


def _fresh(value: object) -> Any:
    return json.loads(canonical_json(value))


def load_config(config: dict | str, config_root: Path | None = None) -> dict[str, Any]:
    """Return a config dict: a dict is copied; a name loads ``default.yaml`` or ``difficulty/<name>.yaml``."""
    if isinstance(config, dict):
        return _fresh(config)
    if not isinstance(config, str) or not config.isidentifier():
        raise ValueError("unknown config preset")
    root = config_root or Path(os.environ.get("AUDITGYM_CONFIGS", str(CONFIG_ROOT)))
    path = root / ("default.yaml" if config == "default" else f"difficulty/{config}.yaml")
    if not path.is_file():
        raise ValueError("unknown config preset")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def expected_episode_id(seed: int, config: dict[str, Any]) -> str:
    """Episode id formula of SPEC §4.1, mirroring ``episode/serialize.py``."""
    config_bytes = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                              allow_nan=False).encode("utf-8")
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    salt = os.environ.get("AUDITGYM_SALT", "")
    digest = hashlib.sha256(f"{seed}{config_hash}{GENERATOR_VERSION}{salt}".encode()).hexdigest()
    return "ep_" + digest[:12]


def ensure_episode(data_root: Path, seed: int, config: dict[str, Any]) -> Path:
    """Return ``data_root/<episode_id>``, generating it with Role 1's generator if absent."""
    episode_dir = Path(data_root) / expected_episode_id(seed, config)
    if not (episode_dir / "open" / "manifest.json").is_file():
        from ..episode.serialize import generate_episode
        generate_episode(seed, config, episode_dir)
    return episode_dir


@dataclass(frozen=True)
class _State:
    data: EpisodeData | None = None
    steps_used: int = 0
    terminated: bool = False
    truncated: bool = False

    @property
    def active(self) -> bool:
        return self.data is not None and not (self.terminated or self.truncated)


class AuditEnv:
    """Read-only audit environment with a JSON action/observation contract (SPEC §8.1)."""

    def __init__(self, data_root: Path, mode: Literal["train", "eval"] = "train",
                 grader: GraderClient | None = None, max_steps: int = 100) -> None:
        if mode not in ("train", "eval"):
            raise ValueError("mode must be 'train' or 'eval'")
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive int")
        self._data_root = Path(data_root)
        self._mode = mode
        self._grader = grader
        self._max_steps = max_steps
        self._state = _State()

    @property
    def episode_id(self) -> str | None:
        """Id of the loaded episode, or ``None`` before reset / after close."""
        return None if self._state.data is None else self._state.data.episode_id

    def reset(self, seed: int, config: dict | str) -> tuple[dict, dict]:
        """Load (or generate) the episode for ``seed + config`` and start it from scratch."""
        if type(seed) is not int:
            raise ValueError("seed must be int")
        self._state = _State()
        cfg = load_config(config)
        episode_dir = ensure_episode(self._data_root, seed, cfg)
        return self._start(EpisodeData.load(episode_dir / "open"), episode_dir.name)

    def reset_from_dir(self, episode_dir: Path) -> tuple[dict, dict]:
        """Start an already generated episode directory (used by tests and transformed episodes)."""
        self._state = _State()
        episode_dir = Path(episode_dir)
        return self._start(EpisodeData.load(episode_dir / "open"), None)

    def step(self, action: dict) -> tuple[dict, float, bool, bool, dict]:
        """Apply one tool call. Errors consume a step; an inactive episode yields EPISODE_DONE."""
        state = self._state
        if not state.active:
            obs = self._obs(False, None, EnvError("EPISODE_DONE"))
            return obs, 0.0, state.terminated, state.truncated, {}
        assert state.data is not None
        steps_used = state.steps_used + 1
        submitted: dict | None = None
        ok, result, error = True, None, None
        try:
            tool, args = validate_action(_fresh(action))
            if tool == "submit":
                submitted = args["findings"]
                result = {"accepted": True}
            else:
                result = READ_TOOLS[tool](state.data, args)
        except EnvError as exc:
            ok, error = False, exc
        except (TypeError, ValueError):  # non-JSON action: NaN, sets, arbitrary objects
            ok, error = False, EnvError("BAD_ARGS")
        except Exception:  # noqa: BLE001 - internals are never shown to the agent
            _log.exception("internal error while handling a tool call")
            ok, error = False, EnvError("BAD_ARGS")
        terminated = submitted is not None
        truncated = not terminated and steps_used >= self._max_steps
        self._state = _State(data=state.data, steps_used=steps_used,
                             terminated=terminated, truncated=truncated)
        obs = self._obs(ok, result, error)
        if terminated:
            reward, info = self._grade(state.data.episode_id, submitted, steps_used - 1)
        elif truncated:
            reward, info = self._grade(state.data.episode_id, None, steps_used)
        else:
            reward, info = 0.0, {}
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Drop the loaded episode; further steps return EPISODE_DONE."""
        self._state = _State()

    def _start(self, data: EpisodeData, expected_id: str | None) -> tuple[dict, dict]:
        if expected_id is not None and data.episode_id != expected_id:
            raise RuntimeError("episode id mismatch between env and generator")
        self._state = _State(data=data)
        manifest = data.manifest
        result = {
            "task": TASK,
            "manifest": manifest,
            "policy": data.policy,
            "tables": data.counts(),
            "tools": list(TOOL_NAMES),
            "finding_types": list(FINDING_TYPES),
        }
        info = {"episode_id": data.episode_id, "schema_version": manifest["schema_version"],
                "config_version": manifest["config_version"]}
        return self._obs(True, result, None), _fresh(info)

    def _grade(self, episode_id: str, findings: dict | None, inspection_steps: int) -> tuple[float, dict]:
        if self._grader is None:
            return 0.0, {}
        grade = self._grader.submit(episode_id, findings, inspection_steps)
        if self._mode == "eval":
            return 0.0, {}
        summary = {name: getattr(grade, name) for name in GRADE_FIELDS}
        return float(grade.reward), _fresh({"grade": summary})

    def _obs(self, ok: bool, result: Mapping[str, Any] | None, error: EnvError | None) -> dict:
        steps_used = self._state.steps_used
        return _fresh({
            "ok": ok,
            "result": result if ok else None,
            "error": None if error is None else error.payload(),
            "steps_used": steps_used,
            "steps_left": self._max_steps - steps_used,
        })

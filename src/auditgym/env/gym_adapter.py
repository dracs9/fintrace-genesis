"""Optional Gymnasium wrapper (SPEC §8.6).

Actions and observations are JSON strings in ``Text(max_length=200_000)`` spaces. Observations
are canonical JSON with ``ensure_ascii=True`` so they always fit the printable-ASCII charset.
An action that is not valid JSON is passed to ``AuditEnv.step`` as-is and yields ``BAD_ARGS``.
Deterministic: ``reset(seed=s)`` always opens episode ``s``; with ``seed=None`` the episode seed
is drawn from ``self.np_random`` (seeded by Gymnasium).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import gymnasium
from gymnasium.spaces import Text

from .core import AuditEnv, GraderClient

MAX_LENGTH = 200_000
CHARSET = "".join(chr(c) for c in range(32, 127))


def _dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


class AuditGymEnv(gymnasium.Env):
    """``gymnasium.Env`` over :class:`AuditEnv` with JSON-string actions and observations."""

    metadata = {"render_modes": []}

    def __init__(self, data_root: Path, config: dict | str = "easy",
                 mode: Literal["train", "eval"] = "train", grader: GraderClient | None = None,
                 max_steps: int = 100) -> None:
        super().__init__()
        self._env = AuditEnv(Path(data_root), mode=mode, grader=grader, max_steps=max_steps)
        self._config = config
        self.observation_space = Text(max_length=MAX_LENGTH, min_length=1, charset=CHARSET)
        self.action_space = Text(max_length=MAX_LENGTH, min_length=1, charset=CHARSET)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[str, dict]:
        super().reset(seed=seed)
        episode_seed = seed if seed is not None else int(self.np_random.integers(0, 2**31 - 1))
        config = (options or {}).get("config", self._config)
        obs, info = self._env.reset(int(episode_seed), config)
        return _dumps(obs), info

    def step(self, action: str) -> tuple[str, float, bool, bool, dict]:
        try:
            parsed: Any = json.loads(action)
        except (TypeError, ValueError):
            parsed = None
        obs, reward, terminated, truncated, info = self._env.step(parsed)
        return _dumps(obs), float(reward), terminated, truncated, info

    def close(self) -> None:
        self._env.close()

"""Fixed agent-facing error codes and messages (SPEC §8.4).

Invariant: the agent only ever sees ``{"code": C, "message": MESSAGES[C]}``.
No tracebacks, paths, IDs or private data are ever attached to an error.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

MESSAGES: Mapping[str, str] = MappingProxyType({
    "UNKNOWN_TOOL": "unknown tool",
    "BAD_ARGS": "invalid arguments",
    "UNKNOWN_ID": "record not found",
    "BAD_CURSOR": "invalid cursor",
    "CALC_ERROR": "invalid expression",
    "EPISODE_DONE": "episode is not active",
})


class EnvError(Exception):
    """An agent-visible failure carrying only a code from SPEC §8.4."""

    def __init__(self, code: str) -> None:
        if code not in MESSAGES:
            raise ValueError("unknown error code")
        super().__init__(code)
        self.code = code

    def payload(self) -> dict[str, str]:
        """Return the fixed ``{"code", "message"}`` object for the observation."""
        return {"code": self.code, "message": MESSAGES[self.code]}

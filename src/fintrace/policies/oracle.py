"""Privileged benchmark ceiling using a grader-side channel and public tool reads."""

from collections import deque
from typing import Protocol

from ._oracle_channel import OracleChannel
from .base import Record, submit

__all__ = ['OraclePolicy', 'OracleChannel']


class _Channel(Protocol):
    def findings(self, episode_id: str) -> list[Record]: ...


class OraclePolicy:
    name = 'oracle'

    def __init__(self, channel: _Channel | None = None) -> None:
        self._channel = channel

    def reset(self, first_obs: Record, *, seed: int) -> None:
        """Load the first proof per issue over the explicit privileged capability."""
        if self._channel is None:
            raise ValueError('invalid arguments')
        self._episode_id: str = first_obs['result']['manifest']['episode_id']
        self._findings = self._channel.findings(self._episode_id)
        self._remaining = deque(sorted({key for item in self._findings for key in item['evidence_ids']}))
        self._pending: str | None = None
        self._inspected: set[str] = set()

    def act(self, obs: Record) -> Record:
        """Read each distinct proof record once, then submit the fully inspected findings."""
        if self._pending is not None and obs['ok']:
            self._inspected.add(self._pending)
        self._pending = None
        if not self._remaining or min(100 - obs['steps_used'], obs['steps_left']) <= 1:
            supported = [item for item in self._findings if self._inspected.issuperset(item['evidence_ids'])]
            return submit(self._episode_id, supported)
        self._pending = self._remaining.popleft()
        table, identifier = self._pending.split(':', 1)
        if table == 'doc':
            return {'tool': 'get_document', 'args': {'doc_id': identifier}}
        return {'tool': 'get_record', 'args': {'table': table, 'id': identifier}}

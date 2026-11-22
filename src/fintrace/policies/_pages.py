"""Shared read-only pagination for deterministic batch policies."""

from abc import ABC, abstractmethod

from .base import Record, page, submit


class PagePolicy(ABC):
    """Scan selected tables, reserving one action for the terminal submission."""

    tables: tuple[str, ...]
    first_pages_only = False

    def reset(self, first_obs: Record, *, seed: int) -> None:
        """Start a new scan in declared table order, clearing all previous records."""
        self.manifest: Record = first_obs['result']['manifest']
        self.types: list[str] = first_obs['result']['finding_types']
        self.rows: dict[str, list[Record]] = {table: [] for table in self.tables}
        self._next = 0
        self._pending: str | None = None
        self._cursor: str | None = None

    def act(self, obs: Record) -> Record:
        """Consume a single page and return the next read or the final findings."""
        if self._pending is not None:
            result = obs['result'] if obs['ok'] else None
            if result is not None:
                self.rows[self._pending].extend(result['items'])
            self._cursor = None if self.first_pages_only or result is None else result['next_cursor']
            if self._cursor is None:
                self._next += 1
            self._pending = None
        if self._next == len(self.tables) or min(100 - obs['steps_used'], obs['steps_left']) <= 1:
            return submit(self.manifest['episode_id'], self.findings())
        self._pending = self.tables[self._next]
        return page(self._pending, self._cursor)

    @abstractmethod
    def findings(self) -> list[Record]:
        """Return up to thirty findings based only on downloaded rows."""
        ...

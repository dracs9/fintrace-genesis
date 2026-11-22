"""Seeded baseline using only record handles returned by public tools."""

from fintrace.common.rng import Streams

from .base import Record, finding, page, submit


class RandomExplorer:
    name = 'random_explorer'

    def reset(self, first_obs: Record, *, seed: int) -> None:
        """Start an independent named policy stream and clear all visited identifiers."""
        self._manifest: Record = first_obs['result']['manifest']
        self._types: list[str] = first_obs['result']['finding_types']
        self._tables = sorted(first_obs['result']['tables'])
        self._rng = Streams(seed, self._manifest['episode_id'], '1.0', '').stream('policy.random_explorer')
        self._seen: dict[tuple[str, str], str] = {}
        self._cursors: dict[str, str | None] = {}
        self._pending: str | None = None

    def act(self, obs: Record) -> Record:
        """Inspect sixty times, then submit zero to three hypotheses on visited records."""
        if self._pending is not None and obs['ok']:
            table = self._pending
            result = obs['result']
            self._cursors[table] = result['next_cursor']
            for item in result['items']:
                if table == 'ledger':
                    identifier, prefix = item['row_id'], 'ledger'
                elif table == 'fx':
                    identifier = f"{item['date']}:{item['source_currency']}:{item['target_currency']}"
                    prefix = 'fx'
                else:
                    identifier, prefix = item['doc_id'], 'doc'
                self._seen[(table, identifier)] = f'{prefix}:{identifier}'
        self._pending = None
        keys = sorted(self._seen)
        if obs['steps_used'] >= 60 or obs['steps_left'] <= 1:
            count = min(int(self._rng.integers(4)), len(keys))
            indices = self._rng.choice(len(keys), size=count, replace=False)
            findings = []
            for i in indices:
                handle = self._seen[keys[int(i)]]
                kind = self._types[int(self._rng.integers(len(self._types)))]
                amount = int(self._rng.integers(0, 1_000_001))
                findings.append(finding(kind, [handle], amount, self._manifest['base_currency'],
                                        [handle], 'Sampled record for review.'))
            return submit(self._manifest['episode_id'], findings)
        if keys and self._rng.integers(2):
            table, identifier = keys[int(self._rng.integers(len(keys)))]
            return {'tool': 'get_record', 'args': {'table': table, 'id': identifier}}
        table = self._tables[int(self._rng.integers(len(self._tables)))]
        self._pending = table
        return page(table, self._cursors.get(table))

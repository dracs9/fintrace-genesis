"""Privileged oracle worker. The policy receives a narrow IPC capability, never a path."""

import json
import multiprocessing
import re
from multiprocessing.connection import Connection
from pathlib import Path
from types import TracebackType
from typing import Self

from fintrace.grader.client import encode
from fintrace.grader.privileges import drop_to
from fintrace.grader.schema import parse_truth

from .base import Record, finding


def _serve(connection: Connection, root: Path, run_as: str | None) -> None:
    with connection:
        if run_as is not None:
            drop_to(run_as)
        while True:
            try:
                request = connection.recv_bytes(128)
            except (OSError, EOFError):
                return
            try:
                episode_id = json.loads(request)
                if not isinstance(episode_id, str) or not re.fullmatch(r'ep_[a-f0-9]{12}', episode_id):
                    raise ValueError('invalid arguments')
                directory = root if (root / 'open' / 'manifest.json').is_file() else root / episode_id
                manifest = json.loads((directory / 'open' / 'manifest.json').read_bytes())
                if manifest['episode_id'] != episode_id:
                    raise ValueError('invalid arguments')
                truth = json.loads((directory / 'private' / 'truth.json').read_bytes())
                issues = parse_truth(truth, episode_id)
                reply = [finding(issue.type.value, sorted(issue.affected_ids), issue.expected_amount_minor,
                                 issue.currency, sorted(issue.evidence_sets[0]),
                                 'The cited records support the reported discrepancy.')
                         for issue in sorted(issues, key=lambda item: item.issue_id)]
            except (ValueError, OSError, KeyError, TypeError, RecursionError):
                reply = None
            try:
                connection.send_bytes(encode(reply))
            except (OSError, EOFError):
                return


class OracleChannel:
    """Trusted runner creates one dedicated grader-side worker for the built-in oracle."""

    def __init__(self, data_root: Path, *, run_as: str | None = None) -> None:
        """Bind to one episode or split root, optionally dropping the worker to grader UID."""
        context = multiprocessing.get_context('spawn')
        self._connection, worker_end = context.Pipe()
        self._worker = context.Process(target=_serve, args=(worker_end, data_root, run_as), daemon=True)
        try:
            self._worker.start()
        except BaseException:
            self._connection.close()
            raise
        finally:
            worker_end.close()

    def findings(self, episode_id: str) -> list[Record]:
        """Request canonical findings for a public episode ID, omitting internal metadata."""
        try:
            self._connection.send_bytes(encode(episode_id))
            reply = json.loads(self._connection.recv_bytes())
        except (ValueError, EOFError, OSError):
            raise ValueError('invalid arguments') from None
        if not isinstance(reply, list):
            raise ValueError('invalid arguments')
        return reply

    def close(self) -> None:
        """Release the dedicated endpoint and join its worker, including on early exit."""
        self._connection.close()
        self._worker.join(timeout=5)
        if self._worker.is_alive():
            self._worker.terminate()
            self._worker.join()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 traceback: TracebackType | None) -> None:
        self.close()

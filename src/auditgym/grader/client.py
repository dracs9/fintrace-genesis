"""Trusted environment adapter. Only the spawned worker opens the answer key."""

import json
import multiprocessing
from pathlib import Path
from threading import Lock
from types import TracebackType
from typing import Self

from .result import GradeResult


def _json_value(value: object) -> bool:
    if type(value) is list:
        return all(_json_value(child) for child in value)
    if type(value) is dict:
        return all(type(k) is str and _json_value(value[k]) for k in sorted(value))
    return value is None or type(value) in (bool, str, int, float)


def encode(value: object) -> bytes:
    """Encode a JSON message deterministically, without Python object serialization."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


class GraderClient:
    def __init__(self, data_root: Path, *, log_dir: Path = Path("logs"),
                 run_as: str | None = None) -> None:
        """Start one worker for a trusted episode directory or data_root/<episode_id>.

        With `run_as`, the worker switches to that POSIX account before reading anything.
        """
        from .service import serve

        context = multiprocessing.get_context("spawn")
        self._connection, worker_end = context.Pipe()
        self._lock = Lock()
        self._worker = context.Process(target=serve, args=(worker_end, data_root, log_dir, run_as),
                                       daemon=True)
        try:
            self._worker.start()
        except BaseException:
            self._connection.close()
            raise
        finally:
            worker_end.close()

    def submit(self, episode_id: str, findings: dict | None, inspection_steps: int) -> GradeResult:
        """Grade one terminal action. The worker retains the private match identifiers."""
        try:
            # json.dumps silently coerces tuples and numeric keys into JSON arrays/strings.
            packet = encode({"episode_id": episode_id, "inspection_steps": inspection_steps,
                             "findings": findings if _json_value(findings) else False})
        except (ValueError, TypeError, RecursionError):
            packet = encode({"episode_id": episode_id, "findings": False,
                             "inspection_steps": inspection_steps})
        with self._lock:
            try:
                self._connection.send_bytes(packet)
                reply = json.loads(self._connection.recv_bytes())
            except (EOFError, OSError):
                raise RuntimeError("episode is not active") from None
        if reply is None:
            raise RuntimeError("invalid arguments")
        return GradeResult(**reply, matches=())

    def close(self) -> None:
        """Release IPC and join the worker. Repeated closure has the same effect."""
        with self._lock:
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

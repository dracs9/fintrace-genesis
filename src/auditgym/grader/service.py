"""Trusted grader process. Deploy with grader-owned keys and log directory (§9.5)."""

import json
import os
import re
from dataclasses import asdict
from multiprocessing.connection import Connection
from pathlib import Path

from .client import encode
from .privileges import drop_to
from .result import GradeResult
from .reward import score
from .schema import parse_truth


def grade(episode_dir: Path, findings: dict | None, inspection_steps: int) -> GradeResult:
    """Trusted worker entry point. Bind the key to the public manifest before scoring."""
    manifest = json.loads((episode_dir / "open" / "manifest.json").read_bytes())
    episode_id = manifest["episode_id"]
    truth = json.loads((episode_dir / "private" / "truth.json").read_bytes())
    return score(episode_id, parse_truth(truth, episode_id), findings, inspection_steps)


def _episode(root: Path, episode_id: str) -> Path:
    if not isinstance(episode_id, str) or re.fullmatch(r"ep_[0-9a-f]{12}", episode_id) is None:
        raise ValueError("invalid arguments")
    if (root / "open" / "manifest.json").is_file():
        return root
    return root / episode_id


def _log(directory: Path, record: dict) -> None:
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(directory / "grader.jsonl", os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "ab", buffering=0) as output:
        output.write(encode(record) + b"\n")


def serve(connection: Connection, data_root: Path, log_dir: Path,
          run_as: str | None = None) -> None:
    """Read JSON requests until EOF. Emit aggregate results and fixed errors on IPC."""
    if run_as is not None:
        drop_to(run_as)
    with connection:
        while True:
            try:
                packet = connection.recv_bytes()
            except (EOFError, OSError):
                return
            try:
                request = json.loads(packet)
                if not isinstance(request, dict) or request.keys() != {"episode_id", "findings", "inspection_steps"}:
                    raise ValueError("invalid arguments")
                episode_id = request["episode_id"]
                directory = _episode(data_root, episode_id)
                result = grade(directory, request["findings"], request["inspection_steps"])
                if result.episode_id != episode_id:
                    raise ValueError("invalid arguments")
                reply = asdict(result)
                _log(log_dir, reply)
                del reply["matches"]
            except (KeyError, TypeError, ValueError, OSError, RecursionError):
                reply = None
                try:
                    _log(log_dir, {"error": "invalid arguments", "valid": False})
                except OSError:
                    pass
            try:
                connection.send_bytes(encode(reply))
            except (EOFError, OSError):
                return

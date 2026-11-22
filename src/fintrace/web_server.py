"""Operator console for FinTrace. Start from the checkout with python -m fintrace.web_server.

Loopback-only by default. A hosted deployment binds ``--host 0.0.0.0`` and names its public
domains with ``--public-host`` or ``FINTRACE_PUBLIC_HOSTS`` (comma-separated); Render's
``RENDER_EXTERNAL_HOSTNAME`` is added automatically. Requests for any other ``Host`` get 403."""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, contextmanager
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from threading import Lock
from typing import Any, Iterator, Literal
from urllib.parse import urlsplit

from .env.core import AuditEnv, canonical_json
from .env.errors import EnvError
from .grader.client import GraderClient
from .logging.trajectory import EpisodeSummary, read_jsonl, write_summary
from .policies import load_policy
from .policies.oracle import OracleChannel

Json = dict[str, Any]
Mode = Literal["train", "eval"]
WEB = Path(__file__).resolve().parents[2] / "web"
POLICIES = ("oracle", "reconciler", "flag_all", "random", "random_explorer")
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/studio.css": ("studio.css", "text/css; charset=utf-8"),
    "/studio.js": ("studio.js", "text/javascript; charset=utf-8"),
    "/explorer.js": ("explorer.js", "text/javascript; charset=utf-8"),
    "/findings.js": ("findings.js", "text/javascript; charset=utf-8"),
    "/ui.js": ("ui.js", "text/javascript; charset=utf-8"),
    "/inter.woff2": ("inter.woff2", "font/woff2"),
    "/jetbrains-mono.woff2": ("jetbrains-mono.woff2", "font/woff2"),
}


def _settings(body: Json) -> tuple[int, str, Mode]:
    seed = body.get("seed")
    preset = body.get("config", "medium")
    mode = body.get("mode", "train")
    if (type(seed) is not int or seed < 0 or preset not in ("easy", "medium", "hard")
            or mode not in ("train", "eval")):
        raise EnvError("BAD_ARGS")
    return seed, preset, mode


def _frame(env: AuditEnv, action: Json) -> Json:
    obs, reward, terminated, truncated, info = env.step(action)
    return dict(obs=obs, reward=reward, terminated=terminated, truncated=truncated, info=info)


class _Episode:
    """Own one environment, grader process and isolated single-submission operator log."""

    def __init__(self, stack: ExitStack, mode: Mode, *, interactive: bool = False) -> None:
        root = Path(stack.enter_context(TemporaryDirectory(prefix="fintrace-studio-")))
        self.data = root / "data"
        self.log = root / "operator" / "grader.jsonl"
        grader = stack.enter_context(GraderClient(self.data, log_dir=self.log.parent))
        env_class = _InteractiveEnv if interactive else AuditEnv
        self.env = env_class(self.data, mode=mode, grader=grader)
        stack.callback(self.env.close)

    def step(self, action: Json) -> Json:
        """Attach the worker's full result only to the transition that finishes this run."""
        frame = _frame(self.env, action)
        error = frame["obs"]["error"] or {}
        if (frame["terminated"] or frame["truncated"]) and error.get("code") != "EPISODE_DONE":
            grade, = read_jsonl(self.log)
            frame["grade"] = grade
        return frame


class _InteractiveEnv(AuditEnv):
    """Keep the step cap for manual review and exclude the operator's reading time."""

    _episode_seconds = float("inf")


def _run_policy(seed: int, config: str, mode: Mode, name: str) -> Json:
    # Executed in a spawned benchmark process. Oracle's channel never enters the HTTP server.
    with ExitStack() as stack:
        episode = _Episode(stack, mode)
        obs, info = episode.env.reset(seed, config)
        channel = stack.enter_context(OracleChannel(episode.data)) if name == "oracle" else None
        policy = load_policy(name, channel=channel)
        policy.reset(obs, seed=0)  # Same independent exploration seed as the CLI.
        trajectory = [dict(action=None, obs=obs, info=info, reward=0.0,
                           terminated=False, truncated=False)]
        while True:
            action = policy.act(obs)
            frame = episode.step(action)
            trajectory.append(dict(action=action, **frame))
            obs = frame["obs"]
            if frame["terminated"] or frame["truncated"]:
                grade = trajectory[-1].pop("grade")
                return dict(policy=policy.name, trajectory=trajectory, grade=grade)


def _runs(root: Path) -> list[Json]:
    """Read completed CLI and Studio summaries, enriching only from the public split manifest."""
    result = []
    runs = root / "runs"
    patterns = ("*/summary.jsonl", "*/*/summary.jsonl", "studio/*/ep_*/*/summary.jsonl")
    paths = sorted(path for pattern in patterns for path in runs.glob(pattern))
    for path in paths:
        if "private" in path.relative_to(root / "runs").parts:
            continue
        try:
            rows = read_jsonl(path)
        except (OSError, ValueError):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            eid = row.get("episode_id")
            if (not isinstance(eid, str) or not re.fullmatch(r"ep_[a-f0-9]{12}", eid)
                    or row.get("policy") not in POLICIES or type(row.get("steps")) is not int):
                continue
            metrics = ("precision", "recall", "f1", "reward")
            if any(type(row.get(key)) not in (int, float) for key in metrics):
                continue  # Unscored CLI eval summaries have null metrics.
            entry = {key: row[key] for key in ("episode_id", "policy", "steps", *metrics)}
            entry.update(split=None, config=None)
            for split in ("train", "dev", "fresh"):
                manifest = root / "data" / split / eid / "open" / "manifest.json"
                if manifest.is_file():
                    public = json.loads(manifest.read_bytes())
                    if public.get("episode_id") == eid:
                        entry.update(split=split, config=public.get("config_version"))
                        break
            if path.is_relative_to(root / "runs" / "studio"):
                entry.update(split="studio", config=path.parents[2].name)
            result.append(entry)
    return result


class Studio:
    """One local workbench with serialized transitions and a revision for stale browser tabs."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.lock = Lock()
        self.revision = 0
        self.episode: _Episode | None = None
        self.resources = ExitStack()

    def close(self) -> None:
        """Release the active environment, grader and temporary workspace."""
        self.resources.close()
        self.episode = None

    def dispatch(self, path: str, body: Json) -> Json:
        """Route one validated local request. Caller holds the workspace lock."""
        if path == "/api/runs":
            return {"runs": _runs(self.root)}
        if path == "/api/env/step":
            if self.episode is None:
                raise EnvError("EPISODE_DONE")
            result = self.episode.step(body)
        elif path == "/api/env/reset":
            seed, config, mode = _settings(body)
            with ExitStack() as pending:
                episode = _Episode(pending, mode, interactive=True)
                obs, info = episode.env.reset(seed, config)
                self.close()
                self.episode, self.resources = episode, pending.pop_all()
            result = dict(obs=obs, info=info)
        elif path == "/api/env/run-policy":
            seed, config, mode = _settings(body)
            name = body.get("policy")
            if name not in POLICIES:
                raise EnvError("BAD_ARGS")
            started = time.perf_counter()
            with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as worker:
                result = worker.submit(_run_policy, seed, config, mode, name).result()
            grade = result["grade"]
            terminal = result["trajectory"][-1]
            summary = EpisodeSummary(grade["episode_id"], result["policy"], grade["precision"],
                                     grade["recall"], grade["f1"], grade["reward"],
                                     terminal["obs"]["steps_used"], time.perf_counter() - started)
            directory = self.root / "runs/studio" / config / summary.episode_id / summary.policy
            directory.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=directory, delete=False) as handle:
                temp = Path(handle.name)
            try:
                write_summary(temp, [summary])
                temp.replace(directory / "summary.jsonl")
            finally:
                temp.unlink(missing_ok=True)
            self.close()  # The completed replay replaces the manual workspace.
        else:
            raise EnvError("UNKNOWN_TOOL")
        self.revision += 1
        return result


class _HTTP(BaseHTTPRequestHandler):
    def __init__(self, *args: Any, studio: Studio, public_hosts: frozenset[str] = frozenset(),
                 **kwargs: Any) -> None:
        self.studio = studio
        self.public_hosts = public_hosts
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: object) -> None:
        """Keep request paths and operator input out of logs."""

    def _send(self, status: int, content: bytes, kind: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; object-src 'none'; "
                         "frame-ancestors 'none'; base-uri 'none'; style-src 'self' 'unsafe-inline'")
        self.send_header("ETag", f'"{self.studio.revision}"')
        self.end_headers()
        self.wfile.write(content)

    def _json(self, status: int, value: Json) -> None:
        self._send(status, canonical_json(value).encode(), "application/json; charset=utf-8")

    def _local(self) -> bool:
        """Accept loopback or a configured public domain, and only same-origin browser calls."""
        port = self.server.server_port
        host = self.headers.get("Host")
        origin = self.headers.get("Origin")
        if host in (f"127.0.0.1:{port}", f"localhost:{port}"):
            return origin in (None, f"http://{host}")
        return host in self.public_hosts and origin in (None, f"https://{host}")

    def do_GET(self) -> None:
        """Serve allowlisted assets and completed benchmark measurements on loopback."""
        path = urlsplit(self.path).path
        if path == "/healthz":  # Platform probes use internal Host headers; reveals nothing.
            self._send(200, b"ok", "text/plain; charset=utf-8")
        elif not self._local():
            self._json(403, {"error": EnvError("BAD_ARGS").payload()})
        elif path == "/api/runs":
            self._api(path, {})
        elif path in ASSETS:
            name, kind = ASSETS[path]
            self._send(200, (WEB / name).read_bytes(), kind)
        else:
            self._json(404, {"error": EnvError("UNKNOWN_TOOL").payload()})

    def do_POST(self) -> None:
        """Decode a bounded same-origin JSON request and forward its action unchanged."""
        if not self._local():
            self._json(403, {"error": EnvError("BAD_ARGS").payload()})
            return
        path = urlsplit(self.path).path
        if path not in ("/api/env/reset", "/api/env/step", "/api/env/run-policy"):
            self._json(404, {"error": EnvError("UNKNOWN_TOOL").payload()})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if self.headers.get_content_type() != "application/json" or not 0 < size <= 1_000_000:
                raise ValueError
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError
        except (ValueError, RecursionError):
            self._json(400, {"error": EnvError("BAD_ARGS").payload()})
            return
        self._api(path, body)

    def _api(self, path: str, body: Json) -> None:
        with self.studio.lock:
            if self.headers.get("If-Match") not in (None, f'"{self.studio.revision}"'):
                self._json(409, {"error": EnvError("EPISODE_DONE").payload()})
                return
            try:
                value = self.studio.dispatch(path, body)
            except EnvError as error:
                self._json(400, {"error": error.payload()})
            except Exception:
                self._json(500, {"error": EnvError("BAD_ARGS").payload()})
            else:
                self._json(200, value)


@contextmanager
def server(root: Path, port: int = 8765, *, host: str = "127.0.0.1",
           public_hosts: tuple[str, ...] = ()) -> Iterator[ThreadingHTTPServer]:
    """Bind a server for a checkout and release every grader on shutdown.

    ``public_hosts`` are exact ``Host`` values (no port) served behind an HTTPS proxy."""
    studio = Studio(root)
    handler = partial(_HTTP, studio=studio, public_hosts=frozenset(h.lower() for h in public_hosts))
    try:
        with ThreadingHTTPServer((host, port), handler) as http:
            yield http
    finally:
        studio.close()


def main() -> None:
    """Serve the local Studio until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8765")))
    parser.add_argument("--public-host", action="append", default=[])
    args = parser.parse_args()
    env_hosts = os.environ.get("FINTRACE_PUBLIC_HOSTS", "").split(",")
    env_hosts.append(os.environ.get("RENDER_EXTERNAL_HOSTNAME", ""))
    public = tuple(sorted({h.strip() for h in [*args.public_host, *env_hosts] if h.strip()}))
    with server(Path.cwd(), args.port, host=args.host, public_hosts=public) as http:
        print(f"FinTrace Studio  http://{args.host}:{http.server_port}", flush=True)
        try:
            http.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

"""``fintrace`` command line (SPEC §15).

Commands owned by other roles are resolved lazily through ``ENTRY_POINTS``; when a module or
function does not exist yet the command prints ``"<cmd>: not implemented yet"`` and exits 2
without affecting the other commands. Deterministic except for wall-clock timings in summaries.
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

from .env.core import AuditEnv, ensure_episode, load_config
from .logging.trajectory import EpisodeSummary, TrajectoryWriter, write_summary

SPLITS = {"train": 10_000, "dev": 20_000}
NOT_IMPLEMENTED = 2
# Docker privilege separation (SPEC §9.5): unset locally, set in the Dockerfile.
GRADER_USER_ENV = "FINTRACE_GRADER_USER"
AGENT_USER_ENV = "FINTRACE_AGENT_USER"

# Other roles' functions. Adjust here when their modules land.
ENTRY_POINTS = {
    "grader": "fintrace.grader.client:GraderClient",                    # Role 3, SPEC §9.5
    "validate": "fintrace.validate.independent_checker:validate_data",  # Role 4, SPEC §12
    "transform": "fintrace.episode.transform:transform",                # Role 4, SPEC §13
    "report": "fintrace.validate.report:build_report",                  # Role 4, SPEC §16
}
POLICIES = {  # Role 4, SPEC §10: module per policy, class found by its ``name`` attribute
    "oracle": "fintrace.policies.oracle",
    "random": "fintrace.policies.random_explorer",
    "flag_all": "fintrace.policies.flag_all",
    "reconciler": "fintrace.policies.reconciler",
}
POLICY_NAMES = {"oracle": "oracle", "random": "random_explorer", "flag_all": "flag_all",
                "reconciler": "reconciler"}


class NotImplementedYet(Exception):
    """Raised when another role's entry point is not available yet."""


def _resolve(spec: str) -> Any:
    """Import ``pkg.module:attr``; missing module/attribute → ``NotImplementedYet``."""
    module_name, _, attr = spec.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name and module_name.startswith(exc.name):
            raise NotImplementedYet(spec) from None
        raise
    try:
        return getattr(module, attr)
    except AttributeError:
        raise NotImplementedYet(spec) from None


def load_policy(spec: str) -> Any:
    """Instantiate a built-in policy by name or an external one from ``pkg.module:Class``."""
    if ":" in spec:
        if str(Path.cwd()) not in sys.path:
            sys.path.insert(0, str(Path.cwd()))
        try:
            cls = _resolve(spec)
        except NotImplementedYet:
            raise ValueError(f"cannot import policy {spec}") from None
        if not isinstance(cls, type):
            raise ValueError("invalid arguments")
        policy = cls()
        if not isinstance(getattr(policy, "name", None), str) or not all(
            callable(getattr(policy, method, None)) for method in ("act", "reset")
        ):
            raise ValueError("invalid arguments")
        return policy
    if spec == "random_explorer":
        spec = "random"
    if spec not in POLICIES:
        raise ValueError(f"unknown policy {spec}")
    try:
        module = importlib.import_module(POLICIES[spec])
    except ModuleNotFoundError:
        raise NotImplementedYet(POLICIES[spec]) from None
    for attr in sorted(vars(module)):
        obj = getattr(module, attr)
        if isinstance(obj, type) and getattr(obj, "name", None) == POLICY_NAMES[spec] \
                and callable(getattr(obj, "act", None)):
            return obj()
    raise NotImplementedYet(POLICIES[spec])


@dataclass
class _RecordingGrader:
    """Forwards to the real grader and keeps the last result for the operator's summary."""

    inner: Any
    last: Any = None

    def submit(self, episode_id: str, findings: dict | None, inspection_steps: int) -> Any:
        self.last = self.inner.submit(episode_id, findings, inspection_steps)
        return self.last


def make_grader(data_root: Path) -> Any | None:
    """Start Role 3's grader if available, else ``None`` (rewards become 0.0)."""
    try:
        cls = _resolve(ENTRY_POINTS["grader"])
    except NotImplementedYet:
        print("grader: not implemented yet (rewards are 0.0)", file=sys.stderr)
        return None
    return cls(data_root, run_as=os.environ.get(GRADER_USER_ENV) or None)


def drop_agent_privileges() -> None:
    """Switch this process to ``$FINTRACE_AGENT_USER`` (irreversible); no-op when unset."""
    user = os.environ.get(AGENT_USER_ENV)
    if user:
        from .grader.privileges import drop_to
        drop_to(user)


def run_policy_episode(env: AuditEnv, policy: Any, seed: int, config: dict | str,
                       writer: TrajectoryWriter, grader: _RecordingGrader | None) -> EpisodeSummary:
    """Run one episode to termination/truncation, logging every step (SPEC §11)."""
    started = time.perf_counter()
    obs, info = env.reset(seed, config)
    return _run_observation(env, policy, writer, grader, obs, info, started)


def _run_observation(env: AuditEnv, policy: Any, writer: TrajectoryWriter,
                     grader: _RecordingGrader | None, obs: dict, info: dict,
                     started: float) -> EpisodeSummary:
    if grader is not None:
        grader.last = None
    writer.begin(info, obs)
    policy.reset(obs, seed=0)  # Public exploration seed, independent of episode generation.
    transition = 0
    while True:
        previous_steps = obs["steps_used"]
        action = policy.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        transition += 1
        recorded_action = action if obs["steps_used"] > previous_steps else None
        writer.step(transition, recorded_action, obs, reward, terminated, truncated)
        if terminated or truncated:
            break
    grade = None if grader is None else grader.last
    return EpisodeSummary(
        episode_id=info["episode_id"], policy=str(getattr(policy, "name", type(policy).__name__)),
        precision=None if grade is None else float(grade.precision),
        recall=None if grade is None else float(grade.recall),
        f1=None if grade is None else float(grade.f1),
        reward=None if grade is None else float(grade.reward),
        steps=obs["steps_used"], wall_time_s=round(time.perf_counter() - started, 3))


def _seeds(split: str, n: int) -> list[int]:
    return [SPLITS[split] + i for i in range(n)]


# --- commands -------------------------------------------------------------------------------

def cmd_generate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    root = Path(args.out) / args.split
    for seed in _seeds(args.split, args.n):
        print(ensure_episode(root, seed, cfg).as_posix())
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    data_root = Path(args.data) / args.split
    config = load_config(args.config)
    paths = [ensure_episode(data_root, seed, config) for seed in _seeds(args.split, args.n)]
    run_dir = run_cached_episodes(
        data_root, paths, args.policy, Path(args.runs), split=args.split, mode=args.mode,
        max_steps=args.max_steps, run_id=args.run_id,
    )
    print(run_dir.as_posix())
    return 0


def run_cached_episodes(data_root: Path, paths: Sequence[Path], policy_spec: str,
                        runs: Path, *, split: str, mode: Literal["train", "eval"] = "train",
                        max_steps: int = 100, run_id: str | None = None) -> Path:
    """Run a policy on prepared episodes, recording observations and independent grader results.

    In a worker configured for UID separation this call drops privileges irreversibly.
    Only the built-in oracle receives the separate trusted capability.
    """
    if not 1 <= max_steps <= 100:
        raise ValueError("max_steps must be between 1 and 100")
    with ExitStack() as stack:
        real = make_grader(data_root)
        if real is not None:
            stack.callback(real.close)
        recording = _RecordingGrader(real) if real is not None and mode == "train" else None
        channel = None
        if policy_spec == "oracle":
            from .policies.oracle import OracleChannel, OraclePolicy
            channel = stack.enter_context(OracleChannel(data_root, run_as=os.getenv(GRADER_USER_ENV) or None))
        drop_agent_privileges()
        salt = os.environ.pop("FINTRACE_SALT", None)
        if salt is not None:
            stack.callback(os.environ.__setitem__, "FINTRACE_SALT", salt)
        policy = OraclePolicy(channel) if channel is not None else load_policy(policy_spec)
        name = str(getattr(policy, "name", policy_spec))
        slug = name.replace(":", "_").replace("/", "_")
        run_dir = runs / (run_id or f"{slug}-{split}-{mode}")
        env = AuditEnv(data_root, mode=mode, grader=recording or real, max_steps=max_steps)
        stack.callback(env.close)
        writer = stack.enter_context(TrajectoryWriter(run_dir / "trajectories.jsonl", name))
        rows = []
        for path in paths:
            started = time.perf_counter()
            obs, info = env.reset_from_dir(path)
            if path.name != info["episode_id"]:
                raise ValueError("invalid arguments")
            rows.append(_run_observation(env, policy, writer, recording, obs, info, started))
        write_summary(run_dir / "summary.jsonl", rows)
    return run_dir


def _call_entry(name: str, *call_args: Any, **kwargs: Any) -> int:
    fn: Callable[..., Any] = _resolve(ENTRY_POINTS[name])
    result = fn(*call_args, **kwargs)
    return result if isinstance(result, int) else 0


def cmd_validate(args: argparse.Namespace) -> int:
    return _call_entry("validate", Path(args.data), Path(args.out))


def cmd_transform(args: argparse.Namespace) -> int:
    _call_entry("transform", Path(args.src), Path(args.dst), permute_seed=args.permute_seed,
                id_salt=args.id_salt)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    return _call_entry("report", Path(args.runs), Path(args.out), Path(args.data))


def cmd_all(args: argparse.Namespace) -> int:
    """generate -> validate -> pytest -> 4 policies -> report; unavailable stages are skipped."""
    skipped: list[str] = []

    def stage(label: str, fn: Callable[[], int]) -> bool:
        try:
            code = fn()
        except NotImplementedYet:
            print(f"{label}: not implemented yet (skipped)")
            skipped.append(label)
            return True
        if code != 0:
            print(f"{label}: failed with exit code {code}")
            return False
        return True

    base = dict(config=args.config, n=args.n, out=args.data, data=args.data, runs=args.runs)
    for split in ("train", "dev"):
        if not stage(f"generate {split}", lambda s=split: cmd_generate(argparse.Namespace(**base, split=s))):
            return 1
    if not stage("validate", lambda: cmd_validate(argparse.Namespace(data=args.data,
                                                                      out="reports/validation.json"))):
        return 1
    if not args.skip_tests:
        if not stage("pytest", lambda: subprocess.call([sys.executable, "-m", "pytest", "-q"])):
            return 1
    for policy in sorted(POLICIES):
        if not stage(f"run {policy}", lambda p=policy: _run_subprocess(p, args)):
            return 1
    if not stage("report", lambda: cmd_report(argparse.Namespace(runs=args.runs,
                                                                  out="reports/baselines.md",
                                                                  data=args.data))):
        return 1
    print("all: done" + (f" (skipped: {', '.join(skipped)})" if skipped else ""))
    return 0


def _run_subprocess(policy: str, args: argparse.Namespace) -> int:
    """Run one policy on both splits with separate child processes.

    Each run drops privileges irreversibly, so ``all`` keeps its own UID and starts a fresh
    process per policy. A policy that is not implemented yet is detected here and skipped.
    """
    load_policy(policy)
    for split in sorted(SPLITS):
        code = subprocess.call([sys.executable, "-m", "fintrace.cli", "run", "--policy", policy,
                                "--split", split, "--mode", "train", "--config", str(args.config),
                                "--n", str(args.n), "--data", str(args.data), "--runs", str(args.runs)])
        if code:
            return code
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fintrace")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="generate episodes for a split")
    p.add_argument("--split", choices=sorted(SPLITS), required=True)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--config", default="medium")
    p.add_argument("--out", default="data")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("run", help="run a policy on a split")
    p.add_argument("--policy", required=True, help="oracle|random|flag_all|reconciler|pkg.mod:Class")
    p.add_argument("--split", choices=sorted(SPLITS), required=True)
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--config", default="medium")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--data", default="data")
    p.add_argument("--runs", default="runs")
    p.add_argument("--run-id", default=None)
    p.add_argument("--max-steps", type=int, default=100)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("validate", help="independent validation of generated data")
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="reports/validation.json")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("transform", help="permute rows and rename ids of an episode")
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--permute-seed", type=int, required=True)
    p.add_argument("--id-salt", required=True)
    p.set_defaults(func=cmd_transform)

    p = sub.add_parser("report", help="baseline report from runs")
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="reports/baselines.md")
    p.add_argument("--data", default="data")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("all", help="generate -> validate -> pytest -> policies -> report")
    p.add_argument("--config", default="medium")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--data", default="data")
    p.add_argument("--runs", default="runs")
    p.add_argument("--skip-tests", action="store_true", help="do not run pytest inside 'all'")
    p.set_defaults(func=cmd_all)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except NotImplementedYet:
        print(f"{args.command}: not implemented yet", file=sys.stderr)
        return NOT_IMPLEMENTED
    except ValueError as exc:
        print(f"{args.command}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

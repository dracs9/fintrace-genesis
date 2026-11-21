"""``auditgym`` command line (SPEC §15).

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .env.core import AuditEnv, ensure_episode, load_config
from .logging.trajectory import EpisodeSummary, TrajectoryWriter, write_summary

SPLITS = {"train": 10_000, "dev": 20_000}
NOT_IMPLEMENTED = 2
# Docker privilege separation (SPEC §9.5): unset locally, set in the Dockerfile.
GRADER_USER_ENV = "AUDITGYM_GRADER_USER"
AGENT_USER_ENV = "AUDITGYM_AGENT_USER"

# Other roles' functions. Adjust here when their modules land.
ENTRY_POINTS = {
    "grader": "auditgym.grader.client:GraderClient",                    # Role 3, SPEC §9.5
    "validate": "auditgym.validate.independent_checker:validate_data",  # Role 4, SPEC §12
    "transform": "auditgym.episode.transform:transform",                # Role 4, SPEC §13
    "report": "auditgym.validate.report:build_report",                  # Role 4, SPEC §16
}
POLICIES = {  # Role 4, SPEC §10: module per policy, class found by its ``name`` attribute
    "oracle": "auditgym.policies.oracle",
    "random": "auditgym.policies.random_explorer",
    "flag_all": "auditgym.policies.flag_all",
    "reconciler": "auditgym.policies.reconciler",
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
        return cls()
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
    """Switch this process to ``$AUDITGYM_AGENT_USER`` (irreversible); no-op when unset."""
    user = os.environ.get(AGENT_USER_ENV)
    if user:
        from .grader.privileges import drop_to
        drop_to(user)


def run_policy_episode(env: AuditEnv, policy: Any, seed: int, config: dict | str,
                       writer: TrajectoryWriter, grader: _RecordingGrader | None) -> EpisodeSummary:
    """Run one episode to termination/truncation, logging every step (SPEC §11)."""
    started = time.perf_counter()
    if grader is not None:
        grader.last = None
    obs, info = env.reset(seed, config)
    writer.begin(info, obs)
    policy.reset(obs, seed=seed)
    step = 0
    while True:
        action = policy.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        step += 1
        writer.step(step, action, obs, reward, terminated, truncated)
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
    policy = load_policy(args.policy)
    name = str(getattr(policy, "name", args.policy)).replace(":", "_").replace("/", "_")
    run_dir = Path(args.runs) / (args.run_id or f"{name}-{args.split}-{args.mode}")
    real = make_grader(data_root)
    grader = None if real is None else _RecordingGrader(real)
    try:
        drop_agent_privileges()  # after the grader worker is started, before any policy step
    except BaseException:
        if real is not None:
            real.close()
        raise
    env = AuditEnv(data_root, mode=args.mode, grader=grader, max_steps=args.max_steps)
    rows = []
    try:
        with TrajectoryWriter(run_dir / "trajectories.jsonl", name) as writer:
            for seed in _seeds(args.split, args.n):
                rows.append(run_policy_episode(env, policy, seed, args.config, writer, grader))
    finally:
        env.close()
        if real is not None and callable(getattr(real, "close", None)):
            real.close()
    write_summary(run_dir / "summary.jsonl", rows)
    print(run_dir.as_posix())
    return 0


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
    return _call_entry("report", Path(args.runs), Path(args.out))


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
    for policy in POLICIES:
        if not stage(f"run {policy}", lambda p=policy: _run_subprocess(p, args)):
            return 1
    if not stage("report", lambda: cmd_report(argparse.Namespace(runs=args.runs,
                                                                  out="reports/baselines.md"))):
        return 1
    print("all: done" + (f" (skipped: {', '.join(skipped)})" if skipped else ""))
    return 0


def _run_subprocess(policy: str, args: argparse.Namespace) -> int:
    """Run one policy on dev in a child ``auditgym run``.

    Each run drops privileges irreversibly, so ``all`` keeps its own UID and starts a fresh
    process per policy. A policy that is not implemented yet is detected here and skipped.
    """
    load_policy(policy)
    return subprocess.call([sys.executable, "-m", "auditgym.cli", "run", "--policy", policy,
                            "--split", "dev", "--mode", "train", "--config", str(args.config),
                            "--n", str(args.n), "--data", str(args.data), "--runs", str(args.runs)])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="auditgym")
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

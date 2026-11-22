"""SPEC §13: compare real pre- and post-transform episodes on 100 fresh seeds."""

from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from auditgym.env.core import AuditEnv, expected_episode_id, load_config
from auditgym.episode.serialize import generate_episode
from auditgym.episode.transform import transform
from auditgym.grader.client import GraderClient
from auditgym.policies import Policy, run_episode
from auditgym.policies.oracle import OracleChannel, OraclePolicy
from auditgym.policies.reconciler import Reconciler


@dataclass(frozen=True)
class EpisodePair:
    seed: int
    episode_id: str
    original: Path
    transformed: Path
    mapping: dict[str, str]


@dataclass(frozen=True)
class Operators:
    env: AuditEnv
    grader: GraderClient
    oracle: OraclePolicy


@dataclass(frozen=True)
class Outcome:
    findings: dict[str, Any]
    steps: int
    reward: float
    metrics: tuple[float, float, float]


@pytest.fixture(scope="module")
def roots(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    workspace = tmp_path_factory.mktemp("metamorphic")
    return workspace / "original", workspace / "transformed"


@pytest.fixture(scope="module", params=range(90000, 90100))
def pair(request: pytest.FixtureRequest, roots: tuple[Path, Path]) -> EpisodePair:
    seed = int(request.param)
    # All three presets include legitimate distractors; hard also exercises interactions.
    config = load_config(("easy", "medium", "hard")[seed % 3])
    episode_id = expected_episode_id(seed, config)
    original, transformed = (root / episode_id for root in roots)
    generate_episode(seed, config, original)
    mapping = transform(original, transformed, permute_seed=seed, id_salt=f"remap-{seed}")
    return EpisodePair(seed, episode_id, original, transformed, mapping)


@pytest.fixture(scope="module")
def operators(roots: tuple[Path, Path]) -> Iterator[tuple[Operators, Operators]]:
    # Keep the trusted workers alive across seeds; each representation has its own
    # data root because a transformation preserves the episode_id.
    with ExitStack() as stack:
        def attach(root: Path) -> Operators:
            root.mkdir(parents=True, exist_ok=True)
            grader = stack.enter_context(GraderClient(root, log_dir=root.parent / f"{root.name}-logs"))
            channel = stack.enter_context(OracleChannel(root))
            env = AuditEnv(root, grader=grader)
            stack.callback(env.close)
            return Operators(env, grader, OraclePolicy(channel))

        yield attach(roots[0]), attach(roots[1])


def _play(env: AuditEnv, policy: Policy, episode: Path, seed: int) -> Outcome:
    first, _ = env.reset_from_dir(episode)
    result = run_episode(env, policy, first, seed=seed)
    grade = result.info["grade"]
    assert grade["valid"] and result.findings is not None
    assert 1 <= result.steps <= 100
    return Outcome(result.findings, result.steps, result.reward,
                   (grade["precision"], grade["recall"], grade["f1"]))


@pytest.fixture(scope="module")
def oracles(pair: EpisodePair, operators: tuple[Operators, Operators]) -> tuple[Outcome, Outcome]:
    a, b = operators
    return _play(a.env, a.oracle, pair.original, pair.seed), _play(b.env, b.oracle, pair.transformed, pair.seed)


def test_m1_validator_accepts_transformed_episode(pair: EpisodePair) -> None:
    from auditgym.validate.independent_checker import validate_episode

    validate_episode(pair.transformed)


def test_m2_oracle_reward(pair: EpisodePair, oracles: tuple[Outcome, Outcome]) -> None:
    original, transformed = oracles
    assert original.findings["episode_id"] == transformed.findings["episode_id"] == pair.episode_id
    assert original.metrics[2] == transformed.metrics[2] == 1.0
    assert original.reward == transformed.reward


def test_m3_reconciler_metrics(pair: EpisodePair, operators: tuple[Operators, Operators]) -> None:
    original = _play(operators[0].env, Reconciler(), pair.original, pair.seed)
    after = _play(operators[1].env, Reconciler(), pair.transformed, pair.seed)
    assert original.metrics == after.metrics


def test_m4_transported_oracle_answer(pair: EpisodePair, oracles: tuple[Outcome, Outcome],
                                      operators: tuple[Operators, Operators]) -> None:
    original = oracles[0]

    def transport(ref: str) -> str:
        namespace, raw = ref.split(":", 1)
        return ref if namespace == "fx" else f"{namespace}:{pair.mapping[raw]}"

    findings = []
    for finding in original.findings["findings"]:
        findings.append({**finding, "affected_ids": [transport(ref) for ref in finding["affected_ids"]],
                         "evidence_ids": [transport(ref) for ref in finding["evidence_ids"]]})
    answer = {"episode_id": pair.episode_id, "findings": findings}
    grade = operators[1].grader.submit(pair.episode_id, answer, original.steps - 1)
    assert grade.valid and grade.f1 == 1.0
    assert grade.reward == original.reward

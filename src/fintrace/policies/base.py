"""Public observation/action interfaces for FinTrace baseline policies."""

from dataclasses import dataclass
from typing import Any, Protocol

Record = dict[str, Any]


class Policy(Protocol):
    """A decision rule sees only reset/step observations and an independent policy seed."""

    name: str

    def reset(self, first_obs: Record, *, seed: int) -> None:
        """Discard earlier episode state and initialize deterministic decisions."""
        ...

    def act(self, obs: Record) -> Record:
        """Return one §8.3 tool call using the latest public observation."""
        ...


class Stepper(Protocol):
    def step(self, action: Record) -> tuple[Record, float, bool, bool, Record]: ...


@dataclass(frozen=True)
class EpisodeRun:
    """Final public outcome, including the submitted §9.1 envelope on termination."""

    findings: Record | None
    steps: int
    reward: float
    info: Record
    terminated: bool
    truncated: bool


def run_episode(env: Stepper, policy: Policy, first_obs: Record, *, seed: int = 0) -> EpisodeRun:
    """Run an already-reset episode to completion, seeding only the decision rule."""
    if not 1 <= first_obs['steps_left'] <= 100 or first_obs['steps_used'] != 0:
        raise ValueError('invalid arguments')
    policy.reset(first_obs, seed=seed)
    obs = first_obs
    while True:
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            findings = action['args']['findings'] if terminated and action.get('tool') == 'submit' else None
            return EpisodeRun(findings, obs['steps_used'], reward, info, terminated, truncated)


def finding(kind: str, affected: list[str], amount: int, currency: str,
            evidence: list[str], explanation: str) -> Record:
    """Construct a §9.1 finding with exact minor units and canonical public handles."""
    return {'type': kind, 'affected_ids': sorted(set(affected)), 'claimed_amount_minor': amount,
            'currency': currency, 'evidence_ids': sorted(set(evidence)), 'explanation': explanation}


def submit(episode_id: str, findings: list[Record]) -> Record:
    """Build the terminal action for one public episode identifier."""
    return {'tool': 'submit', 'args': {'findings': {'episode_id': episode_id, 'findings': findings}}}


def page(table: str, cursor: str | None = None) -> Record:
    """Request up to fifty rows, passing the opaque cursor back unchanged."""
    return {'tool': 'list_records', 'args': {'table': table, 'cursor': cursor, 'limit': 50}}

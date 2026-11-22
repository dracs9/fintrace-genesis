"""Public policy protocol, driver and CLI-compatible loader."""

from typing import cast

from .base import EpisodeRun, Policy, run_episode
from .oracle import OracleChannel, OraclePolicy

__all__ = ['EpisodeRun', 'Policy', 'load_policy', 'run_episode']


def load_policy(spec: str, *, channel: OracleChannel | None = None) -> Policy:
    """Load a built-in or pkg.module:Class, passing the capability only to the built-in oracle."""
    if spec == 'oracle':
        return OraclePolicy(channel)
    from fintrace.cli import load_policy as load_cli_policy

    obj = load_cli_policy('random' if spec == 'random_explorer' else spec)
    if not isinstance(getattr(obj, 'name', None), str) or not all(
        callable(getattr(obj, attr, None)) for attr in ('act', 'reset')
    ):
        raise ValueError('invalid arguments')
    return cast(Policy, obj)

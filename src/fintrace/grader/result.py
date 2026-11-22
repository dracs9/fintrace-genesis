"""GradeResult from SPEC §9.6, kept here until shared types are integrated."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GradeResult:
    episode_id: str
    valid: bool
    error: str | None
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    inspection_steps: int
    reward: float
    matches: tuple[tuple[str, int], ...]

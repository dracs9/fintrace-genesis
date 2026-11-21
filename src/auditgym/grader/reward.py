"""Deterministic terminal metrics from the frozen scoring contract."""

from .matcher import match
from .normalize import Issue
from .result import GradeResult
from .schema import parse_findings


def score(episode_id: str, issues: tuple[Issue, ...], findings: object,
          inspection_steps: int) -> GradeResult:
    """Score one response. None denotes the environment's timeout sentinel."""
    if type(inspection_steps) is not int or not 0 <= inspection_steps <= 100:
        raise ValueError("invalid arguments")
    try:
        rows = () if findings is None else parse_findings(findings, episode_id)
    except ValueError:
        return GradeResult(episode_id, False, "invalid arguments", 0, 0, len(issues),
                           0.0, 0.0, 0.0, inspection_steps, -1.0, ())
    matches = match(issues, rows)
    tp = len(matches)
    fp, fn = len(rows) - tp, len(issues) - tp
    precision = tp / len(rows) if rows else 0.0
    recall = tp / len(issues) if issues else 0.0
    f1 = 2 * tp / (len(rows) + len(issues)) if rows or issues else 1.0
    reward = max(-1.0, min(1.0, 2 * f1 - 1 - 0.002 * inspection_steps))
    return GradeResult(episode_id, True, None, tp, fp, fn, precision, recall, f1,
                       inspection_steps, reward, matches)

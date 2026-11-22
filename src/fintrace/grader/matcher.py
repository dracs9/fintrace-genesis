"""Maximum cardinality matching in issue-ID and submission order."""

from .normalize import Finding, Issue, IssueType


def eligible(truth: Issue, finding: Finding) -> bool:
    """Return true exactly when the finding satisfies all five clauses of §9.3."""
    tolerance = 1 if truth.type is IssueType.FX else 0
    return (truth.type == finding.type and truth.affected_ids == finding.affected_ids
            and truth.currency == finding.currency
            and abs(truth.expected_amount_minor - finding.claimed_amount_minor) <= tolerance
            and any(group <= finding.evidence_ids for group in truth.evidence_sets))


def match(issues: tuple[Issue, ...], findings: tuple[Finding, ...]) -> tuple[tuple[str, int], ...]:
    """Find a deterministic maximum matching using augmenting paths, O(T²F)."""
    ordered = sorted(issues, key=lambda issue: issue.issue_id)
    edges = [[j for j, finding in enumerate(findings) if eligible(issue, finding)] for issue in ordered]
    owners: list[int | None] = [None] * len(findings)

    def augment(i: int, visited: set[int]) -> bool:
        for j in edges[i]:
            if j in visited:
                continue
            visited.add(j)
            previous = owners[j]
            if previous is None or augment(previous, visited):
                owners[j] = i
                return True
        return False

    for i in range(len(ordered)):
        augment(i, set())
    return tuple(sorted((ordered[i].issue_id, j) for j, i in enumerate(owners) if i is not None))

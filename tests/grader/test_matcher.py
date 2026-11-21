from dataclasses import replace

from auditgym.grader.matcher import match
from auditgym.grader.normalize import Finding, Issue, IssueType


def test_augmenting_path_reassigns_first_hit() -> None:
    ids = frozenset(["doc:D-111111"])
    left, right = frozenset(["ledger:R-111111"]), frozenset(["ledger:R-222222"])
    a = Issue("iss_a", IssueType.FX, ids, 1, "KZT", (left, right))
    b = replace(a, issue_id="iss_b", evidence_sets=(left,))
    f = Finding(IssueType.FX, ids, 1, "KZT", left, "")
    assert match((b, a), (f, replace(f, evidence_ids=right))) == (("iss_a", 1), ("iss_b", 0))


def test_tie_break_does_not_depend_on_truth_order() -> None:
    ids = frozenset(["doc:D-111111"])
    a = Issue("iss_a", IssueType.FX, ids, 1, "KZT", (ids,))
    b = replace(a, issue_id="iss_b")
    f = Finding(IssueType.FX, ids, 1, "KZT", ids, "")
    assert match((a, b), (f, f)) == match((b, a), (f, f)) == (("iss_a", 1), ("iss_b", 0))

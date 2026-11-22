from dataclasses import replace
from itertools import permutations
from typing import Any

import pytest
from hypothesis import given, settings, strategies as st

from fintrace.grader.normalize import Issue, IssueType
from fintrace.grader.result import GradeResult
from fintrace.grader.reward import score

EPISODE = "ep_fedcba987654"


def issue(index: int = 0) -> Issue:
    doc, line = f"doc:D-{index:06d}", f"ledger:R-{index:06d}"
    return Issue(f"iss_{index:02d}", IssueType.FX, frozenset([doc]), 250, "KZT",
                 (frozenset([doc, line]),))


def answer(item: Issue) -> dict[str, Any]:
    return {"type": item.type.value, "affected_ids": sorted(item.affected_ids),
            "currency": item.currency, "claimed_amount_minor": item.expected_amount_minor,
            "evidence_ids": sorted(item.evidence_sets[0]), "explanation": ""}


def submit(rows: list, truth: tuple[Issue, ...] | None = None, steps: int = 10) -> GradeResult:
    return score(EPISODE, (issue(),) if truth is None else truth,
                 {"episode_id": EPISODE, "findings": rows}, steps)


def test_full_beats_silence() -> None:
    assert submit([answer(issue())]).reward == 0.98 > submit([]).reward == -1


def test_silence_beats_false_accusation_on_clean() -> None:
    silent = submit([], ())
    assert (silent.precision, silent.recall, silent.f1, silent.reward) == (0, 0, 1, 0.98)
    assert submit([answer(issue())], ()).reward == -1


def test_repeated_tp_is_fp() -> None:
    row = answer(issue())
    result = submit([row, row])
    assert (result.tp, result.fp, result.fn, result.f1) == (1, 1, 0, 2 / 3)
    assert result.reward < submit([row]).reward


def test_flag_all_loses_to_honest_partial() -> None:
    truth = (issue(), issue(1))
    partial = submit([answer(truth[0])], truth)
    spam = submit([answer(issue(i)) for i in range(30)], truth)
    assert (partial.tp, partial.fn, spam.tp, spam.fp) == (1, 1, 2, 28)
    assert spam.reward == -0.77 < partial.reward


def test_step_cost_for_10_and_90() -> None:
    row = answer(issue())
    early, late = submit([row], steps=10), submit([row], steps=90)
    assert early.reward == 0.98 and late.reward == pytest.approx(0.82)
    assert early.reward - late.reward == pytest.approx(0.16, abs=1e-15)


def test_guessed_ids_do_not_match() -> None:
    wrong = answer(issue(1))
    result = submit([wrong])
    assert result.valid and (result.tp, result.fp, result.fn) == (0, 1, 1)


def test_partial_evidence_from_two_alternatives() -> None:
    item = issue()
    alt = frozenset(["ledger:R-000099", "ledger:R-000098"])
    item = replace(item, evidence_sets=(*item.evidence_sets, alt))
    row = answer(item)
    row["evidence_ids"] = [row["evidence_ids"][0], "ledger:R-000099"]
    assert submit([row], (item,)).tp == 0
    row["evidence_ids"] += ["ledger:R-000098"]
    assert submit([row], (item,)).tp == 1


def test_wrong_currency() -> None:
    result = submit([dict(answer(issue()), currency="USD")])
    assert result.valid and result.tp == 0


@pytest.mark.parametrize("kind,delta", [(IssueType.FX, -2), (IssueType.FX, 2),
                                       (IssueType.CUTOFF, 1), (IssueType.DUPLICATE, -1),
                                       (IssueType.MISALLOCATION, 1)])
def test_amount_outside_tolerance(kind: IssueType, delta: int) -> None:
    item = replace(issue(), type=kind)
    result = submit([dict(answer(item), claimed_amount_minor=250 + delta)], (item,))
    assert result.valid and result.tp == 0


@pytest.mark.parametrize("delta", [-1, 1])
def test_tolerance_boundary(delta: int) -> None:
    assert submit([dict(answer(issue()), claimed_amount_minor=250 + delta)]).tp == 1


def test_wrong_type() -> None:
    assert submit([dict(answer(issue()), type="cutoff_error")]).tp == 0


def test_extra_affected_id() -> None:
    row = answer(issue())
    row["affected_ids"].append("doc:D-999999")
    assert submit([row]).tp == 0


@pytest.mark.parametrize("field,value", [
    ("findings", {}), ("episode_id", "ep_000000000000"), ("root_extra", 3),
    ("length", 31), ("evidence_ids", ["doc:D-000000"] * 11),
    ("explanation", " " * 1001), ("claimed_amount_minor", True),
    ("claimed_amount_minor", 250.0), ("currency", "ZZZ"), ("type", "other"),
    ("affected_ids", []), ("affected_ids", ["doc:D-000000"] * 11),
    ("affected_ids", ["ent:E-1234"]), ("finding_extra", None), ("missing", None),
])
def test_invalid_structure_is_minus_one(field: str, value: Any) -> None:
    row = answer(issue())
    payload = {"episode_id": EPISODE, "findings": [row]}
    if field == "length":
        payload["findings"] = [row] * value
    elif field in ("findings", "episode_id", "root_extra"):
        payload[field] = value
    elif field == "missing":
        del row["explanation"]
    else:
        row[field] = value
    result = score(EPISODE, (issue(),), payload, 0)
    assert (result.valid, result.reward, result.error) == (False, -1, "invalid arguments")


def test_timeout_is_empty_response_at_100_steps() -> None:
    for truth in ((), (issue(),)):
        timeout = score(EPISODE, truth, None, 100)
        assert timeout == submit([], truth, 100)
        assert timeout.reward == (0.8 if not truth else -1)


def test_permutation_invariant_reward() -> None:
    truth = (issue(), issue(1))
    rows = [answer(issue(i)) for i in (0, 0, 1, 2)]
    for ordering in permutations(rows):
        assert submit(list(ordering), truth).reward == submit(rows, truth).reward


@settings(derandomize=True, deadline=None, max_examples=100)
@given(st.integers(1, 12), st.integers(0, 11), st.integers(0, 16), st.integers(0, 99))
def test_monotonicity(n: int, hits: int, false_positives: int, steps: int) -> None:
    truth = tuple(issue(i) for i in range(n))
    hits = min(hits, n - 1)
    rows = [answer(item) for item in truth[:hits]] + [answer(issue(99))] * false_positives
    base = submit(rows, truth, steps)
    more_tp = submit([*rows, answer(truth[hits])], truth, steps)
    more_fp = submit([*rows, answer(issue(98))], truth, steps)
    assert more_tp.tp == base.tp + 1 and more_fp.fp == base.fp + 1
    assert more_tp.reward >= base.reward >= more_fp.reward

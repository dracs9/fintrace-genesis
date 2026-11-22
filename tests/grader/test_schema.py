import pytest

from fintrace.grader.normalize import IssueType
from fintrace.grader.schema import parse_findings


def test_normalization_before_matching() -> None:
    episode = "ep_123456abcdef"
    row = {"type": "  cutoff_error ", "affected_ids": [" DOC:D-123abc ", "doc:D-123abc"],
           "evidence_ids": [" LEDGER:R-abcdef "], "currency": " KZT ",
           "claimed_amount_minor": 123, "explanation": "  e\u0301  "}
    result, = parse_findings({"episode_id": " " + episode, "findings": [row]}, episode)
    assert result.type is IssueType.CUTOFF
    assert result.affected_ids == {"doc:D-123abc"}
    assert result.evidence_ids == {"ledger:R-abcdef"}
    assert result.explanation == "é"


@pytest.mark.parametrize("amount", [False, True, 0.0, 1.0, -1, "0"])
def test_requires_python_integer(amount: object) -> None:
    episode = "ep_123456abcdef"
    row = {"type": "cutoff_error", "affected_ids": ["doc:D-123abc"], "evidence_ids": [],
           "currency": "KZT", "claimed_amount_minor": amount, "explanation": ""}
    with pytest.raises(ValueError, match="^invalid arguments$"):
        parse_findings({"episode_id": episode, "findings": [row]}, episode)

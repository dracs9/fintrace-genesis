"""Canonical representation for schema-checked records."""

import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class IssueType(StrEnum):
    DUPLICATE = "duplicate_invoice"
    MISALLOCATION = "payment_misallocation"
    FX = "fx_rate_error"
    CUTOFF = "cutoff_error"


def clean(value: str) -> str:
    """Strip whitespace and return NFC, preserving case."""
    return unicodedata.normalize("NFC", value.strip())


def canonical_id(value: str) -> str:
    """Fold only the namespace of an ID, preserving its case-sensitive suffix."""
    prefix, separator, suffix = clean(value).partition(":")
    return prefix.lower() + separator + suffix


@dataclass(frozen=True)
class Finding:
    type: IssueType
    affected_ids: frozenset[str]
    claimed_amount_minor: int
    currency: str
    evidence_ids: frozenset[str]
    explanation: str


@dataclass(frozen=True)
class Issue:
    issue_id: str
    type: IssueType
    affected_ids: frozenset[str]
    expected_amount_minor: int
    currency: str
    evidence_sets: tuple[frozenset[str], ...]


def finding_from(row: dict[str, Any]) -> Finding:
    """Freeze a canonical, validated finding without changing its amount."""
    return Finding(IssueType(row["type"]), frozenset(row["affected_ids"]),
                   row["claimed_amount_minor"], row["currency"], frozenset(row["evidence_ids"]),
                   row["explanation"])


def issue_from(row: dict[str, Any]) -> Issue:
    """Freeze a validated canonical truth item, retaining every alternative proof."""
    return Issue(row["issue_id"], IssueType(row["type"]), frozenset(row["affected_ids"]),
                 row["expected_amount_minor"], row["currency"],
                 tuple(frozenset(group) for group in row["evidence_sets"]))

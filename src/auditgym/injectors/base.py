"""Immutable mutation interface. Document and row handles precede final ID assignment."""

from dataclasses import dataclass, replace
from datetime import timedelta, timezone
from typing import Protocol

from numpy.random import Generator

from ..books import Books, Row
from ..world.events import Invoice, Payment


Document = Invoice | Payment
COMPANY_TZ = timezone(timedelta(hours=5))


@dataclass(frozen=True)
class Candidate:
    document: Document
    variant: str
    line: int = 0


@dataclass(frozen=True)
class TruthItem:
    issue_id: str
    type: str
    affected_ids: tuple[str, ...]
    expected_amount_minor: int
    currency: str
    evidence_sets: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class Mutation:
    truth_items: tuple[TruthItem, ...]
    diff: dict[str, list[str]]
    causal_note: str
    decomposition: dict | None
    books: Books


class Injector(Protocol):
    type: str

    def eligible(self, books: Books) -> list[Candidate]: ...

    def apply(self, cand: Candidate, books: Books, rng: Generator) -> Mutation: ...


def key(doc: Document) -> tuple[str, int]:
    """Return the internal document reference shared by books and provenance."""
    return ("invoice" if isinstance(doc, Invoice) else "payment", doc.key)


def handle(doc: Document) -> str:
    """Return a deterministic document handle for later ID remapping."""
    table, index = key(doc)
    return f"doc:{table}/{index}"


def lines(books: Books, doc: Document) -> list[Row]:
    """Read a document's postings in their stable internal order."""
    ref = key(doc)
    return [row for row in books.ledger if row.document == ref]


def row_handle(doc: Document, index: int) -> str:
    """Identify a ledger row independent of changes to other documents."""
    table, ident = key(doc)
    return f"ledger:{table}/{ident}/{index}"


def finish(books: Books, doc: Document, rows: list[Row], kind: str, amount: int,
           affected: tuple[str, ...], evidence: tuple[tuple[str, ...], ...], note: str,
           *, added_invoice: bool = False) -> Mutation:
    """Replace one balanced posting group and derive its diff from the actual changes."""
    old = lines(books, doc)
    if amount <= 0 or sum(r.debit_minor - r.credit_minor for r in rows):
        raise ValueError("invalid mutation")
    ref = key(doc)
    ledger = tuple(r for r in books.ledger if r.document != ref) + tuple(rows)
    events = books.events
    added = []
    if added_invoice and isinstance(doc, Invoice):
        events = replace(events, invoices=events.invoices + (doc,))
        added.append(handle(doc))
    added.extend(row_handle(doc, i) for i in range(len(old), len(rows)))
    diff = {"added": sorted(added),
            "removed": [row_handle(doc, i) for i in range(len(rows), len(old))],
            "modified": [row_handle(doc, i) for i, (a, b) in enumerate(zip(old, rows)) if a != b]}
    truth = TruthItem("iss_01", kind, tuple(sorted(affected)), amount, books.world.base_currency, evidence)
    return Mutation((truth,), diff, note, None, replace(books, events=events, ledger=ledger))

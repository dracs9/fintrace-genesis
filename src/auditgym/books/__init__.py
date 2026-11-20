"""Clean double-entry books, before any distractors or mutations."""

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Callable

from ..world import World
from ..world.events import Events, Invoice, Payment


Converter = Callable[[int, str, str, Decimal], int]
_COMPANY_TZ = timezone(timedelta(hours=5))


@dataclass(frozen=True)
class Row:
    document: tuple[str, int]  # Internal provenance for I5. Omit from open CSV.
    timestamp: datetime
    account: str
    debit_minor: int
    credit_minor: int
    currency: str
    entity: int
    invoice: int | None
    payment: int | None


@dataclass(frozen=True)
class Books:
    world: World
    events: Events
    ledger: tuple[Row, ...]


def _base(world: World, amount: int, currency: str, at: datetime, convert: Converter | None) -> int:
    if currency == world.base_currency:
        return amount
    day = at.astimezone(_COMPANY_TZ).date()
    try:
        quote = world.market.lookup(currency, day)
    except LookupError as exc:
        raise ValueError("I7: missing FX quote") from exc
    if convert is None:
        from ..common.money import convert as shared_convert
        convert = shared_convert
    return convert(amount, currency, world.base_currency, quote.rate)


def _document_rows(world: World, doc: Invoice | Payment, sales: dict[int, Invoice],
                   convert: Converter | None) -> tuple[Row, ...]:
    rows: list[Row] = []
    is_payment = isinstance(doc, Payment)
    moment = doc.timestamp if isinstance(doc, Payment) else doc.issued_at
    at = moment.astimezone(_COMPANY_TZ)

    def line(account: str, signed: int, invoice: int | None = None) -> None:
        if signed:
            rows.append(Row(("payment" if is_payment else "invoice", doc.key), at, account,
                            max(signed, 0), max(-signed, 0), world.base_currency, doc.entity,
                            invoice, doc.key if is_payment else None))

    if isinstance(doc, Invoice):
        original = doc.key if doc.original is None else doc.original
        # A credit note reverses the receivable at its original carrying rate.
        value = _base(world, doc.total_minor, doc.currency, sales[original].issued_at, convert)
        if doc.original is None:
            line("1200", value, original)
            line("4000", -value, original)
        else:
            line("4100", value, original)
            line("1200", -value, original)
    else:
        line("1010", _base(world, doc.amount_minor, doc.currency, at, convert))
        line("6100", _base(world, doc.fee_minor, doc.currency, at, convert))
        for ref in doc.refs:
            historical = sales[ref.invoice].issued_at
            line("1200", -_base(world, ref.amount_minor, doc.currency, historical, convert), ref.invoice)
        line("2150", -_base(world, doc.unapplied_minor, doc.currency, at, convert))
        line("7100", sum(r.credit_minor - r.debit_minor for r in rows))
    return tuple(rows)


def _documents(world: World, events: Events) -> dict[int, Invoice]:
    sales = {i.key: i for i in events.invoices if i.original is None}
    entities = {e.index: e.currency for e in world.entities}
    if (len({i.key for i in events.invoices}) != len(events.invoices)
            or len({p.key for p in events.payments}) != len(events.payments)):
        raise ValueError("I5: duplicate document")
    consumed: Counter[int] = Counter()
    for doc in (*events.invoices, *events.payments):
        if entities.get(doc.entity) != doc.currency:
            raise ValueError("I6: invalid counterparty")
        at = doc.timestamp if isinstance(doc, Payment) else doc.issued_at
        if at.utcoffset() is None or not (
            world.start <= at.astimezone(_COMPANY_TZ).date() <= world.end + timedelta(days=world.tail_days)
        ):
            raise ValueError("I5: invalid document date")
        if isinstance(doc, Payment):
            amounts = [doc.amount_minor, doc.fee_minor, doc.unapplied_minor]
            allocations = [(ref.invoice, ref.amount_minor) for ref in doc.refs]
            if (any(type(a) is not int or a < 0 for a in amounts)
                    or doc.amount_minor != sum(a for _, a in allocations) + doc.unapplied_minor - doc.fee_minor):
                raise ValueError("I4: invalid net receipt")
        else:
            if type(doc.total_minor) is not int or doc.total_minor <= 0:
                raise ValueError("I3: invalid invoice total")
            allocations = [] if doc.original is None else [(doc.original, doc.total_minor)]
        for key, amount in allocations:
            original = sales.get(key)
            if original is None or original.entity != doc.entity or original.issued_at > at:
                raise ValueError("I6: invalid invoice reference")
            if type(amount) is not int or amount <= 0:
                raise ValueError("I3: invalid allocation")
            consumed[key] += amount
            if consumed[key] > original.total_minor:
                raise ValueError("I3: overapplied invoice")
    return sales


def post(world: World, events: Events, convert: Converter | None = None) -> Books:
    """Post primary documents in KZT and check I1–I7 deterministically."""
    sales = _documents(world, events)
    rows = tuple(row for doc in (*events.invoices, *events.payments)
                 for row in _document_rows(world, doc, sales, convert))
    books = Books(world, events, rows)
    check_invariants(books, convert)
    return books


def check_invariants(books: Books, convert: Converter | None = None) -> None:
    """Raise ValueError for an inconsistent clean book. I4 uses net bank receipts."""
    world, events = books.world, books.events
    sales = _documents(world, events)
    expected: Counter[Row] = Counter()
    expected_ar: Counter[int] = Counter()
    for doc in (*events.invoices, *events.payments):
        planned = _document_rows(world, doc, sales, convert)
        if not planned:
            raise ValueError("I5: empty posting")
        expected.update(planned)
    for invoice in events.invoices:
        key = invoice.original if invoice.original is not None else invoice.key
        value = _base(world, invoice.total_minor, invoice.currency, sales[key].issued_at, convert)
        expected_ar[key] += value if invoice.original is None else -value
    for payment in events.payments:
        for ref in payment.refs:
            invoice = sales[ref.invoice]
            expected_ar[ref.invoice] -= _base(world, ref.amount_minor, invoice.currency,
                                              invoice.issued_at, convert)
    groups: Counter[tuple[str, int | None]] = Counter()
    balances: Counter[int] = Counter()
    payments = {p.key for p in events.payments}
    entities = {e.index for e in world.entities}
    total = 0
    for row in books.ledger:
        if (row.entity not in entities or (row.invoice is None and row.payment is None)
                or row.invoice is not None and row.invoice not in sales
                or row.payment is not None and row.payment not in payments):
            raise ValueError("I6: invalid ledger reference")
        if (type(row.debit_minor) is not int or type(row.credit_minor) is not int
                or min(row.debit_minor, row.credit_minor) != 0 or max(row.debit_minor, row.credit_minor) <= 0):
            raise ValueError("invalid ledger amount")
        net = row.debit_minor - row.credit_minor
        key = ("invoice", row.invoice) if row.payment is None else ("payment", row.payment)
        groups[key] += net
        total += net
        if row.account == "1200" and row.invoice is not None:
            balances[row.invoice] += net
    if total:
        raise ValueError("I2: unbalanced ledger")
    if any(groups[k] for k in sorted(groups)):
        raise ValueError("I1: unbalanced posting group")
    if balances != expected_ar or any(balances[k] < 0 for k in sorted(balances)):
        raise ValueError("I3: invalid receivable")
    if Counter(books.ledger) != expected or any(
        r.timestamp.utcoffset() != _COMPANY_TZ.utcoffset(None) for r in books.ledger
    ):
        raise ValueError("I5: document posting mismatch")

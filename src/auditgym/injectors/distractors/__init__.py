"""Select or create legitimate, auditable cases on clean books (В§7.7)."""

from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, time, timedelta, timezone
from fractions import Fraction
from typing import Any

from numpy.random import Generator

from ...books import Books, check_invariants, post
from ...common.money import exponent
from ...world import World
from ...world.events import Allocation, Events, Invoice, Payment, generate_events


DocKey = tuple[str, int]


@dataclass(frozen=True)
class Distractor:
    kind: str
    documents: tuple[DocKey, ...]  # Private provenance, resolved to doc IDs at serialization.


@dataclass(frozen=True)
class _Case:
    record: Distractor
    invoices: tuple[Invoice, ...] = ()
    payments: tuple[Payment, ...] = ()


def _case(kind: str, invoices: tuple[int, ...], payments: tuple[int, ...] = ()) -> Distractor:
    return Distractor(kind, tuple(("invoice", k) for k in invoices)
                      + tuple(("payment", k) for k in payments))


def _candidates(books: Books, rng: Generator) -> list[_Case]:
    events, world = books.events, books.world
    cases: list[_Case] = []
    receipts: dict[int, list[Payment]] = defaultdict(list)
    applied: dict[int, int] = defaultdict(int)
    for p in events.payments:
        refs = tuple(sorted({r.invoice for r in p.refs}))
        total = sum(r.amount_minor for r in p.refs)
        for ref in p.refs:
            applied[ref.invoice] += ref.amount_minor
        for key in refs:
            receipts[key].append(p)
        if len(refs) > 1:
            cases.append(_Case(_case("multi_invoice_payment", refs, (p.key,))))
        if p.fee_minor:
            cases.append(_Case(_case("bank_fee", refs, (p.key,))))
        if p.unapplied_minor and p.amount_minor > total:
            cases.append(_Case(_case("overpayment", refs, (p.key,))))
        elif not p.fee_minor:
            # One real overpayment also makes a previously ordinary receipt eligible.
            extra = max(1, p.amount_minor // 10)
            changed = replace(p, unapplied_minor=p.unapplied_minor + extra, amount_minor=p.amount_minor + extra)
            cases.append(_Case(_case("overpayment", refs, (p.key,)), payments=(changed,)))
    for inv in events.invoices:
        if inv.original is not None:
            applied[inv.original] += inv.total_minor
            cases.append(_Case(_case("credit_note", (inv.original, inv.key))))

    next_key = max((i.key for i in events.invoices), default=-1) + 1
    company_zone = timezone(timedelta(hours=5))
    previous_month = (world.start - timedelta(days=1)).strftime("%Y-%m")
    for inv in events.invoices:
        if inv.original is not None:
            continue
        pays = receipts[inv.key]
        paid = sum(r.amount_minor for p in pays for r in p.refs if r.invoice == inv.key)
        if len(pays) == 2 and paid == inv.total_minor:
            cases.append(_Case(_case("split_payment", (inv.key,), tuple(p.key for p in pays))))
        if pays and 0 < applied[inv.key] < inv.total_minor:
            cases.append(_Case(_case("partial_open", (inv.key,), tuple(p.key for p in pays))))
        company_date = inv.issued_at.astimezone(company_zone).date()
        period = inv.service_period or company_date.strftime("%Y-%m")
        original = replace(inv, service_period=period)
        day = world.end + timedelta(days=1) if company_date <= world.end else world.start
        issued = datetime.combine(day, time(12), company_zone).astimezone(inv.issued_at.tzinfo)
        new_period = day.strftime("%Y-%m")
        other = replace(inv, key=next_key, issued_at=issued, due_at=issued + timedelta(days=30),
                        service_period=new_period if new_period != period else previous_month)
        cases.append(_Case(_case("recurring_subscription", (inv.key, next_key)), (original, other)))
        next_key += 1

    # Construct the boundary explicitly in company time, then record client local time.
    foreign_zones = [e for e in world.entities if e.timezone != world.timezone]
    if foreign_zones:
        entity = foreign_zones[int(rng.integers(len(foreign_zones)))]
        zone = datetime.fromisoformat("2000-01-01T00:00" + entity.timezone).tzinfo
        boundary = datetime.combine(world.end + timedelta(days=1), time(), company_zone)
        offset = timedelta(minutes=30 if zone.utcoffset(None) < company_zone.utcoffset(None) else -30)
        issued = (boundary + offset).astimezone(zone)
        inv = Invoice(next_key, entity.index, entity.currency, issued, issued + timedelta(days=30),
                      int(rng.integers(1000, 500001)))
        cases.append(_Case(_case("boundary_correct", (next_key,)), (inv,)))
        next_key += 1

    exotic = []
    for currency in ("JPY", "KWD"):
        eligible = [e for e in world.entities if e.currency == currency]
        if eligible:
            entity = eligible[int(rng.integers(len(eligible)))]
            zone = datetime.fromisoformat("2000-01-01T00:00" + entity.timezone).tzinfo
            issued = datetime.combine(world.start, time(12), company_zone).astimezone(zone)
            exotic.append(Invoice(next_key, entity.index, currency, issued, issued + timedelta(days=30),
                                  int(rng.integers(1001, 500001))))
            next_key += 1
    if len(exotic) == 2:
        cases.append(_Case(_case("exotic_exponent", tuple(i.key for i in exotic)), tuple(exotic)))
    existing = {case.record.kind for case in cases}
    cases.extend(_missing_cash_cases(books, rng, next_key, existing))
    return cases


def _missing_cash_cases(books: Books, rng: Generator, next_key: int, existing: set[str]) -> list[_Case]:
    events, world = books.events, books.world
    company_zone = timezone(timedelta(hours=5))
    cases = []
    next_pay = max((p.key for p in events.payments), default=-1) + 1
    for kind in ("split_payment", "multi_invoice_payment", "partial_open", "overpayment", "bank_fee", "credit_note"):
        if kind in existing:
            continue
        # Build an isolated receivable when the input book lacks an eligible case.
        entity = world.entities[int(rng.integers(len(world.entities)))]
        zone = datetime.fromisoformat("2000-01-01T00:00" + entity.timezone).tzinfo
        day = world.start + timedelta(days=int(rng.integers((world.end - world.start).days + 1)))
        issued = datetime.combine(day, time(10), company_zone).astimezone(zone)
        total = int(rng.integers(1000, 500001))
        if kind == "split_payment":
            total = 2 * 10 ** (4 + exponent(entity.currency))
        sale = Invoice(next_key, entity.index, entity.currency, issued, issued + timedelta(days=30), total)
        next_key += 1
        docs = [sale]
        new_payments = []
        at = issued + timedelta(days=1)
        if kind == "credit_note":
            docs.append(replace(sale, key=next_key, issued_at=at, total_minor=total // 3, original=sale.key))
            next_key += 1
        else:
            if kind == "multi_invoice_payment":
                docs.append(replace(sale, key=next_key, total_minor=total + 1))
                next_key += 1
            refs = tuple(Allocation(i.key, i.total_minor) for i in docs)
            if kind in ("partial_open", "split_payment"):
                refs = (Allocation(sale.key, total // 2),)
            fee = total // 100 if kind == "bank_fee" else 0
            extra = total // 10 if kind == "overpayment" else 0
            net = sum(r.amount_minor for r in refs) + extra - fee
            payment = Payment(next_pay, entity.index, entity.currency, at, net, refs, fee, extra)
            new_payments.append(payment)
            next_pay += 1
            if kind == "split_payment":
                new_payments.append(replace(payment, key=next_pay, timestamp=at + timedelta(days=1)))
                next_pay += 1
        record = _case(kind, tuple(i.key for i in docs), tuple(p.key for p in new_payments))
        cases.append(_Case(record, tuple(docs), tuple(new_payments)))
    return cases


def apply(books: Books, rng: Generator, *, distractor_rate: Fraction | str | int,
          n_issues: int) -> tuple[Books, tuple[Distractor, ...]]:
    """Apply clean cases using stream('distract') and return checked books and provenance.

    Target the fraction of unique participating documents in the resulting book.
    Whole cases can cross the target. A clean episode always receives a case.
    Eligible kinds are sampled round-robin in random order to provide variety.
    """
    rate = Fraction(str(distractor_rate))
    if not 0 <= rate <= 1 or type(n_issues) is not int or not 0 <= n_issues <= 12:
        raise ValueError("invalid distractor configuration")
    check_invariants(books)
    if not rate and n_issues:
        return books, ()
    invoices = {i.key: i for i in books.events.invoices}
    payments = {p.key: p for p in books.events.payments}
    groups: dict[str, list[_Case]] = defaultdict(list)
    for case in _candidates(books, rng):
        groups[case.record.kind].append(case)
    kinds = sorted(groups)
    for kind in kinds:
        rng.shuffle(groups[kind])
    rng.shuffle(kinds)
    covered: set[DocKey] = set()
    records = []
    while (not records or len(covered) < rate * (len(invoices) + len(payments))) and kinds:
        for kind in tuple(kinds):
            options = groups[kind]
            while options:
                case = options.pop()
                if any(key not in covered for key in case.record.documents):
                    for invoice in case.invoices:
                        invoices[invoice.key] = invoice
                    for payment in case.payments:
                        payments[payment.key] = payment
                    covered.update(case.record.documents)
                    records.append(case.record)
                    break
            if not options:
                kinds.remove(kind)
            if records and len(covered) >= rate * (len(invoices) + len(payments)):
                break
    if not records and n_issues == 0:
        raise ValueError("clean book requires a distractor")
    events = Events(tuple(invoices[k] for k in sorted(invoices)), tuple(payments[k] for k in sorted(payments)))
    return post(books.world, events), tuple(records)


def prepare(world: World, config: dict[str, Any], events_rng: Generator, distract_rng: Generator,
            *, n_issues: int) -> tuple[Books, tuple[Distractor, ...]]:
    """Build the clean input to issue injection using the events and distract streams.

    Resolve n_issues before calling so a sampled clean episode receives a case
    even when the configured participation rate is zero.
    """
    events = generate_events(world, config["n_invoices"], events_rng, ordinary_only=True)
    return apply(post(world, events), distract_rng, distractor_rate=config["distractor_rate"],
                 n_issues=n_issues)

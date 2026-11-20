"""Primary documents with internal keys, prior to public ID assignment."""

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Mapping, Sequence

from numpy.random import Generator

from . import World


@dataclass(frozen=True)
class Invoice:
    key: int
    entity: int
    currency: str
    issued_at: datetime
    due_at: datetime
    total_minor: int
    original: int | None = None  # Set only for a credit note.
    service_period: str | None = None  # YYYY-MM for a billed service period.
    number_source: int | None = None  # Original number assigned by the IDs stream.
    number_style: str = "canonical"

    def number(self, assigned: Mapping[int, str]) -> str:
        """Render the billing number after opaque canonical numbers are assigned."""
        value = assigned[self.key if self.number_source is None else self.number_source]
        if self.number_style == "compact":
            return value.replace("-", "")
        if self.number_style == "lower":
            return value.lower()
        if self.number_style == "zero":
            return value.replace("INV-", "INV-0")
        return value


@dataclass(frozen=True)
class Allocation:
    invoice: int
    amount_minor: int


@dataclass(frozen=True)
class Payment:
    key: int
    entity: int
    currency: str
    timestamp: datetime
    amount_minor: int  # Net bank receipt.
    refs: tuple[Allocation, ...]
    fee_minor: int = 0
    unapplied_minor: int = 0


@dataclass(frozen=True)
class Events:
    invoices: tuple[Invoice, ...]
    payments: tuple[Payment, ...]


def generate_events(world: World, n_invoices: Sequence[int], rng: Generator, *,
                    ordinary_only: bool = False) -> Events:
    """Generate a reproducible calendar using the caller's named events stream.

    The range counts sales invoices. A credit note gets its own document key.
    Each scenario owns its invoices, so independently planned receipts cannot
    consume another scenario's receivables. With ordinary_only, emit full
    settlements as the baseline for a separately budgeted distractor stage.
    """
    low, high = n_invoices
    if not 1 <= low <= high:
        raise ValueError("invalid invoice range")
    remaining = int(rng.integers(low, high + 1))
    invoices: list[Invoice] = []
    payments: list[Payment] = []
    scenarios = ("full", "partial", "split", "multi", "credit", "fee", "overpayment")
    if ordinary_only:
        scenarios = ("full",)
    company_zone = timezone(timedelta(hours=5))
    schedule: list[str] = []
    while remaining:
        if not schedule:
            schedule = [scenarios[int(i)] for i in rng.permutation(len(scenarios))]
        scenario = schedule.pop()
        count = 2 if scenario == "multi" and remaining >= 2 else 1
        entity = world.entities[int(rng.integers(len(world.entities)))]
        zone = datetime.fromisoformat("2000-01-01T00:00" + entity.timezone).tzinfo
        # Leave two days for both split receipts, including at the end of the tail.
        days = (world.end - world.start).days + world.tail_days - 2
        day = world.start + timedelta(days=int(rng.integers(days + 1)))
        issued = (datetime.combine(day, time(), company_zone)
                  + timedelta(minutes=int(rng.integers(1440)))).astimezone(zone)
        refs = []
        for _ in range(count):
            total = int(rng.integers(1000, 500_001))
            if scenario == "split":
                # Both halves convert exactly at any six-place rate, even in KWD.
                quantum = {"KWD": 10_000_000, "JPY": 10_000}.get(entity.currency, 1_000_000)
                total = 2 * quantum * int(rng.integers(1, 5))
            key = len(invoices)
            invoices.append(Invoice(key, entity.index, entity.currency, issued,
                                    issued + timedelta(days=30), total))
            refs.append(Allocation(key, total))
        remaining -= count
        next_day = issued + timedelta(days=1)
        if scenario == "credit":
            invoices.append(Invoice(len(invoices), entity.index, entity.currency,
                                    next_day, next_day, total // 3, key))
            continue
        if scenario in ("partial", "split"):
            refs = [Allocation(key, total // 2)]
        fee = total // 100 if scenario == "fee" else 0
        excess = total // 10 if scenario == "overpayment" else 0
        net = sum(ref.amount_minor for ref in refs) + excess - fee
        payments.append(Payment(len(payments), entity.index, entity.currency,
                                next_day, net, tuple(refs), fee, excess))
        if scenario == "split":
            payments.append(Payment(len(payments), entity.index, entity.currency,
                                    next_day + timedelta(days=1), net, tuple(refs)))
    return Events(tuple(invoices), tuple(payments))

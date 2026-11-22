"""Daily business-day quotes in KZT per unit of source currency."""

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Sequence

from numpy.random import Generator


@dataclass(frozen=True)
class Quote:
    date: date
    source_currency: str
    rate: Decimal


@dataclass(frozen=True)
class Market:
    quotes: tuple[Quote, ...]
    end: date

    def lookup(self, currency: str, day: date) -> Quote:
        """Return the latest eligible quote, including its actual evidence date."""
        if day <= self.end:
            for quote in reversed(self.quotes):
                if quote.source_currency == currency and quote.date <= day:
                    return quote
        raise LookupError("quote unavailable")


def build_market(start: date, end: date, currencies: Sequence[str], rng: Generator) -> Market:
    """Generate six-place weekday quotes from the supplied world stream.

    Each move is equiprobably up or down by 0.4% of the previous close,
    expressed in whole micro-KZT ticks. The first close is the base quote.
    """
    base_ticks = {"USD": 500_000_000, "EUR": 545_000_000, "JPY": 3_400_000, "KWD": 1_630_000_000}
    sources = sorted(c for c in currencies if c != "KZT")
    ticks = {c: base_ticks[c] for c in sources}
    first = start - timedelta(days=10)
    first -= timedelta(days=max(0, first.weekday() - 4))
    last = end + timedelta(days=3)
    quotes = []
    for offset in range((last - first).days + 1):
        day = first + timedelta(days=offset)
        if day.weekday() >= 5:
            continue
        for source in sources:
            if offset:
                move = ticks[source] // 250
                ticks[source] += move if rng.integers(2) else -move
            units, fraction = divmod(ticks[source], 1_000_000)
            quotes.append(Quote(day, source, Decimal(f"{units}.{fraction:06d}")))
    return Market(tuple(quotes), last)

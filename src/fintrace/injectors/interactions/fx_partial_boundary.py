"""IA: move correctly valued cash across the cut-off, then isolate the FX difference.

Cut-off is the entire net bank receipt at the correct company-date rate. FX is
the absolute difference between posted cash and that correctly valued receipt.
Both evidence sets include the payment, the 1010 row and the actual correct
quote. The erroneous 1010 amount alone cannot establish the cut-off amount.
"""

from dataclasses import replace
from datetime import datetime, time, timedelta

from numpy.random import Generator

from ...books import Books, post
from ...common.money import convert
from ...world.events import Allocation, Events, Invoice, Payment
from ..base import COMPANY_TZ, Candidate, Mutation, finish, handle, lines
from ..fx_rate_error import FxRateError


TYPES = ("cutoff_error", "fx_rate_error")


def stage(books: Books, rng: Generator) -> tuple[Books, Candidate] | None:
    """Add a valid partial receipt and its sale before mutation; require distinct effective quotes."""
    world = books.world
    first = world.end + timedelta(days=1)
    boundary = datetime.combine(first, time(0, 1), COMPANY_TZ)
    eligible = []
    for entity in world.entities:
        zone = datetime.fromisoformat("2000-01-01T00:00" + entity.timezone).tzinfo
        at = boundary.astimezone(zone)
        if entity.currency == world.base_currency or at.date() != world.end:
            continue
        old = world.market.lookup(entity.currency, world.end)
        new = world.market.lookup(entity.currency, first)
        if old.rate != new.rate:
            eligible.append((entity, at, old.rate, new.rate))
    if not eligible:
        return None
    entity, at, old_rate, new_rate = eligible[int(rng.integers(len(eligible)))]
    amount = int(rng.integers(10000, 500001))
    while convert(amount, entity.currency, "KZT", old_rate) == convert(amount, entity.currency, "KZT", new_rate):
        amount *= 2
    inv = Invoice(max((i.key for i in books.events.invoices), default=-1) + 1, entity.index, entity.currency,
                  at - timedelta(days=3), at + timedelta(days=27), amount * 2)
    pay = Payment(max((p.key for p in books.events.payments), default=-1) + 1, entity.index, entity.currency,
                  at, amount, (Allocation(inv.key, amount),))
    events = Events(books.events.invoices + (inv,), books.events.payments + (pay,))
    return post(world, events), Candidate(pay, "client_date")


def apply(cand: Candidate, books: Books, rng: Generator) -> Mutation:
    """Apply both errors to the same payment, measuring cut-off before FX distortion."""
    pay = cand.document
    fx = FxRateError().apply(cand, books, rng)
    fx_item, = fx.truth_items
    wrong_rows = lines(fx.books, pay)
    shifted = [replace(r, timestamp=r.timestamp - timedelta(days=1)) for r in wrong_rows]
    correct = next(r.debit_minor for r in lines(books, pay) if r.account == "1010")
    wrong = next(r.debit_minor for r in shifted if r.account == "1010")
    result = finish(books, pay, shifted, TYPES[0], correct, (handle(pay),), fx_item.evidence_sets,
                    "Partial receipt posted in the closing month using its last available quote.")
    return replace(result, truth_items=(*result.truth_items, replace(fx_item, issue_id="iss_02")),
                   decomposition={"cutoff_error": "cash at correct company-date rate",
                                  "fx_rate_error": "absolute difference between posted and correct cash",
                                  "correct_cash_minor": correct, "posted_cash_minor": wrong})

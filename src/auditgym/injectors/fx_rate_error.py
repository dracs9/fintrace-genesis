from dataclasses import replace
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext

from numpy.random import Generator

from ..books import Books
from ..common.money import convert
from ..world.events import Invoice
from .base import COMPANY_TZ, Candidate, Document, Mutation, finish, handle, lines, row_handle
from .registry import register


def _rates(books: Books, doc: Document) -> tuple[str, Decimal, dict[str, Decimal]]:
    world = books.world
    at = doc.issued_at if isinstance(doc, Invoice) else doc.timestamp
    correct = world.market.lookup(doc.currency, at.astimezone(COMPANY_TZ).date())
    choices = {}
    try:
        local = world.market.lookup(doc.currency, at.date())
    except LookupError:
        local = None
    if local and local.date != correct.date:
        choices["client_date"] = local.rate
    quotes = [q for q in world.market.quotes if q.source_currency == doc.currency]
    index = quotes.index(correct)
    if index:
        choices["previous_day"] = quotes[index - 1].rate
    if index + 1 < len(quotes):
        choices["next_day"] = quotes[index + 1].rate
    with localcontext(Context(prec=80, rounding=ROUND_HALF_EVEN)):
        choices["inverse"] = Decimal("1") / correct.rate
    return f"fx:{correct.date}:{doc.currency}:{world.base_currency}", correct.rate, choices


@register("fx_rate_error")
class FxRateError:
    type: str

    def eligible(self, books: Books) -> list[Candidate]:
        """Offer only wrong rates producing a positive posting and a measurable delta."""
        result = []
        for doc in (*books.events.invoices, *books.events.payments):
            if doc.currency == books.world.base_currency or isinstance(doc, Invoice) and doc.original is not None:
                continue
            amount = doc.total_minor if isinstance(doc, Invoice) else doc.amount_minor
            _, rate, choices = _rates(books, doc)
            correct = convert(amount, doc.currency, books.world.base_currency, rate)
            for variant in sorted(choices):
                posted = convert(amount, doc.currency, books.world.base_currency, choices[variant])
                if posted > 0 and posted != correct:
                    result.append(Candidate(doc, variant))
        return result

    def apply(self, cand: Candidate, books: Books, rng: Generator) -> Mutation:
        """Misvalue the document while retaining historical invoice allocations."""
        doc = cand.document
        fx_id, correct, choices = _rates(books, doc)
        rate = choices[cand.variant]
        base = books.world.base_currency
        old = lines(books, doc)
        if isinstance(doc, Invoice):
            amount = doc.total_minor
            wrong = convert(amount, doc.currency, base, rate)
            rows = [replace(r, debit_minor=wrong if r.debit_minor else 0,
                            credit_minor=wrong if r.credit_minor else 0) for r in old]
            account = "1200"
        else:
            amount = doc.amount_minor
            account = "1010"
            rows = []
            values = {"1010": amount, "6100": doc.fee_minor, "2150": doc.unapplied_minor}
            for r in old:
                if r.account in values:
                    value = convert(values[r.account], doc.currency, base, rate)
                    if value:
                        rows.append(replace(r, debit_minor=value if r.account != "2150" else 0,
                                            credit_minor=value if r.account == "2150" else 0))
                elif r.account != "7100":
                    rows.append(r)
            net = sum(r.credit_minor - r.debit_minor for r in rows)
            if net:
                rows.append(replace(old[0], account="7100", invoice=None, debit_minor=max(net, 0),
                                    credit_minor=max(-net, 0)))
        index = next(i for i, r in enumerate(rows) if r.account == account)
        delta = abs(rows[index].debit_minor - convert(amount, doc.currency, base, correct))
        return finish(books, doc, rows, self.type, delta, (handle(doc),),
                      ((handle(doc), row_handle(doc, index), fx_id),), f"Used {cand.variant} FX quote.")

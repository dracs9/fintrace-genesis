from dataclasses import replace
from datetime import timedelta

from numpy.random import Generator

from ..books import Books
from ..world.events import Invoice
from .base import COMPANY_TZ, Candidate, Mutation, finish, handle, lines, row_handle
from .registry import register


@register("cutoff_error")
class CutoffError:
    type: str

    def eligible(self, books: Books) -> list[Candidate]:
        """Select documented revenue or cash in the two closing days or tail."""
        result = []
        for doc in (*books.events.invoices, *books.events.payments):
            if isinstance(doc, Invoice) and doc.original is not None:
                continue
            at = doc.issued_at if isinstance(doc, Invoice) else doc.timestamp
            offset = (at.astimezone(COMPANY_TZ).date() - books.world.end).days
            if -1 <= offset <= books.world.tail_days:
                for i, row in enumerate(lines(books, doc)):
                    if row.account in ("4000", "1010"):
                        result.append(Candidate(doc, "into_period" if offset > 0 else "into_tail", i))
        return result

    def apply(self, cand: Candidate, books: Books, rng: Generator) -> Mutation:
        """Shift the complete posting across month end while retaining the primary document."""
        doc = cand.document
        rows = lines(books, doc)
        day = books.world.end
        if cand.variant == "into_tail":
            day += timedelta(days=int(rng.integers(1, books.world.tail_days + 1)))
        else:
            day -= timedelta(days=int(rng.integers(2)))
        shifted = [replace(r, timestamp=r.timestamp.replace(year=day.year, month=day.month, day=day.day))
                   for r in rows]
        row = rows[cand.line]
        return finish(books, doc, shifted, self.type, row.credit_minor + row.debit_minor, (handle(doc),),
                      ((handle(doc), row_handle(doc, cand.line)),), f"Posting moved {cand.variant}.")

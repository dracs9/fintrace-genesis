from dataclasses import replace
from datetime import timedelta

from numpy.random import Generator

from ..books import Books, _document_rows
from ..world.events import Invoice
from .base import COMPANY_TZ, Candidate, Mutation, finish, handle, row_handle
from .registry import register


@register("duplicate_invoice")
class DuplicateInvoice:
    type: str

    def eligible(self, books: Books) -> list[Candidate]:
        """Find sales that can be reissued later within the reporting month."""
        return [Candidate(i, "reissue") for i in books.events.invoices
                if i.original is None and i.number_source is None
                and books.world.start <= i.issued_at.astimezone(COMPANY_TZ).date() < books.world.end]

    def apply(self, cand: Candidate, books: Books, rng: Generator) -> Mutation:
        """Reissue a sale with the same service and a normalized-equivalent number."""
        original = cand.document
        if not isinstance(original, Invoice):
            raise ValueError("invoice required")
        day = original.issued_at.astimezone(COMPANY_TZ).date()
        limit = min(10, (books.world.end - day).days)
        delay = timedelta(days=int(rng.integers(1, limit + 1)))
        style = ("compact", "lower", "zero")[int(rng.integers(3))]
        new = replace(original, key=max(i.key for i in books.events.invoices) + 1,
                      issued_at=original.issued_at + delay, due_at=original.due_at + delay,
                      number_source=original.key, number_style=style)
        rows = list(_document_rows(books.world, new, {new.key: new}, None))
        docs = (handle(original), handle(new))
        evidence = tuple((*docs, row_handle(new, i)) for i in range(2))
        return finish(books, new, rows, self.type, rows[0].debit_minor, (handle(new),), evidence,
                      f"Sale reissued after {delay.days} days with a {style} billing number.", added_invoice=True)

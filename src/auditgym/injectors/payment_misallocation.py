from dataclasses import replace

from numpy.random import Generator

from ..books import Books
from .base import Candidate, Mutation, finish, handle, lines, row_handle
from .registry import register


@register("payment_misallocation")
class PaymentMisallocation:
    type: str

    def eligible(self, books: Books) -> list[Candidate]:
        """Find independently evidenced full and partial diversions of remittances."""
        result = []
        for p in books.events.payments:
            if p.unapplied_minor:
                continue  # The evidence row must represent only the unexplained diversion.
            for index, row in enumerate(lines(books, p)):
                if row.account == "1200" and any(r.invoice == row.invoice for r in p.refs):
                    result.append(Candidate(p, "full", index))
                    if row.credit_minor > 1:
                        result.append(Candidate(p, "partial", index))
        return result

    def apply(self, cand: Candidate, books: Books, rng: Generator) -> Mutation:
        """Divert the selected historical AR credit to its own unapplied-cash row."""
        p = cand.document
        rows = lines(books, p)
        old = rows[cand.line]
        delta = old.credit_minor
        index = cand.line
        if cand.variant == "partial":
            delta = int(rng.integers(1, delta))
            rows[index] = replace(old, credit_minor=old.credit_minor - delta)
            index = len(rows)
            rows.append(old)
        rows[index] = replace(old, invoice=None, account="2150", credit_minor=delta)
        docs = (handle(p), f"doc:invoice/{old.invoice}")
        return finish(books, p, rows, self.type, delta, docs, ((*docs, row_handle(p, index)),),
                      f"{cand.variant} diversion of the intended AR credit to unapplied cash.")

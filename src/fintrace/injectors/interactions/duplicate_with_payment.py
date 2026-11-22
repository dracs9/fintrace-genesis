"""IB: recognize the duplicate sale and the misapplied historical AR credit separately.

The duplicate amount is its base-currency 1200 debit (equivalently its 4000
credit). Misallocation is the base-currency 1200 credit actually redirected
from the original invoice; remittance and cash remain unchanged.
"""

from dataclasses import replace

from numpy.random import Generator

from ...books import Books
from ..base import Candidate, Mutation, finish, handle, lines, row_handle
from ..duplicate_invoice import DuplicateInvoice


TYPES = ("duplicate_invoice", "payment_misallocation")


def stage(books: Books, rng: Generator) -> tuple[Books, Candidate] | None:
    """Pick a real remittance leaving at least one day to issue the duplicate first."""
    sales = {c.document.key: c.document for c in DuplicateInvoice().eligible(books)}
    candidates = []
    for p in books.events.payments:
        refs = {r.invoice for r in p.refs}
        for i, row in enumerate(lines(books, p)):
            if row.account == "1200" and row.invoice in refs and row.invoice in sales:
                if (p.timestamp - sales[row.invoice].issued_at).days >= 1:
                    candidates.append(Candidate(p, "redirect", i))
    if not candidates:
        return None
    return books, candidates[int(rng.integers(len(candidates)))]


def apply(cand: Candidate, books: Books, rng: Generator) -> Mutation:
    """Reissue before settlement and relink one AR credit without changing its value."""
    pay = cand.document
    rows = lines(books, pay)
    credit = rows[cand.line]
    original = next(i for i in books.events.invoices if i.key == credit.invoice)
    max_days = (pay.timestamp - original.issued_at).days
    dup = DuplicateInvoice().reissue(Candidate(original, "reissue"), books, rng, max_days)
    new = dup.books.events.invoices[-1]
    rows[cand.line] = replace(credit, invoice=new.key)
    docs = (handle(pay), handle(original))
    result = finish(dup.books, pay, rows, TYPES[1], credit.credit_minor, docs,
                    ((*docs, row_handle(pay, cand.line)),), "Sale reissued and intended payment credited to its duplicate.")
    return replace(result, truth_items=(*dup.truth_items, replace(result.truth_items[0], issue_id="iss_02")),
                   diff={k: sorted(dup.diff[k] + result.diff[k]) for k in sorted(result.diff)},
                   decomposition={"duplicate_invoice": "base amount of the new sale",
                                  "payment_misallocation": "historical AR credit redirected to the duplicate",
                                  "duplicate_base_minor": dup.truth_items[0].expected_amount_minor,
                                  "redirected_base_minor": credit.credit_minor})

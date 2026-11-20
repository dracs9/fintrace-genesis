"""Generator-side evidence checks on the final public values and remapped IDs."""

from datetime import datetime
from typing import Any

from ..common.money import convert, parse_rate
from ..injectors.base import COMPANY_TZ
from ..injectors.base import TruthItem, handle, lines
from ..world.events import Invoice, Payment
from . import Books


def _number(value: str) -> str:
    compact = "".join(ch for ch in value.upper() if ch not in " -_")
    return "INV" + str(int(compact.removeprefix("INV")))


def _require(condition: bool) -> None:
    if not condition:
        raise ValueError("declared discrepancy does not match public evidence")


def verify_deltas(baseline: Books, final: Books, issues: list[TruthItem]) -> None:
    """Measure issue amounts against the pre-injection postings, including the paired FX decomposition."""
    docs = {handle(d): d for d in (*final.events.invoices, *final.events.payments)}
    for issue in issues:
        affected = [docs[h] for h in issue.affected_ids]
        if issue.type == "payment_misallocation":
            pay, = [d for d in affected if isinstance(d, Payment)]
            sale, = [d for d in affected if isinstance(d, Invoice)]

            def allocated(book: Books) -> int:
                return sum(r.credit_minor for r in lines(book, pay) if r.account == "1200" and r.invoice == sale.key)

            delta = allocated(baseline) - allocated(final)
        else:
            doc, = affected
            old, new = lines(baseline, doc), lines(final, doc)
            if issue.type == "duplicate_invoice":
                _require(not old and isinstance(doc, Invoice) and doc.number_source is not None)
                delta = sum(r.credit_minor for r in new if r.account == "4000")
            elif issue.type == "fx_rate_error":
                account = "1010" if isinstance(doc, Payment) else "1200"
                delta = abs(sum(r.debit_minor for r in old if r.account == account)
                            - sum(r.debit_minor for r in new if r.account == account))
            else:
                _require(issue.type == "cutoff_error" and bool(old) and bool(new))
                inside = old[0].timestamp.date() <= baseline.world.end
                _require(all((r.timestamp.date() <= baseline.world.end) != inside for r in new))
                account = "1010" if isinstance(doc, Payment) else "4000"
                delta = sum(r.credit_minor + r.debit_minor for r in old if r.account == account)
        _require(delta > 0 and delta == issue.expected_amount_minor)


def verify(truth: list[dict[str, Any]], documents: list[dict[str, Any]], ledger: list[tuple],
           fx: list[tuple], period_end: str) -> None:
    """Recompute each claimed amount separately from every final evidence set.

    The public calendar additionally establishes that a supplied quote is the
    latest eligible business-day quote. No private amounts enter the calculation.
    """
    docs = {"doc:" + d["doc_id"]: d for d in documents}
    rows = {"ledger:" + r[0]: r for r in ledger}
    quotes = {f"fx:{d}:{s}:{t}": (d, s, t, rate) for d, s, t, rate in fx}
    _require(len(docs) == len(documents) and len(rows) == len(ledger) and len(quotes) == len(fx))
    _require(len({i["issue_id"] for i in truth}) == len(truth))
    for issue in truth:
        kind = issue["type"]
        _require(bool(issue["evidence_sets"]) and issue["currency"] == "KZT")
        for group in issue["evidence_sets"]:
            _require(all(h in docs or h in rows or h in quotes for h in group))
            documents_in_set = [h for h in group if h in docs]
            ledger_refs = [h for h in group if h in rows]
            fx_refs = [h for h in group if h in quotes]
            _require(len(ledger_refs) == 1 and bool(documents_in_set))
            _, timestamp, account, debit, credit, currency, entity, inv, pay = rows[ledger_refs[0]]
            _require(currency == issue["currency"] and all(docs[h]["entity_id"] == entity for h in documents_in_set))
            doc = docs[documents_in_set[0]]
            if kind == "duplicate_invoice":
                _require(len(documents_in_set) == 2 and not fx_refs)
                a, b = sorted(documents_in_set, key=lambda h: datetime.fromisoformat(docs[h]["issued_at"]))
                original, duplicate = docs[a], docs[b]
                _require(original["invoice_id"] != duplicate["invoice_id"] and inv == duplicate["invoice_id"])
                _require(_number(original["invoice_id"]) == _number(inv))
                _require(all(original[k] == duplicate[k] for k in ("currency", "total_minor", "line_items")))
                delay = datetime.fromisoformat(duplicate["issued_at"]) - datetime.fromisoformat(original["issued_at"])
                _require(1 <= delay.days <= 10 and account in ("1200", "4000") and pay is None)
                expected = debit if account == "1200" else credit
                affected = [b]
            elif kind == "payment_misallocation":
                _require(len(documents_in_set) == 2 and not fx_refs)
                payment, invoice = sorted((docs[h] for h in documents_in_set), key=lambda d: "payment_id" not in d)
                _require(pay == payment["payment_id"] and
                         any(r["invoice_id"] == invoice["invoice_id"] for r in payment["invoice_refs"]))
                _require((account == "2150" and inv is None) or
                         (account == "1200" and inv != invoice["invoice_id"] and
                          _number(inv) == _number(invoice["invoice_id"])))
                expected, affected = credit, documents_in_set
            else:
                _require(kind in ("fx_rate_error", "cutoff_error") and len(documents_in_set) == 1)
                payment_doc = "payment_id" in doc
                if payment_doc:
                    _require(pay == doc["payment_id"] and account == "1010")
                else:
                    _require(inv == doc["invoice_id"] and account == ("1200" if kind == "fx_rate_error" else "4000"))
                day = datetime.fromisoformat(doc["timestamp" if payment_doc else "issued_at"]).astimezone(COMPANY_TZ).date().isoformat()
                posted = debit + credit
                expected, affected = posted, documents_in_set
                if kind == "cutoff_error":
                    _require((day <= period_end) != (timestamp[:10] <= period_end))
                if fx_refs or kind == "fx_rate_error":
                    _require(len(fx_refs) == 1)
                    d, src, dst, rate = quotes[fx_refs[0]]
                    _require(src == doc["currency"] and dst == currency and
                             d == max(q[0] for q in fx if q[1] == src and q[0] <= day))
                    amount = doc["amount_minor" if payment_doc else "total_minor"]
                    correct = convert(amount, src, dst, parse_rate(rate))
                    expected = abs(posted - correct) if kind == "fx_rate_error" else correct
            _require(sorted(affected) == issue["affected_ids"] and expected > 0 and
                     type(issue["expected_amount_minor"]) is int and issue["expected_amount_minor"] == expected)

"""A deliberately literal auditor: exact billing numbers and printed local dates (§10)."""

from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import TypeAlias, cast

from fintrace.common.money import convert, parse_rate

from ._pages import PagePolicy
from .base import Record, finding


InvoiceKey: TypeAlias = tuple[str, str]
DuplicateKey: TypeAlias = tuple[str, str, str, int]


def _doc(row: Record) -> str:
    return 'doc:' + cast(str, row['doc_id'])


def _line(row: Record) -> str:
    return 'ledger:' + cast(str, row['row_id'])


def _amount(row: Record) -> int:
    return cast(int, row['debit_minor']) + cast(int, row['credit_minor'])


def _business_key(doc: Record) -> tuple[datetime, str, str, int]:
    if 'invoice_id' in doc:
        return datetime.fromisoformat(doc['issued_at']), doc['entity_id'], doc['invoice_id'], doc['total_minor']
    return datetime.fromisoformat(doc['timestamp']), doc['entity_id'], doc['payment_id'], doc['amount_minor']


@dataclass(frozen=True)
class _Quote:
    day: str
    rate: Decimal
    handle: str


class _Rates:
    def __init__(self, rows: Sequence[Record], base: str) -> None:
        self.base = base
        self._quotes: dict[str, list[_Quote]] = defaultdict(list)
        for row in sorted(rows, key=lambda r: (r['source_currency'], r['date'])):
            if row['target_currency'] == base:
                day, src = row['date'], row['source_currency']
                self._quotes[src].append(_Quote(day, parse_rate(row['rate']), f'fx:{day}:{src}:{base}'))

    def quote(self, currency: str, timestamp: str) -> _Quote | None:
        quotes = self._quotes.get(currency, [])
        # Intentional baseline limitation: local printed date, even at the company month boundary.
        end = bisect_right(quotes, timestamp[:10], key=lambda q: q.day)
        return quotes[end - 1] if end else None

    def base_amount(self, amount: int, currency: str, at: str) -> int | None:
        if currency == self.base:
            return amount
        quote = self.quote(currency, at)
        return None if quote is None else convert(amount, currency, self.base, quote.rate)


class Reconciler(PagePolicy):
    name = 'reconciler'
    tables = ('invoices', 'payments', 'ledger', 'fx')

    def findings(self) -> list[Record]:
        """Test public documents against their postings, ranking by stable business fields."""
        sales = sorted((r for r in self.rows['invoices'] if r['kind'] == 'invoice'), key=_business_key)
        payments = sorted(self.rows['payments'], key=_business_key)
        invoice_rows: dict[InvoiceKey, list[Record]] = defaultdict(list)
        payment_rows: dict[str, list[Record]] = defaultdict(list)
        for row in self.rows['ledger']:
            if row['payment_id']:
                payment_rows[row['payment_id']].append(row)
            elif row['invoice_id']:
                invoice_rows[row['entity_id'], row['invoice_id']].append(row)
        by_number: dict[InvoiceKey, list[Record]] = defaultdict(list)
        originals: dict[DuplicateKey, list[Record]] = defaultdict(list)
        for sale in sales:
            number = sale['entity_id'], sale['invoice_id']
            by_number[number].append(sale)
            # Sorted by instant. A tied earliest issue time gives no unique original.
            earliest = originals[(*number, sale['currency'], sale['total_minor'])]
            if not earliest or datetime.fromisoformat(earliest[0]['issued_at']) == datetime.fromisoformat(sale['issued_at']):
                earliest.append(sale)
        rates = _Rates(self.rows['fx'], self.manifest['base_currency'])
        findings: list[Record] = []
        for sale in sales:
            number = sale['entity_id'], sale['invoice_id']
            rows = invoice_rows[number]
            if len(by_number[number]) > 1:
                # A repeated exact business number needs the posting of the later document.
                # Ledger dates have the company offset; compare instants for document ownership.
                instant = datetime.fromisoformat(sale['issued_at'])
                rows = [r for r in rows if datetime.fromisoformat(r['timestamp']) == instant]
            earliest = originals[(*number, sale['currency'], sale['total_minor'])]
            ar = [r for r in rows if r['account'] == '1200' and r['debit_minor'] > 0]
            if len(earliest) == 1 and sale is not earliest[0] and len(ar) == 1:
                docs = [_doc(earliest[0]), _doc(sale)]
                findings.append(finding('duplicate_invoice', [docs[1]], _amount(ar[0]), rates.base,
                                        [*docs, _line(ar[0])], 'Repeated exact invoice number and amount.'))
            findings.extend(self._dates(sale, rows, rates, payment=False))
        for pay in payments:
            rows = payment_rows[pay['payment_id']]
            findings.extend(self._allocations(pay, rows, by_number, rates))
            findings.extend(self._dates(pay, rows, rates, payment=True))
        return findings[:30]

    def _dates(self, doc: Record, rows: list[Record], rates: _Rates, *, payment: bool) -> Iterator[Record]:
        at = doc['timestamp'] if payment else doc['issued_at']
        handle = _doc(doc)
        fx_account = '1010' if payment else '1200'
        fx_rows = [r for r in rows if r['account'] == fx_account and r['debit_minor'] > 0]
        quote = rates.quote(doc['currency'], at)
        if len(fx_rows) == 1 and quote is not None and doc['currency'] != rates.base:
            amount = doc['amount_minor'] if payment else doc['total_minor']
            expected = convert(amount, doc['currency'], rates.base, quote.rate)
            delta = abs(_amount(fx_rows[0]) - expected)
            if delta:
                yield finding('fx_rate_error', [handle], delta, rates.base,
                              [handle, _line(fx_rows[0]), quote.handle], 'Posting differs from the local-date quote.')
        cut_account = '1010' if payment else '4000'
        cut_rows = [r for r in rows if r['account'] == cut_account]
        if len(cut_rows) == 1:
            row = cut_rows[0]
            start, end = self.manifest['period_start'][:10], self.manifest['period_end'][:10]
            if (start <= at[:10] <= end) != (start <= row['timestamp'][:10] <= end):
                yield finding('cutoff_error', [handle], _amount(row), rates.base,
                              [handle, _line(row)], 'Document and posting dates straddle the reporting period.')

    @staticmethod
    def _allocations(pay: Record, rows: list[Record], by_number: dict[InvoiceKey, list[Record]],
                     rates: _Rates) -> Iterator[Record]:
        intended = {ref['invoice_id'] for ref in pay['invoice_refs']}
        for ref in sorted(pay['invoice_refs'], key=lambda item: item['invoice_id']):
            number = ref['invoice_id']
            sales = by_number.get((pay['entity_id'], number), [])
            if len(sales) != 1:
                continue
            sale = sales[0]
            expected = rates.base_amount(ref['amount_minor'], sale['currency'], sale['issued_at'])
            if expected is None:
                continue
            credits = [r for r in rows if r['account'] == '1200' and r['invoice_id'] == number]
            delta = expected - sum(r['credit_minor'] for r in credits)
            if delta <= 0:
                continue
            suspect = [r for r in rows if r['credit_minor'] == delta and (
                r['account'] == '2150' or (r['account'] == '1200' and r['invoice_id'] not in intended))]
            affected = [_doc(pay), _doc(sale)]
            # A discrepancy with no diversion row remains a hypothesis. Cite the observed AR credit.
            support = suspect or credits
            if len(support) <= 8:
                yield finding('payment_misallocation', affected, delta, rates.base,
                              [*affected, *(_line(r) for r in support)], 'Remittance exceeds the posted AR credit.')

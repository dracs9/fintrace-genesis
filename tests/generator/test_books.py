import unittest
from dataclasses import replace
from datetime import date, datetime
from decimal import Decimal

from numpy.random import Generator, PCG64

from auditgym.books import check_invariants, post
from auditgym.world import Entity, World, build
from auditgym.world.events import Allocation, Events, Invoice, Payment, generate_events
from auditgym.world.fx import Market, Quote
from test_step0 import _preset


class BookTests(unittest.TestCase):
    def setUp(self) -> None:
        issued = datetime.fromisoformat("2026-03-06T12:00:00-04:00")
        paid = datetime.fromisoformat("2026-03-08T20:00:00-04:00")  # Monday in company TZ.
        market = Market((Quote(date(2026, 3, 6), "USD", Decimal("500.000000")),
                         Quote(date(2026, 3, 9), "USD", Decimal("510.000000"))), date(2026, 4, 3))
        self.world = World(date(2026, 3, 1), date(2026, 3, 31), (Entity(0, "USD", "-04:00"),), market)
        self.invoice = Invoice(0, 0, "USD", issued, paid, 1000)
        self.credit = Invoice(1, 0, "USD", paid, paid, 200, 0)
        self.payment = Payment(0, 0, "USD", paid, 810, (Allocation(0, 800),), 10, 20)
        self.events = Events((self.invoice, self.credit), (self.payment,))
        self.books = post(self.world, self.events)

    def test_exact_postings_historical_ar_and_net_cash(self) -> None:
        actual = [(r.account, r.debit_minor, r.credit_minor) for r in self.books.ledger]
        self.assertCountEqual(actual, [
            ("1200", 500000, 0), ("4000", 0, 500000), ("4100", 100000, 0), ("1200", 0, 100000),
            ("1010", 413100, 0), ("6100", 5100, 0), ("1200", 0, 400000), ("2150", 0, 10200),
            ("7100", 0, 8000),
        ])
        quotes = self.world.market.quotes
        market = replace(self.world.market, quotes=(quotes[0], replace(quotes[1], rate=Decimal("490.000000"))))
        loss = post(replace(self.world, market=market), self.events)
        self.assertEqual([r.debit_minor for r in loss.ledger if r.account == "7100"], [8000])

    def test_all_invariants_reject_corruption(self) -> None:
        rows = list(self.books.ledger)
        for label, edits in (
            ("I1", {0: replace(rows[0], debit_minor=499999), 4: replace(rows[4], debit_minor=413101)}),
            ("I2", {0: replace(rows[0], debit_minor=499999)}),
            ("I3", {0: replace(rows[0], debit_minor=499999), 1: replace(rows[1], credit_minor=499999)}),
            ("I5", {2: replace(rows[2], timestamp=self.invoice.issued_at)}),
            ("I6", {0: replace(rows[0], entity=99)}),
        ):
            changed = rows.copy()
            for index in sorted(edits):
                changed[index] = edits[index]
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, label):
                check_invariants(replace(self.books, ledger=tuple(changed)))
        with self.assertRaisesRegex(ValueError, "I4"):
            post(self.world, replace(self.events, payments=(replace(self.payment, amount_minor=830),)))
        with self.assertRaisesRegex(ValueError, "I7"):
            post(replace(self.world, market=replace(self.world.market, quotes=())), self.events)
        with self.assertRaisesRegex(ValueError, "I5"):
            check_invariants(replace(self.books, ledger=tuple(rows + rows)))

    def test_generated_multicurrency_books(self) -> None:
        config = _preset("hard")
        currencies = set()
        for seed in range(32):
            world = build(config, Generator(PCG64(seed)))
            events = generate_events(world, config["n_invoices"], Generator(PCG64(seed + 100)))
            currencies.update(i.currency for i in events.invoices)
            books = post(world, events)
            self.assertTrue(all(row.currency == "KZT" for row in books.ledger))
        self.assertEqual(currencies, {"KZT", "USD", "EUR", "JPY", "KWD"})

    def test_generated_domestic_books(self) -> None:
        config = dict(_preset("medium"), currencies=["KZT"])
        for seed in range(32):
            world = build(config, Generator(PCG64(seed)))
            events = generate_events(world, config["n_invoices"], Generator(PCG64(seed + 100)))
            books = post(world, events)
            self.assertEqual(sum(r.debit_minor for r in books.ledger), sum(r.credit_minor for r in books.ledger))

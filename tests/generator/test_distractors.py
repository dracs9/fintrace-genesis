import unittest
from dataclasses import replace
from datetime import timedelta, timezone
from fractions import Fraction

from numpy.random import Generator, PCG64

from fintrace.books import check_invariants, post
from fintrace.injectors.distractors import apply, prepare
from fintrace.world import Entity, build
from fintrace.world.events import Events, generate_events
from test_step0 import _preset


def stream(seed: int) -> Generator:
    return Generator(PCG64(seed))


class DistractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = dict(_preset("hard"), n_invoices=[40, 40])
        self.world = replace(build(self.config, stream(1)), entities=(
            Entity(0, "JPY", "+09:00"), Entity(1, "KWD", "-04:00"), Entity(2, "KZT", "+05:00"),
        ))

    def test_all_nine_have_verifiable_business_evidence(self) -> None:
        books, cases = prepare(self.world, dict(self.config, distractor_rate=1), stream(2), stream(3), n_issues=0)
        check_invariants(books)
        invoices = {i.key: i for i in books.events.invoices}
        payments = {p.key: p for p in books.events.payments}
        kinds = {c.kind for c in cases}
        self.assertEqual(kinds, {"recurring_subscription", "split_payment", "multi_invoice_payment",
                                 "partial_open", "overpayment", "bank_fee", "credit_note",
                                 "boundary_correct", "exotic_exponent"})
        covered = {key for c in cases for key in c.documents}
        self.assertEqual(len(covered), len(invoices) + len(payments))
        for c in cases:
            with self.subTest(kind=c.kind):
                inv = [invoices[k] for table, k in c.documents if table == "invoice"]
                pay = [payments[k] for table, k in c.documents if table == "payment"]
                ar = sum(r.debit_minor - r.credit_minor for r in books.ledger
                         if r.account == "1200" and r.invoice == inv[0].key)
                if c.kind == "recurring_subscription":
                    a, b = inv
                    self.assertEqual((a.entity, a.currency, a.total_minor), (b.entity, b.currency, b.total_minor))
                    self.assertNotEqual(a.key, b.key)
                    self.assertNotEqual(a.service_period, b.service_period)
                    for i in inv:
                        self.assertRegex(i.service_period, r"^\d{4}-\d{2}$")
                elif c.kind == "split_payment":
                    self.assertEqual(len(pay), 2)
                    self.assertEqual(sum(r.amount_minor for p in pay for r in p.refs), inv[0].total_minor)
                    self.assertEqual(ar, 0)
                elif c.kind == "multi_invoice_payment":
                    self.assertEqual({r.invoice for r in pay[0].refs}, {i.key for i in inv})
                    self.assertGreater(len(inv), 1)
                elif c.kind == "partial_open":
                    self.assertLess(sum(r.amount_minor for p in pay for r in p.refs), inv[0].total_minor)
                    self.assertGreater(ar, 0)
                elif c.kind in ("overpayment", "bank_fee"):
                    p = pay[0]
                    account = "2150" if c.kind == "overpayment" else "6100"
                    row, = [r for r in books.ledger if r.account == account and r.payment == p.key]
                    alloc = sum(r.amount_minor for r in p.refs)
                    if c.kind == "overpayment":
                        self.assertGreater(p.amount_minor, alloc)
                        self.assertGreater(row.credit_minor, 0)
                    else:
                        self.assertEqual(p.amount_minor, alloc - p.fee_minor)
                        self.assertGreater(row.debit_minor, 0)
                elif c.kind == "credit_note":
                    self.assertEqual(inv[1].original, inv[0].key)
                    self.assertTrue(any(r.account == "4100" and r.invoice == inv[0].key for r in books.ledger))
                elif c.kind == "boundary_correct":
                    issued = inv[0].issued_at
                    at = issued.astimezone(timezone(timedelta(hours=5)))
                    self.assertNotEqual(issued.month, at.month)
                    self.assertTrue(all(r.timestamp.isoformat() == at.isoformat() for r in books.ledger
                                        if r.document == ("invoice", inv[0].key)))
                elif c.kind == "exotic_exponent":
                    self.assertEqual({i.currency for i in inv}, {"JPY", "KWD"})

    def test_rates_replay_and_clean_override(self) -> None:
        events = generate_events(self.world, [40, 40], stream(2), ordinary_only=True)
        original = post(self.world, events)
        self.assertEqual(apply(original, stream(3), distractor_rate=0, n_issues=1), (original, ()))
        for rate, issues in (("0", 0), ("0.1", 1), ("0.4", 0), ("1", 12)):
            result = apply(original, stream(11), distractor_rate=rate, n_issues=issues)
            self.assertEqual(result, apply(original, stream(11), distractor_rate=rate, n_issues=issues))
            books, cases = result
            self.assertTrue(cases)
            self.assertEqual(original.events, events)
            covered = len({k for c in cases for k in c.documents})
            target = Fraction(rate) * (len(books.events.invoices) + len(books.events.payments))
            self.assertGreaterEqual(covered, target)
            self.assertLessEqual(covered - target, 3)
            check_invariants(books)

    def test_domestic_empty_clean_book_and_presets(self) -> None:
        domestic = build(dict(self.config, currencies=["KZT"], tz_spread=0), stream(9))
        books, cases = apply(post(domestic, Events((), ())), stream(7), distractor_rate=0, n_issues=0)
        self.assertEqual(len(cases), 1)
        self.assertNotIn(cases[0].kind, ("boundary_correct", "exotic_exponent"))
        check_invariants(books)
        for name in ("easy", "medium", "hard"):
            config = _preset(name)
            for seed in range(10):
                world = build(config, stream(seed))
                books, cases = prepare(world, config, stream(seed + 100), stream(seed + 200), n_issues=0)
                self.assertTrue(cases)
                check_invariants(books)

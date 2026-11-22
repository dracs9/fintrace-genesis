import unittest
from collections import Counter
from datetime import date, datetime, timedelta
from decimal import Decimal

from numpy.random import Generator, PCG64

from auditgym.books import Books, post
from auditgym.common.money import convert
from auditgym.injectors import BUILTINS, apply, registry
from auditgym.injectors.base import COMPANY_TZ, handle, lines, row_handle
from auditgym.injectors.distractors import prepare
from auditgym.world import Entity, World, build
from test_step0 import _preset
from auditgym.world.events import Allocation, Events, Invoice, Payment
from auditgym.world.fx import Market, Quote


def rng(seed: int) -> Generator:
    return Generator(PCG64(seed))


class InjectorTests(unittest.TestCase):
    def setUp(self) -> None:
        dates = (date(2026, 3, 26), date(2026, 3, 27), date(2026, 3, 30),
                 date(2026, 3, 31), date(2026, 4, 1))
        quotes = tuple(Quote(d, "USD", Decimal(str(490 + 5 * i))) for i, d in enumerate(dates))
        self.world = World(date(2026, 3, 1), date(2026, 3, 31), (Entity(0, "USD", "-04:00"),),
                           Market(quotes, date(2026, 4, 3)))
        at = datetime.fromisoformat("2026-03-27T20:00:00-04:00")
        self.sale = Invoice(0, 0, "USD", at, at + timedelta(days=30), 100000, service_period="2026-03")
        sale2 = Invoice(1, 0, "USD", at + timedelta(days=3), at + timedelta(days=30), 200000)
        p = Payment(0, 0, "USD", at + timedelta(days=4), 149900,
                    (Allocation(0, 50000), Allocation(1, 100000)), 100)
        single = Payment(1, 0, "USD", at + timedelta(days=4), 31000, (Allocation(0, 31000),))
        self.books = post(self.world, Events((self.sale, sale2), (p, single)))
        self.plugins = registry(BUILTINS)

    def test_duplicate_identity_and_two_independent_evidence_sets(self) -> None:
        plugin = self.plugins["duplicate_invoice"]
        c = plugin.eligible(self.books)[0]
        styles = set()
        for seed in range(16):
            m = plugin.apply(c, self.books, rng(seed))
            new = m.books.events.invoices[-1]
            self.assertTrue(1 <= (new.issued_at - c.document.issued_at).days <= 10)
            self.assertLessEqual(new.issued_at.astimezone(COMPANY_TZ).date(), self.world.end)
            self.assertEqual((new.entity, new.currency, new.total_minor, new.service_period),
                             (0, "USD", 100000, "2026-03"))
            number = new.number({0: "INV-1042"})
            styles.add(number)
            t, = m.truth_items
            rows = lines(m.books, new)
            self.assertEqual(t.affected_ids, (handle(new),))
            self.assertEqual(t.evidence_sets, tuple((handle(self.sale), handle(new), row_handle(new, i))
                                                    for i in range(2)))
            self.assertEqual(rows[0].debit_minor, t.expected_amount_minor)
            self.assertEqual(rows[1].credit_minor, t.expected_amount_minor)
            self.assertEqual(len(m.diff["added"]), 3)
        self.assertEqual(styles, {"INV1042", "inv-1042", "INV-01042"})

    def test_all_diversions_preserve_remittance_and_balance(self) -> None:
        plugin = self.plugins["payment_misallocation"]
        candidates = plugin.eligible(self.books)
        self.assertEqual({c.variant for c in candidates}, {"full", "partial"})
        self.assertEqual(len(candidates), 2)
        for c in candidates:
            self.assertEqual(c.document.key, 1)
            m = plugin.apply(c, self.books, rng(3))
            t, = m.truth_items
            rows = lines(m.books, c.document)
            old = lines(self.books, c.document)[c.line]
            i, = [i for i, r in enumerate(rows) if r.account == "2150"]
            lost = old.credit_minor - sum(r.credit_minor for r in rows if r.invoice == old.invoice)
            self.assertEqual(lost, rows[i].credit_minor)
            self.assertEqual(lost, t.expected_amount_minor)
            self.assertEqual(lost == old.credit_minor, c.variant == "full")
            docs = (handle(c.document), f"doc:invoice/{old.invoice}")
            self.assertEqual(t.evidence_sets, ((*docs, row_handle(c.document, i)),))
            self.assertEqual(t.affected_ids, tuple(sorted(docs)))
            self.assertEqual(self.books.events, m.books.events)
            self.assertEqual(sum(r.debit_minor - r.credit_minor for r in rows), 0)

    def test_fx_evidence_recomputes_each_variant(self) -> None:
        plugin = self.plugins["fx_rate_error"]
        variants = set()
        for c in plugin.eligible(self.books):
            variants.add(c.variant)
            m = plugin.apply(c, self.books, rng(5))
            t, = m.truth_items
            doc = c.document
            invoice = isinstance(doc, Invoice)
            at = doc.issued_at if invoice else doc.timestamp
            amount = doc.total_minor if invoice else doc.amount_minor
            quote = self.world.market.lookup("USD", at.astimezone(COMPANY_TZ).date())
            rows = lines(m.books, doc)
            i, = [i for i, r in enumerate(rows) if r.account == ("1200" if invoice else "1010")]
            self.assertEqual(t.evidence_sets, ((handle(doc), row_handle(doc, i), f"fx:{quote.date}:USD:KZT"),))
            self.assertEqual(t.affected_ids, (handle(doc),))
            self.assertEqual(t.expected_amount_minor, abs(rows[i].debit_minor - convert(amount, "USD", "KZT", quote.rate)))
            self.assertGreater(t.expected_amount_minor, 0)
            self.assertEqual(sum(r.debit_minor - r.credit_minor for r in rows), 0)
            if not invoice:
                self.assertEqual([r for r in rows if r.account == "1200"],
                                 [r for r in lines(self.books, doc) if r.account == "1200"])
        self.assertEqual(variants, {"client_date", "previous_day", "next_day", "inverse"})

    def test_cutoff_both_directions(self) -> None:
        plugin = self.plugins["cutoff_error"]
        candidates = plugin.eligible(self.books)
        self.assertEqual({c.variant for c in candidates}, {"into_period", "into_tail"})
        for c in candidates:
            m = plugin.apply(c, self.books, rng(0))
            self.assertEqual(self.books.events, m.books.events)
            for old, new in zip(lines(self.books, c.document), lines(m.books, c.document), strict=True):
                self.assertNotEqual(old.timestamp.month, new.timestamp.month)
                self.assertEqual((old.account, old.debit_minor, old.credit_minor),
                                 (new.account, new.debit_minor, new.credit_minor))
            t, = m.truth_items
            row = lines(m.books, c.document)[c.line]
            self.assertEqual(t.expected_amount_minor, row.credit_minor + row.debit_minor)
            self.assertEqual(t.evidence_sets, ((handle(c.document), row_handle(c.document, c.line)),))

    def test_registry_switches_streams_and_replay(self) -> None:
        calls = Counter()

        def stream(name: str) -> Generator:
            calls[name] += 1
            return rng(19)

        self.assertEqual(apply([], self.books, stream, config={"enabled_types": []}), (self.books, ()))
        for kind in sorted(self.plugins):
            config = {"enabled_types": [kind]}
            a = apply([kind], self.books, stream, config=config)
            self.assertEqual(a, apply([kind], self.books, stream, config=config))
            self.assertEqual(calls["inj." + kind], 2)
            with self.assertRaises(ValueError):
                apply([kind], self.books, stream, config={"enabled_types": []})
        with self.assertRaises(ValueError):
            registry((BUILTINS[0], BUILTINS[0]))

    def test_composed_plan_keeps_all_evidence_and_exact_diff(self) -> None:
        config = _preset("hard")
        plan = ["duplicate_invoice", "payment_misallocation", "fx_rate_error", "cutoff_error"]
        for seed in range(12):
            world = build(config, rng(seed))
            books, _ = prepare(world, config, rng(seed + 100), rng(seed + 200), n_issues=len(plan))
            final, mutations = apply(plan, books, lambda name: rng(seed + plan.index(name[4:])), config=config)

            def records(b: Books) -> dict[str, object]:
                result = {}
                for doc in (*b.events.invoices, *b.events.payments):
                    result[handle(doc)] = doc
                    for i, row in enumerate(lines(b, doc)):
                        result[row_handle(doc, i)] = row
                return result

            before = records(books)
            final_records = records(final)
            for n, mutation in enumerate(mutations, 1):
                after = records(mutation.books)
                diff = mutation.diff
                self.assertEqual(set(diff["added"]), after.keys() - before.keys())
                self.assertEqual(set(diff["removed"]), before.keys() - after.keys())
                self.assertEqual(set(diff["modified"]),
                                 {k for k in sorted(before.keys() & after.keys()) if before[k] != after[k]})
                truth, = mutation.truth_items
                self.assertEqual(truth.issue_id, f"iss_{n:02d}")
                for group in truth.evidence_sets:
                    for ref in group:
                        if ref.startswith("fx:"):
                            _, day, src, dst = ref.split(":")
                            quote = world.market.lookup(src, date.fromisoformat(day))
                            self.assertEqual((quote.date.isoformat(), dst), (day, "KZT"))
                        else:
                            self.assertEqual(after[ref], final_records[ref])
                before = after

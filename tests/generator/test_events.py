import unittest
from collections import Counter

from numpy.random import Generator, PCG64

from fintrace.world import build
from fintrace.world.events import generate_events
from test_step0 import _preset


class EventTests(unittest.TestCase):
    def test_scenarios_references_and_replay(self) -> None:
        world = build(_preset("medium"), Generator(PCG64(7)))
        events = generate_events(world, (40, 40), Generator(PCG64(8)))
        self.assertEqual(events, generate_events(world, (40, 40), Generator(PCG64(8))))
        sales = {i.key: i for i in events.invoices if i.original is None}
        self.assertEqual(len(sales), 40)
        self.assertTrue(any(i.original in sales for i in events.invoices))
        self.assertTrue(any(len(p.refs) == 2 for p in events.payments))
        self.assertTrue(any(p.fee_minor for p in events.payments))
        self.assertTrue(any(p.unapplied_minor for p in events.payments))
        counts = Counter(r.invoice for p in events.payments for r in p.refs)
        self.assertIn(2, counts.values())
        for p in events.payments:
            self.assertEqual(p.amount_minor, sum(r.amount_minor for r in p.refs)
                             + p.unapplied_minor - p.fee_minor)
            for r in p.refs:
                self.assertEqual((sales[r.invoice].entity, sales[r.invoice].currency),
                                 (p.entity, p.currency))

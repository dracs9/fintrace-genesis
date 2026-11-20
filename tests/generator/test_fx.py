import unittest
from datetime import date, timedelta
from decimal import Decimal

from numpy.random import Generator, PCG64

from auditgym.world.fx import build_market


class MarketTests(unittest.TestCase):
    def test_calendar_and_prices(self) -> None:
        start, end = date(2026, 3, 1), date(2026, 3, 31)
        bases = {"USD": "500", "EUR": "545", "JPY": "3.4", "KWD": "1630"}
        market = build_market(start, end, ["KZT", *sorted(bases)], Generator(PCG64(23)))
        for currency in sorted(bases):
            previous = None
            for offset in range(-10, 34):
                day = start + timedelta(days=offset)
                quote = market.lookup(currency, day)
                if day.weekday() >= 5:
                    self.assertEqual(quote, previous)
                    continue
                self.assertEqual(quote.date, day)
                self.assertEqual(quote.rate.as_tuple().exponent, -6)
                if previous is None:
                    self.assertEqual(quote.rate, Decimal(bases[currency]))
                else:
                    error = previous.rate * Decimal("0.004") - abs(quote.rate - previous.rate)
                    self.assertTrue(0 <= error < Decimal("0.000001"))
                previous = quote
        self.assertTrue(all(q.date.weekday() < 5 for q in market.quotes))
        for day in (start - timedelta(days=11), end + timedelta(days=4)):
            with self.assertRaises(LookupError):
                market.lookup("USD", day)

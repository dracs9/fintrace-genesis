import unittest
from decimal import Decimal, ROUND_UP, localcontext

from auditgym.common.money import convert, exponent, parse_rate


class MoneyTests(unittest.TestCase):
    def test_currency_scale_and_ties(self) -> None:
        for amount, src, dst, rate, expected in (
            (4, "JPY", "KZT", "3.400000", 1360),
            (1, "KWD", "KZT", "1630.000000", 163),
            (1, "EUR", "KWD", "2.75", 28),
            (5, "KZT", "JPY", "10", 0),
            (3, "USD", "KZT", "1.5", 4),
            (5, "USD", "KZT", "1.5", 8),
            (-5, "USD", "KZT", "1.5", -8),
            (0, "KZT", "KZT", "1", 0),
        ):
            with self.subTest(amount=amount, src=src, dst=dst):
                self.assertEqual(convert(amount, src, dst, parse_rate(rate)), expected)

    def test_context_independence_and_rate_precision(self) -> None:
        with localcontext() as ctx:
            ctx.prec = 2
            ctx.rounding = ROUND_UP
            rate = parse_rate("1.500000")
            self.assertEqual(str(rate), "1.500000")
            self.assertEqual(convert(10**50 + 3, "EUR", "USD", rate), 15 * 10**49 + 4)

    def test_invalid_inputs(self) -> None:
        for text in ("NaN", "Infinity", "-1", "0", "invalid"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_rate(text)
        with self.assertRaises(TypeError):
            parse_rate(1)
        with self.assertRaises(TypeError):
            convert(True, "USD", "KZT", Decimal("500"))
        with self.assertRaises(TypeError):
            convert(1, "USD", "KZT", 500)
        with self.assertRaises(ValueError):
            exponent("ZZZ")

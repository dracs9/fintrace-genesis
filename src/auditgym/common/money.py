"""Exact currency conversion for AuditGym documents and postings."""

from decimal import (
    MAX_EMAX, MIN_EMIN, ROUND_HALF_EVEN, Context, Decimal, InvalidOperation, localcontext,
)


def exponent(currency: str) -> int:
    """Return the fixed number of minor-unit decimal places for a supported currency."""
    if currency in ("KZT", "USD", "EUR"):
        return 2
    if currency == "JPY":
        return 0
    if currency == "KWD":
        return 3
    raise ValueError("unsupported currency")


def parse_rate(value: str) -> Decimal:
    """Parse a finite positive decimal quote exactly, preserving its precision."""
    if not isinstance(value, str):
        raise TypeError("rate text required")
    try:
        rate = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("invalid rate") from exc
    if not rate.is_finite() or rate <= 0:
        raise ValueError("invalid rate")
    return rate


def convert(amount_minor: int, src: str, dst: str, rate: Decimal) -> int:
    """Convert signed minor units at a target-per-source rate, rounding once to even.

    The product retains every coefficient digit. Neither precision nor rounding
    depends on the caller's decimal context.
    """
    if type(amount_minor) is not int:
        raise TypeError("integer minor units required")
    if not isinstance(rate, Decimal):
        raise TypeError("decimal rate required")
    if not rate.is_finite() or rate <= 0:
        raise ValueError("invalid rate")
    shift = exponent(dst) - exponent(src)
    amount = Decimal(amount_minor)
    digits = len(amount.as_tuple().digits) + len(rate.as_tuple().digits)
    with localcontext(Context(prec=digits, Emax=MAX_EMAX, Emin=MIN_EMIN)):
        target = (amount * rate).scaleb(shift)
        return int(target.to_integral_value(rounding=ROUND_HALF_EVEN))

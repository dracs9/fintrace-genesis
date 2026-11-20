"""Company, counterparties and market for one accounting period."""

from calendar import monthrange
from dataclasses import dataclass
from datetime import date
from fractions import Fraction
from typing import Any

from numpy.random import Generator

from .fx import Market, build_market


@dataclass(frozen=True)
class Entity:
    index: int
    currency: str
    timezone: str


@dataclass(frozen=True)
class World:
    start: date
    end: date
    entities: tuple[Entity, ...]
    market: Market
    base_currency: str = "KZT"
    timezone: str = "+05:00"
    tail_days: int = 3


def build(config: dict[str, Any], rng: Generator) -> World:
    """Build one calendar month deterministically from common.rng.stream("world").

    Entity indices are internal references until final ID assignment.
    foreign_share and tz_spread control independent probabilities of a
    foreign denomination and a non-company time zone, respectively.
    """
    start = date.fromisoformat(config["period"]["start"])
    end = date.fromisoformat(config["period"]["end"])
    if start.day != 1 or end != start.replace(day=monthrange(start.year, start.month)[1]):
        raise ValueError("period must be one calendar month")
    low, high = config["n_entities"]
    if type(low) is not int or type(high) is not int or not 5 <= low <= high <= 40:
        raise ValueError("invalid entity range")
    currencies = sorted(config["currencies"])
    allowed = ("EUR", "JPY", "KWD", "KZT", "USD")
    if (not currencies or len(currencies) != len(set(currencies))
            or any(c not in allowed for c in currencies)):
        raise ValueError("invalid currencies")
    foreign = [c for c in currencies if c != "KZT"]
    foreign_share = Fraction(str(config["foreign_share"]))
    spread = Fraction(str(config["tz_spread"]))
    if not (0 <= foreign_share <= 1 and 0 <= spread <= 1):
        raise ValueError("invalid probability")
    zones = ("-08:00", "-05:00", "-04:00", "-03:00", "+00:00", "+01:00", "+03:00", "+09:00")
    entities = []
    for index in range(int(rng.integers(low, high + 1))):
        currency = "KZT"
        if foreign and ("KZT" not in currencies or rng.integers(foreign_share.denominator) < foreign_share.numerator):
            currency = foreign[int(rng.integers(len(foreign)))]
        zone = "+05:00"
        if rng.integers(spread.denominator) < spread.numerator:
            zone = zones[int(rng.integers(len(zones)))]
        entities.append(Entity(index, currency, zone))
    return World(start, end, tuple(entities), build_market(start, end, currencies, rng))

"""Plan paired errors as units of two issues, before filling the base issue budget."""

from collections.abc import Callable
from dataclasses import replace
from fractions import Fraction
from typing import Any

from numpy.random import Generator

from ...books import Books
from ...world.events import Payment
from .. import fill
from ..base import Mutation, handle
from . import duplicate_with_payment, fx_partial_boundary


def inject(books: Books, config: dict[str, Any], n_issues: int, stream: Callable[[str], Generator],
           chooser: Generator) -> tuple[Books, tuple[tuple[str, Mutation], ...]]:
    """Sample only enabled pairs with room in the budget; fill unused slots with basic issues."""
    _, final, mutations = inject_with_baseline(books, config, n_issues, stream, chooser)
    return final, mutations


def inject_with_baseline(books: Books, config: dict[str, Any], n_issues: int, stream: Callable[[str], Generator],
                         chooser: Generator) -> tuple[Books, Books, tuple[tuple[str, Mutation], ...]]:
    """Retain the checked clean book, including any prerequisites staged for a pair."""
    if not 0 <= n_issues <= 12:
        raise ValueError("invalid issue budget")
    staged = []
    remaining = n_issues
    scenarios = [("duplicate_with_payment", duplicate_with_payment), ("fx_partial_boundary", fx_partial_boundary)]
    if 2 <= remaining < 4:
        chooser.shuffle(scenarios)  # Both pairs compete for the same two issue slots.
    for name, module in scenarios:
        chance = Fraction(str(config["interactions"].get(name, 0)))
        if not 0 <= chance <= 1:
            raise ValueError("invalid interaction probability")
        if remaining < 2 or not chance or any(t not in config["enabled_types"] for t in module.TYPES):
            continue
        rng = stream("inter." + name)
        if rng.integers(chance.denominator) >= chance.numerator:
            continue
        staged_case = module.stage(books, rng)
        if staged_case is not None:
            books, candidate = staged_case
            staged.append((name, module, candidate, rng))
            remaining -= 2
    baseline = books
    reserved: set[str] = set()
    result = []
    for name, module, candidate, rng in staged:
        mutation = module.apply(candidate, books, rng)
        books = mutation.books
        reserved.add(handle(candidate.document))
        if isinstance(candidate.document, Payment):
            reserved.update(f"doc:invoice/{r.invoice}" for r in candidate.document.refs)
        for item in mutation.truth_items:
            reserved.update(item.affected_ids)
        result.append((name, mutation))
    books, singles = fill(remaining, books, stream, config=config, reserved=reserved, chooser=chooser)
    result.extend((m.truth_items[0].type, m) for m in singles)
    numbered = []
    next_id = 1
    for name, mutation in result:
        items = tuple(replace(t, issue_id=f"iss_{i:02d}") for i, t in enumerate(mutation.truth_items, next_id))
        numbered.append((name, replace(mutation, truth_items=items)))
        next_id += len(items)
    return baseline, books, tuple(numbered)

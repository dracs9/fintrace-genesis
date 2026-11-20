"""Basic issue planner. Interaction scenarios are handled separately."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace

from numpy.random import Generator

from ..books import Books
from ..world.events import Payment
from .base import Candidate, Injector, Mutation, handle
from .cutoff_error import CutoffError
from .duplicate_invoice import DuplicateInvoice
from .fx_rate_error import FxRateError
from .payment_misallocation import PaymentMisallocation
from .registry import register, registry


BUILTINS = (DuplicateInvoice, PaymentMisallocation, FxRateError, CutoffError)


def _dependencies(candidate: Candidate) -> set[str]:
    doc = candidate.document
    result = {handle(doc)}
    if isinstance(doc, Payment):
        result.update(f"doc:invoice/{ref.invoice}" for ref in doc.refs)
    return result


def apply(plan: Sequence[str], books: Books, stream: Callable[[str], Generator], *,
          config: Mapping[str, object],
          classes: Sequence[type[Injector]] = BUILTINS) -> tuple[Books, tuple[Mutation, ...]]:
    """Apply an explicit plan using cached inj.<type> streams and clean candidates.

    Only enabled types may appear. Reserve related documents to keep base issues
    independent. An infeasible plan raises without modifying the input books.
    The empty plan preserves a clean episode.
    """
    return _run(plan, books, stream, config, classes, set(), None)


def fill(count: int, books: Books, stream: Callable[[str], Generator], *, config: Mapping[str, object],
         reserved: set[str], chooser: Generator) -> tuple[Books, tuple[Mutation, ...]]:
    """Fill remaining issue slots with eligible enabled types outside interaction documents."""
    return _run([None] * count, books, stream, config, BUILTINS, reserved, chooser)


def _run(plan: Sequence[str | None], books: Books, stream: Callable[[str], Generator],
         config: Mapping[str, object], classes: Sequence[type[Injector]], reserved: set[str],
         chooser: Generator | None) -> tuple[Books, tuple[Mutation, ...]]:
    plugins = registry(classes)
    enabled = config["enabled_types"]
    if not isinstance(enabled, (tuple, list)) or any(t not in plugins for t in enabled):
        raise ValueError("unknown enabled type")
    if len(plan) > 12 or any(t not in enabled for t in plan if t is not None or chooser is None):
        raise ValueError("invalid injection plan")
    current = books
    used = set(reserved)
    mutations = []
    types = sorted(set(enabled if chooser is not None and plan else plan))
    streams = {t: stream("inj." + t) for t in types}
    pools = {t: plugins[t].eligible(books) for t in types}
    for index, kind in enumerate(plan, 1):
        if kind is None:
            options = [t for t in types if any(used.isdisjoint(_dependencies(c)) for c in pools[t])]
            if not options or chooser is None:
                raise ValueError("insufficient independent candidates")
            kind = options[int(chooser.integers(len(options)))]
        available = [c for c in pools[kind] if used.isdisjoint(_dependencies(c))]
        if not available:
            raise ValueError("insufficient independent candidates")
        # Prefer an observable client/company date discrepancy when available.
        if kind == "fx_rate_error":
            date_docs = {handle(c.document) for c in available if c.variant == "client_date"}
            if date_docs:
                available = [c for c in available if handle(c.document) in date_docs]
        rng = streams[kind]
        cand = available[int(rng.integers(len(available)))]
        mutation = plugins[kind].apply(cand, current, rng)
        truth, = mutation.truth_items
        mutations.append(replace(mutation, truth_items=(replace(truth, issue_id=f"iss_{index:02d}"),)))
        current = mutation.books
        used.update(_dependencies(cand))
    return current, tuple(mutations)

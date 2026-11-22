"""Serialize generated FinTrace books and the original on-disk integration fixture."""

import calendar
import csv
import hashlib
import io
import json
import os
from datetime import date, timedelta
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Sequence

from numpy.random import Generator

from ..books import Books, Row
from ..books.discrepancies import verify, verify_deltas
from ..common.rng import Streams
from ..injectors.base import COMPANY_TZ, handle, key, row_handle
from ..injectors.distractors import prepare
from ..injectors.interactions import inject_with_baseline
from ..world import build


_GENERATOR_VERSION = "0.1.1"
_TZ = "+05:00"
_LEDGER_COLUMNS = (
    "row_id", "timestamp", "account", "debit_minor", "credit_minor",
    "currency", "entity_id", "invoice_id", "payment_id",
)
_FX_COLUMNS = ("date", "source_currency", "target_currency", "rate")


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _csv_bytes(columns: Sequence[str], rows: Iterable[Sequence[object]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _policy() -> dict[str, Any]:
    return {
        "fx_rate_date": "document_date_in_company_tz",
        "fx_weekend_rule": "last_available_on_or_before",
        "fx_direction": "rate = units of target per 1 unit of source",
        "period_boundary_tz": _TZ,
        "tail_days": 3,
        "ledger_currency": "base",
        "finding_amount_currency": "base",
        "duplicate_rule": (
            "the later-issued of two invoices with same entity, normalized number "
            "and amount is the duplicate"
        ),
        "invoice_number_normalization": (
            "uppercase, remove spaces/dashes/underscores, strip leading zeros of numeric part"
        ),
        "rounding": "ROUND_HALF_EVEN",
        "amount_tolerance_minor": {"fx_rate_error": 1, "default": 0},
    }


def _fixture_tables(
    start: date, currencies: Sequence[str]
) -> tuple[dict[str, bytes], dict[str, int]]:
    # Literal example IDs keep the fixture usable before the shared ID stream exists.
    issued = start + timedelta(days={5: 2, 6: 1}.get(start.weekday(), 0))
    issued_at = f"{issued}T10:00:00{_TZ}"
    paid_at = f"{issued + timedelta(days=1)}T11:00:00{_TZ}"
    entity_id, invoice_id, payment_id = "E-19ad", "INV-1042", "P-3x7a"
    amount = 100_000
    paid = amount // 2
    invoice = {
        "doc_id": "D-7q2x9a", "invoice_id": invoice_id, "entity_id": entity_id,
        "issued_at": issued_at, "due_at": f"{issued + timedelta(days=14)}T23:59:59{_TZ}",
        "currency": "KZT", "total_minor": amount, "kind": "invoice",
        "original_invoice_id": None,
        "line_items": [{
            "description": "Услуги за месяц", "quantity": 1,
            "unit_price_minor": amount, "service_period": issued.strftime("%Y-%m"),
        }],
    }
    payment = {
        "doc_id": "D-k1p0zz", "payment_id": payment_id, "entity_id": entity_id,
        "timestamp": paid_at, "currency": "KZT", "amount_minor": paid, "fee_minor": 0,
        "invoice_refs": [{"invoice_id": invoice_id, "amount_minor": paid}],
    }
    ledger = [
        ("R-8k2m1q", issued_at, "1200", amount, 0, "KZT", entity_id, invoice_id, None),
        ("R-c3z8aa", issued_at, "4000", 0, amount, "KZT", entity_id, invoice_id, None),
        ("R-4p8x1t", paid_at, "1010", paid, 0, "KZT", entity_id, None, payment_id),
        ("R-d8y1bb", paid_at, "1200", 0, paid, "KZT", entity_id, invoice_id, payment_id),
    ]
    quotes = {
        "USD": "500.000000", "EUR": "545.000000", "JPY": "3.400000", "KWD": "1630.000000",
    }
    fx = [
        (issued.isoformat(), source, "KZT", Decimal(quotes[source]))
        for source in sorted(currencies) if source != "KZT"
    ]
    return {
        "ledger.csv": _csv_bytes(_LEDGER_COLUMNS, ledger),
        "invoices.jsonl": _json_bytes(invoice) + b"\n",
        "payments.jsonl": _json_bytes(payment) + b"\n",
        "fx.csv": _csv_bytes(_FX_COLUMNS, fx),
    }, {"ledger": len(ledger), "invoices": 1, "payments": 1, "fx": len(fx)}


def generate_fixture(seed: int, config: dict[str, Any], out_dir: Path) -> Path:
    """Write a deterministic §4 fixture to out_dir/{open,private} and return out_dir.

    In step 0 every preset yields the same balanced KZT sale and partial settlement.
    The calendar month and currencies follow the config. The episode identity
    includes the seed, canonical config digest, generator version and audit salt.
    No random streams or currency conversions are needed for these fixed records.
    """
    start = date.fromisoformat(config["period"]["start"])
    end = date.fromisoformat(config["period"]["end"])
    if start.day != 1 or end != start.replace(day=calendar.monthrange(start.year, start.month)[1]):
        raise ValueError("period must be one calendar month")

    config_hash = hashlib.sha256(_json_bytes(config)).hexdigest()
    salt = os.environ.get("FINTRACE_SALT", "")
    identity = f"{seed}{config_hash}{_GENERATOR_VERSION}{salt}".encode("utf-8")
    episode_id = "ep_" + hashlib.sha256(identity).hexdigest()[:12]
    tables, counts = _fixture_tables(start, config["currencies"])
    manifest = {
        "schema_version": "1.0", "config_version": config["config_version"],
        "generator_version": _GENERATOR_VERSION, "episode_id": episode_id,
        "base_currency": "KZT",
        "currency_exponents": {"KZT": 2, "USD": 2, "EUR": 2, "JPY": 0, "KWD": 3},
        "period_start": f"{start}T00:00:00{_TZ}", "period_end": f"{end}T23:59:59.999{_TZ}",
        "timezone": _TZ, "counts": counts, "rounding": "ROUND_HALF_EVEN",
    }
    provenance = {
        "seed": seed, "config": config, "config_hash": config_hash,
        "generator_version": _GENERATOR_VERSION, "stream_keys": [],
        "mutations": [],
        "distractors": [{"kind": "partial_open", "ids": ["doc:D-7q2x9a", "doc:D-k1p0zz"]}],
    }
    files = {f"open/{name}": tables[name] for name in sorted(tables)}
    files["open/policy.json"] = _json_bytes(_policy())
    files["open/manifest.json"] = _json_bytes(manifest)
    files["private/truth.json"] = _json_bytes({"episode_id": episode_id, "issues": []})
    files["private/provenance.json"] = _json_bytes(provenance)

    return _publish(files, out_dir)


def _publish(files: dict[str, bytes], out_dir: Path) -> Path:
    # Publish a new episode in one rename, after both visibility domains exist.
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out_dir.parent.resolve()) as staging:
        ready = Path(staging)
        (ready / "open").mkdir()
        (ready / "private").mkdir()
        for name in sorted(files):
            (ready / name).write_bytes(files[name])
        try:
            # Windows requires the target of a directory rename to be absent.
            if out_dir.is_dir() and not any(out_dir.iterdir()):
                out_dir.rmdir()
            ready.rename(out_dir)
        except OSError:
            # A retry (or simultaneous publisher) can reuse the same complete episode.
            present = sorted(p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*"))
            if present != sorted(["open", "private", *sorted(files)]):
                raise
            if any((out_dir / name).read_bytes() != files[name] for name in sorted(files)):
                raise
    return out_dir


def _ids(count: int, prefix: str, width: int, rng: Generator, *, numeric: bool = False) -> list[str]:
    alphabet = "0123456789" if numeric else "abcdefghijklmnopqrstuvwxyz0123456789"
    space = len(alphabet) ** width
    if count > space:
        raise ValueError("identifier space exhausted")
    seen: set[int] = set()
    result = []
    while len(result) < count:
        number = int(rng.integers(space))
        if number in seen:
            continue
        seen.add(number)
        digits = []
        for _ in range(width):
            number, digit = divmod(number, len(alphabet))
            digits.append(alphabet[digit])
        result.append(prefix + "".join(reversed(digits)))
    return result


def generate_episode(seed: int, config: dict[str, Any], out_dir: Path) -> Path:
    """Generate deterministic public records and private, ID-resolved truth and provenance."""
    version = "1.0.1"
    config_hash = hashlib.sha256(_json_bytes(config)).hexdigest()
    salt = os.environ.get("FINTRACE_SALT", "")
    source = Streams(seed, config_hash, version, salt)
    used: dict[str, Generator] = {}

    def stream(name: str) -> Generator:
        if name not in used:
            used[name] = source.stream(name)
        return used[name]

    world = build(config, stream("world"))
    chooser = stream("events")
    chance = Fraction(str(config["clean_probability"]))
    lo, hi = config["n_issues"]
    if not (0 <= chance <= 1 and 0 <= lo <= hi <= 12):
        raise ValueError("invalid issue configuration")
    count = 0 if chooser.integers(chance.denominator) < chance.numerator else int(chooser.integers(lo, hi + 1))
    # One slot per issue also covers the two clean prerequisites of the paired FX case.
    clean, distractors = prepare(world, config, chooser, stream("distract"), n_issues=count,
                                 document_limit=200 - count)
    baseline, books, mutations = inject_with_baseline(clean, config, count, stream, chooser)
    verify_deltas(baseline, books, [t for _, mutation in mutations for t in mutation.truth_items])
    invoices, payments = books.events.invoices, books.events.payments
    if not (20 <= len(invoices) + len(payments) <= 200 and 200 <= len(books.ledger) <= 2000):
        raise ValueError("generated episode size outside specification")
    docs = (*invoices, *payments)
    ids = stream("ids")
    doc_ids = dict(zip((handle(d) for d in docs), _ids(len(docs), "D-", 6, ids), strict=True))
    entities = dict(zip((e.index for e in world.entities), _ids(len(world.entities), "E-", 4, ids), strict=True))
    numbers = dict(zip((i.key for i in invoices), _ids(len(invoices), "INV-", 4, ids, numeric=True), strict=True))
    invoice_ids = {i.key: i.number(numbers) for i in invoices}
    payment_ids = dict(zip((p.key for p in payments), _ids(len(payments), "P-", 4, ids), strict=True))
    invoice_by_key = {i.key: i for i in invoices}
    sales, receipts = [], []
    for i in invoices:
        original = invoice_by_key[i.number_source] if i.number_source is not None else i
        period = original.service_period or original.issued_at.astimezone(COMPANY_TZ).strftime("%Y-%m")
        sales.append({"doc_id": doc_ids[handle(i)], "invoice_id": invoice_ids[i.key], "entity_id": entities[i.entity],
                      "issued_at": i.issued_at.isoformat(), "due_at": i.due_at.isoformat(), "total_minor": i.total_minor,
                      "currency": i.currency, "kind": "invoice" if i.original is None else "credit_note",
                      "original_invoice_id": invoice_ids.get(i.original),
                      "line_items": [{"description": "Услуги за месяц", "quantity": 1,
                                      "unit_price_minor": i.total_minor, "service_period": period}]})
    for p in payments:
        receipts.append({"doc_id": doc_ids[handle(p)], "payment_id": payment_ids[p.key], "entity_id": entities[p.entity],
                         "timestamp": p.timestamp.isoformat(), "amount_minor": p.amount_minor, "fee_minor": p.fee_minor,
                         "currency": p.currency, "invoice_refs": [{"invoice_id": invoice_ids[r.invoice],
                                                                  "amount_minor": r.amount_minor} for r in p.refs]})
    lookup = {key(d): d for d in docs}
    def indexed(book: Books) -> dict[str, Row]:
        offsets: dict[tuple[str, int], int] = {}
        result = {}
        for row in book.ledger:
            index = offsets.get(row.document, 0)
            result[row_handle(lookup[row.document], index)] = row
            offsets[row.document] = index + 1
        return result

    before, after = indexed(baseline), indexed(books)
    historical = before.keys() | after.keys()
    actual = {"added": after.keys() - before.keys(), "removed": before.keys() - after.keys(),
              "modified": {h for h in sorted(before.keys() & after.keys()) if before[h] != after[h]}}
    actual["added"].update(handle(d) for d in invoices if d.number_source is not None)
    for action in sorted(actual):
        declared = [ref for _, m in mutations for ref in m.diff[action]]
        if len(declared) != len(actual[action]) or set(declared) != actual[action]:
            raise ValueError("mutation diff does not match the clean book")
    row_ids = dict(zip(sorted(historical), _ids(len(historical), "R-", 6, ids), strict=True))
    remap = {k: "doc:" + doc_ids[k] for k in sorted(doc_ids)}
    remap.update({k: "ledger:" + row_ids[k] for k in sorted(row_ids)})
    def public_row(h: str, r: Row) -> tuple:
        return (row_ids[h], r.timestamp.isoformat(), r.account, r.debit_minor, r.credit_minor, r.currency,
                entities[r.entity], invoice_ids.get(r.invoice), payment_ids.get(r.payment))

    ledger = [public_row(h, after[h]) for h in sorted(after)]
    fx = [(q.date.isoformat(), q.source_currency, world.base_currency, str(q.rate)) for q in world.market.quotes]
    for day, src, dst, _ in fx:
        h = f"fx:{day}:{src}:{dst}"
        remap[h] = h
    shuffle = stream("shuffle")
    for table in (sales, receipts, ledger, fx):
        shuffle.shuffle(table)
    ledger.sort(key=lambda row: (row[1], row[0]))
    truth, history = [], []
    for name, m in mutations:
        for t in m.truth_items:
            truth.append({"issue_id": t.issue_id, "type": t.type, "currency": t.currency,
                          "affected_ids": sorted(remap[h] for h in t.affected_ids),
                          "evidence_sets": [[remap[h] for h in group] for group in t.evidence_sets],
                          "expected_amount_minor": t.expected_amount_minor})
        history.append({"issue_ids": [t.issue_id for t in m.truth_items], "injector": name, "causal_note": m.causal_note,
                        "decomposition": m.decomposition,
                        "diff": {**{k: sorted(remap[h] for h in m.diff[k]) for k in sorted(m.diff)},
                                 "before": {remap[h]: dict(zip(_LEDGER_COLUMNS, public_row(h, before[h]), strict=True))
                                            for h in sorted(m.diff["modified"] + m.diff["removed"])}}})
    verify(truth, sales + receipts, ledger, fx, world.end.isoformat())
    episode_id = "ep_" + hashlib.sha256(f"{seed}{config_hash}{version}{salt}".encode()).hexdigest()[:12]
    manifest = {"schema_version": "1.0", "generator_version": version, "config_version": config["config_version"],
                "episode_id": episode_id, "base_currency": world.base_currency, "timezone": world.timezone,
                "period_start": f"{world.start}T00:00:00{_TZ}", "period_end": f"{world.end}T23:59:59.999{_TZ}",
                "currency_exponents": {"EUR": 2, "JPY": 0, "KWD": 3, "KZT": 2, "USD": 2},
                "rounding": "ROUND_HALF_EVEN",
                "counts": {"ledger": len(ledger), "invoices": len(sales), "payments": len(receipts), "fx": len(fx)}}
    provenance = {"seed": seed, "config": config, "config_hash": config_hash, "generator_version": version,
                  "stream_keys": sorted(used), "mutations": history,
                  "distractors": [{"kind": d.kind, "ids": [remap[handle(lookup[k])] for k in d.documents]}
                                  for d in distractors]}
    return _publish({"open/manifest.json": _json_bytes(manifest), "open/policy.json": _json_bytes(_policy()),
                     "open/fx.csv": _csv_bytes(_FX_COLUMNS, fx), "open/ledger.csv": _csv_bytes(_LEDGER_COLUMNS, ledger),
                     "open/invoices.jsonl": b"".join(_json_bytes(i) + b"\n" for i in sales),
                     "open/payments.jsonl": b"".join(_json_bytes(p) + b"\n" for p in receipts),
                     "private/truth.json": _json_bytes({"episode_id": episode_id, "issues": truth}),
                     "private/provenance.json": _json_bytes(provenance)}, out_dir)

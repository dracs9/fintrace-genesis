"""Step 0 integration fixture for the on-disk AuditGym episode contract."""

import calendar
import csv
import hashlib
import io
import json
import os
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Sequence


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


def generate_episode(seed: int, config: dict[str, Any], out_dir: Path) -> Path:
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
    salt = os.environ.get("AUDITGYM_SALT", "")
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

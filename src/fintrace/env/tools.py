"""Read-only tools over an episode's ``open/`` data (SPEC §8.3).

``EpisodeData.load`` reads only ``open/`` and builds id indexes once. Tools never mutate
the data and return fresh JSON-safe copies (the caller round-trips the observation).
Argument validation uses the action JSON Schema; failures become ``BAD_ARGS``.
Deterministic: output order is file order.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

from jsonschema import Draft202012Validator

from . import calculator
from .errors import EnvError
from .pagination import page

TABLES = ("ledger", "invoices", "payments", "fx")
TOOLS = ("list_records", "get_record", "get_document", "calculator", "submit")
LEDGER_COLUMNS = (
    "row_id", "timestamp", "account", "debit_minor", "credit_minor",
    "currency", "entity_id", "invoice_id", "payment_id",
)
FX_COLUMNS = ("date", "source_currency", "target_currency", "rate")
_INT_COLUMNS = frozenset({"debit_minor", "credit_minor"})

_PACKAGED_SCHEMAS = Path(__file__).resolve().parent / "_schemas"
_REPO_SCHEMAS = Path(__file__).resolve().parents[3] / "schemas"


@lru_cache(maxsize=None)
def load_schema(name: str) -> Mapping[str, Any]:
    """Load a JSON Schema: ``schemas/<name>`` if it declares every tool, else the packaged draft."""
    repo = _REPO_SCHEMAS / name
    if repo.is_file():
        schema = json.loads(repo.read_text(encoding="utf-8"))
        if name != "action.schema.json" or all(t in schema.get("$defs", {}) for t in TOOLS):
            return schema
    return json.loads((_PACKAGED_SCHEMAS / name).read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _validator(definition: str | None) -> Draft202012Validator:
    schema = dict(load_schema("action.schema.json"))
    if definition is not None:
        schema = {"$schema": schema.get("$schema"), "$defs": schema["$defs"],
                  "$ref": f"#/$defs/{definition}"}
    return Draft202012Validator(schema)


def validate_action(action: object) -> tuple[str, dict[str, Any]]:
    """Return ``(tool, args)``; malformed envelope → BAD_ARGS, unknown tool → UNKNOWN_TOOL."""
    if not isinstance(action, dict) or not _validator(None).is_valid(action):
        raise EnvError("BAD_ARGS")
    tool = action["tool"]
    if tool not in TOOLS:
        raise EnvError("UNKNOWN_TOOL")
    args = action.get("args", {})
    if not _validator(tool).is_valid(args):
        raise EnvError("BAD_ARGS")
    return tool, args


def _read_csv(path: Path, columns: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
    text = path.read_text(encoding="utf-8")
    reader = csv.reader(io.StringIO(text, newline=""))
    header = next(reader)
    if tuple(header) != columns:
        raise ValueError("unexpected csv header")
    rows = []
    for values in reader:
        row: dict[str, Any] = {}
        for column, value in zip(columns, values, strict=True):
            if value == "":
                row[column] = None
            elif column in _INT_COLUMNS:
                row[column] = int(value)
            else:
                row[column] = value
        rows.append(MappingProxyType(row))
    return tuple(rows)


def _read_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return tuple(MappingProxyType(json.loads(line)) for line in lines if line.strip())


def _index(rows: tuple[Mapping[str, Any], ...], key: Callable[[Mapping[str, Any]], str]) -> Mapping[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        k = key(row)
        if k in index:
            raise ValueError("duplicate record id")
        index[k] = row
    return MappingProxyType(index)


def fx_key(row: Mapping[str, Any]) -> str:
    """Return the ``YYYY-MM-DD:SRC:DST`` id of an fx row."""
    return f"{row['date']}:{row['source_currency']}:{row['target_currency']}"


@dataclass(frozen=True)
class EpisodeData:
    """Immutable in-memory view of ``open/``. Holds no path to ``private/``."""

    episode_id: str
    manifest: Mapping[str, Any]
    policy: Mapping[str, Any]
    tables: Mapping[str, tuple[Mapping[str, Any], ...]]
    indexes: Mapping[str, Mapping[str, Mapping[str, Any]]]

    @classmethod
    def load(cls, open_dir: Path) -> "EpisodeData":
        """Read the six public files of an episode and build id indexes once."""
        manifest = json.loads((open_dir / "manifest.json").read_text(encoding="utf-8"))
        policy = json.loads((open_dir / "policy.json").read_text(encoding="utf-8"))
        tables = {
            "ledger": _read_csv(open_dir / "ledger.csv", LEDGER_COLUMNS),
            "invoices": _read_jsonl(open_dir / "invoices.jsonl"),
            "payments": _read_jsonl(open_dir / "payments.jsonl"),
            "fx": _read_csv(open_dir / "fx.csv", FX_COLUMNS),
        }
        indexes = {
            "ledger": _index(tables["ledger"], lambda r: r["row_id"]),
            "invoices": _index(tables["invoices"], lambda r: r["doc_id"]),
            "payments": _index(tables["payments"], lambda r: r["doc_id"]),
            "fx": _index(tables["fx"], fx_key),
        }
        if set(indexes["invoices"]) & set(indexes["payments"]):
            raise ValueError("doc_id shared between tables")
        return cls(manifest["episode_id"], MappingProxyType(manifest), MappingProxyType(policy),
                   MappingProxyType(tables), MappingProxyType(indexes))

    def counts(self) -> dict[str, int]:
        """Return row counts per table in SPEC order."""
        return {t: len(self.tables[t]) for t in TABLES}


def list_records(data: EpisodeData, args: dict[str, Any]) -> dict[str, Any]:
    """Page through a table in file order."""
    table = args["table"]
    return page(data.tables[table], table, args.get("cursor"), args.get("limit"), data.episode_id)


def get_record(data: EpisodeData, args: dict[str, Any]) -> dict[str, Any]:
    """Fetch one record by its table-specific id; missing → UNKNOWN_ID."""
    record = data.indexes[args["table"]].get(args["id"])
    if record is None:
        raise EnvError("UNKNOWN_ID")
    return {"record": record}


def get_document(data: EpisodeData, args: dict[str, Any]) -> dict[str, Any]:
    """Fetch an invoice or payment by ``doc_id``; missing → UNKNOWN_ID."""
    doc_id = args["doc_id"]
    for table in ("invoices", "payments"):
        record = data.indexes[table].get(doc_id)
        if record is not None:
            return {"document": record, "table": table}
    raise EnvError("UNKNOWN_ID")


def calculate(data: EpisodeData, args: dict[str, Any]) -> dict[str, Any]:
    """Evaluate an arithmetic expression with Decimal (see ``calculator``)."""
    return {"value": calculator.evaluate(args["expression"])}


READ_TOOLS: Mapping[str, Callable[[EpisodeData, dict[str, Any]], dict[str, Any]]] = MappingProxyType({
    "list_records": list_records,
    "get_record": get_record,
    "get_document": get_document,
    "calculator": calculate,
})

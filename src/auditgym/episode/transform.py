"""Offline, deterministic row permutation and evidence-ID renaming (SPEC §13)."""

import csv
import json
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from auditgym.common.rng import Streams


_TABLES = ("ledger", "invoices", "payments", "fx")
_COLUMNS = {
    "ledger": ("row_id", "timestamp", "account", "debit_minor", "credit_minor",
               "currency", "entity_id", "invoice_id", "payment_id"),
    "fx": ("date", "source_currency", "target_currency", "rate"),
}
_ID_TOKEN = re.compile(r"(?<![A-Za-z0-9_])[DR]-[a-z0-9]{6}(?![A-Za-z0-9_])")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _rows(path: Path, table: str) -> list[dict[str, Any]]:
    if table in _COLUMNS:
        with (path / f"{table}.csv").open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != _COLUMNS[table]:
                raise ValueError("unexpected table header")
            return list(reader)
    with (path / f"{table}.jsonl").open(encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def _write_rows(path: Path, table: str, rows: list[dict[str, Any]]) -> None:
    if table in _COLUMNS:
        with (path / f"{table}.csv").open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=_COLUMNS[table], lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    else:
        with (path / f"{table}.jsonl").open("w", encoding="utf-8", newline="\n") as target:
            target.writelines(_canonical(row) + "\n" for row in rows)


def _id_map(tables: dict[str, list[dict[str, Any]]], history: dict[str, Any],
            episode_id: str, salt: str) -> dict[str, str]:
    identifiers = [row["row_id"] for row in tables["ledger"]]
    identifiers.extend(row["doc_id"] for name in ("invoices", "payments") for row in tables[name])
    used = set(identifiers)
    if len(used) != len(identifiers):
        raise ValueError("repeated public identifier")
    # A deleted posting exists only in the private preimage but must still roll back.
    for mutation in history["mutations"]:
        for ref in mutation["diff"]["removed"]:
            kind, _, raw = ref.partition(":")
            if kind in ("doc", "ledger"):
                used.add(raw)
    rng = Streams(0, episode_id, "transform-1", salt).stream("ids")
    mapping: dict[str, str] = {}
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    for old in sorted(used):
        while True:
            digits = rng.integers(len(alphabet), size=6)
            new = old[:2] + "".join(alphabet[int(digit)] for digit in digits)
            if new not in used:
                break
        used.add(new)
        mapping[old] = new
    return mapping


def _rename(text: str, mapping: dict[str, str]) -> str:
    """Rewrite complete raw-ID tokens, preserving table qualifiers and ordinary text."""
    return _ID_TOKEN.sub(lambda match: mapping.get(match[0], match[0]), text)


def _references(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        return _rename(value, mapping)
    if isinstance(value, list):
        return [_references(item, mapping) for item in value]
    if isinstance(value, dict):
        # The keys of diff.before are evidence handles in their own right.
        return {_rename(key, mapping): _references(value[key], mapping) for key in sorted(value)}
    return value


def transform(src: Path, dst: Path, *, permute_seed: int, id_salt: str) -> dict[str, str]:
    """Publish a transformed episode and return its raw row_id/doc_id bijection.

    The map includes deleted historical records. Changing only permute_seed changes
    row order independently of the map. Business identifiers, FX keys, the episode
    identity and the order of alternative evidence sets keep their meaning.
    """
    src, dst = src.resolve(), dst.resolve()
    if src == dst or src in dst.parents or dst in src.parents:
        raise ValueError("episode directories must be disjoint")
    manifest = json.loads((src / "open/manifest.json").read_bytes())
    policy = json.loads((src / "open/policy.json").read_bytes())
    truth = json.loads((src / "private/truth.json").read_bytes())
    history = json.loads((src / "private/provenance.json").read_bytes())
    tables = {name: _rows(src / "open", name) for name in _TABLES}
    mapping = _id_map(tables, history, manifest["episode_id"], id_salt)
    for issue in truth["issues"]:
        issue["affected_ids"] = sorted(_references(issue["affected_ids"], mapping))
        issue["evidence_sets"] = _references(issue["evidence_sets"], mapping)
    history["mutations"] = _references(history["mutations"], mapping)
    history["distractors"] = _references(history["distractors"], mapping)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dst.parent) as scratch:
        ready = Path(scratch) / "episode"
        public, private = ready / "open", ready / "private"
        public.mkdir(parents=True)
        private.mkdir(mode=0o700)
        streams = Streams(permute_seed, manifest["episode_id"], "transform-1", "")
        for table in _TABLES:
            rows = tables[table]
            if table != "fx":
                field = "row_id" if table == "ledger" else "doc_id"
                for row in rows:
                    row[field] = mapping[row[field]]
            streams.stream(f"shuffle.{table}").shuffle(rows)
            _write_rows(public, table, rows)
        for path, value in ((public / "manifest.json", manifest), (public / "policy.json", policy),
                            (private / "truth.json", truth), (private / "provenance.json", history)):
            path.write_text(_canonical(value), encoding="utf-8")
        # A Docker operator transforms as root. Preserve the grader's access
        # when the resulting episode is opened later under separate Unix UIDs.
        owner = (src / "private").stat()
        for secret in sorted(private.iterdir()):
            secret.chmod(0o400)
            if hasattr(os, "geteuid") and os.geteuid() == 0:
                os.chown(secret, owner.st_uid, owner.st_gid)
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            os.chown(private, owner.st_uid, owner.st_gid)
        try:
            ready.rename(dst)
        except OSError:
            # An identical retry can reuse the published directory. A conflicting
            # or incomplete destination stays intact, including under concurrent calls.
            names = sorted(p.relative_to(ready) for p in ready.rglob("*"))
            if sorted(p.relative_to(dst) for p in dst.rglob("*")) != names:
                raise
            if any((ready / name).read_bytes() != (dst / name).read_bytes()
                   for name in names if (ready / name).is_file()):
                raise
    return mapping

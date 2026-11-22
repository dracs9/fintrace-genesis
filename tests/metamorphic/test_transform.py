"""Lossless table transformation, historical references and atomic publication."""

import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from auditgym.episode.transform import transform


_FIXTURE = Path(__file__).resolve().parents[1] / "env/fixtures/ep_tiny"


def _files(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file()}


def _records(root: Path, name: str) -> list[dict[str, Any]]:
    with (root / "open" / name).open(encoding="utf-8", newline="") as source:
        if name.endswith(".csv"):
            return list(csv.DictReader(source))
        return [json.loads(line) for line in source]


def test_bijection_covers_open_and_deleted_records(tmp_path: Path) -> None:
    src = tmp_path / "src"
    shutil.copytree(_FIXTURE, src)
    history_path = src / "private/provenance.json"
    history = json.loads(history_path.read_bytes())
    mutation = history["mutations"][0]
    diff = mutation["diff"]
    removed = "R-gone01"
    row = dict(_records(src, "ledger.csv")[0], row_id=removed)
    diff["removed"] = [f"ledger:{removed}"]
    diff["before"] = {f"ledger:{removed}": row}
    mutation["causal_note"] = f"Removed ledger:{removed}. Keep R-gone010 and INV-1042."
    history["distractors"] = [{"kind": "partial_open", "ids": ["doc:D-7q2x9a"]}]
    history_path.write_text(json.dumps(history), encoding="utf-8")
    initial = _files(src)

    dst = tmp_path / "dst"
    mapping = transform(src, dst, permute_seed=7, id_salt="first")
    assert _files(src) == initial
    assert (dst / "private").stat().st_mode & 0o077 == 0
    for path in sorted((dst / "private").iterdir()):
        assert path.stat().st_mode & 0o777 == 0o400
    public_ids = {r["row_id"] for r in _records(src, "ledger.csv")}
    public_ids.update(r["doc_id"] for name in ("invoices.jsonl", "payments.jsonl")
                      for r in _records(src, name))
    assert set(mapping) == public_ids | {removed}
    targets = {mapping[key] for key in sorted(mapping)}
    assert len(targets) == len(mapping)
    assert targets.isdisjoint(mapping)
    for name in sorted(mapping):
        assert mapping[name][:2] == name[:2]
        assert re.fullmatch(r"[DR]-[a-z0-9]{6}", mapping[name])

    for table, field in (("ledger.csv", "row_id"), ("invoices.jsonl", "doc_id"),
                         ("payments.jsonl", "doc_id")):
        expected = [dict(row, **{field: mapping[row[field]]}) for row in _records(src, table)]
        actual = _records(dst, table)
        assert sorted(expected, key=lambda r: r[field]) == sorted(actual, key=lambda r: r[field])
    assert sorted(_records(src, "fx.csv"), key=str) == sorted(_records(dst, "fx.csv"), key=str)
    copied = json.loads((dst / "private/provenance.json").read_bytes())
    copied_diff = copied["mutations"][0]["diff"]
    remapped = f"ledger:{mapping[removed]}"
    assert copied_diff["before"] == {remapped: dict(row, row_id=mapping[removed])}
    assert copied_diff["removed"] == [remapped]
    assert copied["mutations"][0]["causal_note"] == f"Removed {remapped}. Keep R-gone010 and INV-1042."
    assert copied["distractors"][0]["ids"] == [f"doc:{mapping['D-7q2x9a']}"]
    assert copied["config"] == history["config"]
    assert copied["config_hash"] == history["config_hash"]

    old_issue = json.loads((src / "private/truth.json").read_bytes())["issues"][0]
    new_issue = json.loads((dst / "private/truth.json").read_bytes())["issues"][0]
    for old, new in zip(old_issue["evidence_sets"], new_issue["evidence_sets"], strict=True):
        assert [prefix + ":" + mapping[raw] for prefix, raw in (ref.split(":") for ref in old)] == new


def test_id_salt_is_independent_of_table_permutation(tmp_path: Path) -> None:
    first, replay, permuted, renamed = (tmp_path / name for name in ("a", "b", "c", "d"))
    mapping = transform(_FIXTURE, first, permute_seed=42, id_salt="first")
    assert transform(_FIXTURE, first, permute_seed=42, id_salt="first") == mapping
    assert transform(_FIXTURE, replay, permute_seed=42, id_salt="first") == mapping
    assert _files(first) == _files(replay)
    assert transform(_FIXTURE, permuted, permute_seed=43, id_salt="first") == mapping
    assert _records(first, "ledger.csv") != _records(permuted, "ledger.csv")
    other = transform(_FIXTURE, renamed, permute_seed=42, id_salt="second")
    assert mapping != other
    unmap = {mapping[k]: other[k] for k in sorted(mapping)}
    assert [unmap[r["row_id"]] for r in _records(first, "ledger.csv")] == [
        r["row_id"] for r in _records(renamed, "ledger.csv")
    ]


def test_conflicting_destination_and_failed_read_preserve_files(tmp_path: Path) -> None:
    dst = tmp_path / "transformed"
    transform(_FIXTURE, dst, permute_seed=0, id_salt="one")
    before = _files(dst)
    with pytest.raises(OSError):
        transform(_FIXTURE, dst, permute_seed=0, id_salt="two")
    assert _files(dst) == before
    with pytest.raises(ValueError, match="disjoint"):
        transform(dst, dst / "other", permute_seed=0, id_salt="one")
    assert _files(dst) == before
    (dst / "private/truth.json").unlink()
    with pytest.raises(OSError):
        transform(dst, tmp_path / "missing", permute_seed=0, id_salt="two")
    assert not (tmp_path / "missing").exists()

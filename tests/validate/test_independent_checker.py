"""Black-box checks of the independent validator on genuine serialized books."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from auditgym.episode.serialize import generate_episode
from auditgym.validate.independent_checker import InvalidEpisode, validate_data, validate_episode

ROOT = Path(__file__).resolve().parents[2]
Record = dict[str, Any]


def preset(name: str) -> Record:
    """Load the repository preset used by the public generator."""
    return yaml.safe_load((ROOT / "configs" / "difficulty" / f"{name}.yaml").read_text(encoding="utf-8"))


def read(path: Path) -> Record:
    """Read a test JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, data: Any) -> None:
    """Rewrite a test JSON file deterministically after a single corruption."""
    path.write_text(json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")


def table(path: Path) -> list[Record]:
    """Read one public table in its native disk representation."""
    with path.open(encoding="utf-8", newline="") as stream:
        if path.suffix == ".csv":
            return list(csv.DictReader(stream))
        return [json.loads(line) for line in stream]


def write_table(path: Path, rows: list[Record]) -> None:
    """Preserve the test table's existing column order and UTF-8 encoding."""
    if path.suffix == ".csv":
        with path.open(encoding="utf-8") as stream:
            cols = next(csv.reader(stream))
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=cols, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    else:
        path.write_text("".join(json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
                                for r in rows), encoding="utf-8")


@pytest.fixture
def episode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fixed hard episode containing both interactions, nonzero fees and all four issue types."""
    monkeypatch.setenv("AUDITGYM_SALT", "")
    config = preset("hard")
    config.update(n_issues=[10, 10], clean_probability=0,
                  interactions={"duplicate_with_payment": 1, "fx_partial_boundary": 1})
    return generate_episode(0, config, tmp_path / "episode")


@pytest.fixture
def clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build an untouched book with real FX, bank fees, returns and historical allocations."""
    monkeypatch.setenv("AUDITGYM_SALT", "")
    config = preset("hard")
    config.update(n_issues=[0, 0], clean_probability=1)
    return generate_episode(0, config, tmp_path / "clean")


@pytest.mark.parametrize(("file", "reason"), [
    ("open/manifest.json", "schema"), ("open/policy.json", "schema"), ("open/invoices.jsonl", "schema"),
    ("open/payments.jsonl", "schema"), ("open/ledger.csv", "schema"), ("open/fx.csv", "schema"),
    ("private/truth.json", "schema"), ("private/provenance.json", "schema"),
])
def test_all_file_schemas(episode: Path, file: str, reason: str) -> None:
    path = episode / file
    if path.suffix == ".json":
        data = read(path)
        data["undeclared"] = 1
        dump(path, data)
    else:
        rows = table(path)
        if path.suffix == ".jsonl":
            rows[0]["amount_minor" if "amount_minor" in rows[0] else "total_minor"] = True
        else:
            rows[0]["rate" if file.endswith("fx.csv") else "debit_minor"] = "1e2"
        write_table(path, rows)
    with pytest.raises(InvalidEpisode, match=reason):
        validate_episode(episode)


def test_counts_and_size(episode: Path) -> None:
    path = episode / "open/manifest.json"
    data = read(path)
    data["counts"]["ledger"] = 10
    dump(path, data)
    with pytest.raises(InvalidEpisode, match="count"):
        validate_episode(episode)
    ledger = episode / "open/ledger.csv"
    write_table(ledger, table(ledger)[:10])
    with pytest.raises(InvalidEpisode, match="size"):
        validate_episode(episode)


@pytest.mark.parametrize("kind", ["duplicate_invoice", "payment_misallocation", "fx_rate_error", "cutoff_error"])
def test_recalculate_each_amount(episode: Path, kind: str) -> None:
    path = episode / "private/truth.json"
    truth = read(path)
    next(item for item in truth["issues"] if item["type"] == kind)["expected_amount_minor"] += 1
    dump(path, truth)
    with pytest.raises(InvalidEpisode, match="amount mismatch"):
        validate_episode(episode)


def test_reject_unrelated_alternative_evidence(episode: Path) -> None:
    path = episode / "private/truth.json"
    truth = read(path)
    duplicate = next(i for i in truth["issues"] if i["type"] == "duplicate_invoice")
    second = duplicate["evidence_sets"][1]
    new_row = next(r for r in table(episode / "open/ledger.csv") if r["account"] == "6100")
    second[second.index(next(k for k in second if k.startswith("ledger:")))] = "ledger:" + new_row["row_id"]
    dump(path, truth)
    with pytest.raises(InvalidEpisode, match="duplicate posting"):
        validate_episode(episode)


def test_ia_requires_quote_in_its_own_cutoff_proof(episode: Path) -> None:
    path = episode / "private/truth.json"
    truth = read(path)
    item = next(i for i in truth["issues"] if i["type"] == "cutoff_error" and
                any(k.startswith("fx:") for k in i["evidence_sets"][0]))
    item["evidence_sets"][0] = [k for k in item["evidence_sets"][0] if not k.startswith("fx:")]
    dump(path, truth)
    with pytest.raises(InvalidEpisode, match="cutoff quote coverage"):
        validate_episode(episode)


def test_t2_unapplied_cash_requires_unique_intended_invoice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUDITGYM_SALT", "")
    config = preset("medium")
    config.update(n_issues=[1, 1], clean_probability=0, enabled_types=["payment_misallocation"],
                  interactions={"duplicate_with_payment": 0, "fx_partial_boundary": 0})
    episode = generate_episode(90_321, config, tmp_path / "episode")
    public = episode / "open"
    ledger = {"ledger:" + row["row_id"]: row for row in table(public / "ledger.csv")}
    truth = read(episode / "private/truth.json")
    issue = next(i for i in truth["issues"] if i["type"] == "payment_misallocation" and
                 any(ledger[k]["account"] == "2150" for k in i["evidence_sets"][0] if k in ledger))
    path = public / "payments.jsonl"
    payments = table(path)
    payment = next(p for p in payments if "doc:" + p["doc_id"] in issue["affected_ids"])
    ref = payment["invoice_refs"][0]
    other = next(i for i in table(public / "invoices.jsonl") if i["kind"] == "invoice" and
                 i["entity_id"] == payment["entity_id"] and i["invoice_id"] != ref["invoice_id"])
    ref["amount_minor"] -= 1
    payment["invoice_refs"].append({"invoice_id": other["invoice_id"], "amount_minor": 1})
    write_table(path, payments)
    with pytest.raises(InvalidEpisode, match="ambiguous unapplied cash target"):
        validate_episode(episode)


@pytest.mark.parametrize("key", ["affected_ids", "evidence_sets"])
def test_missing_public_fact(episode: Path, key: str) -> None:
    path = episode / "private/truth.json"
    truth = read(path)
    target = truth["issues"][0][key]
    if key == "evidence_sets":
        target = target[0]
    target[0] = "doc:D-absent"
    dump(path, truth)
    with pytest.raises(InvalidEpisode, match="IDs"):
        validate_episode(episode)


@pytest.mark.parametrize("damage", ["missing", "wrong_key", "bad_post"])
def test_rollback_images(episode: Path, damage: str) -> None:
    path = episode / "private/provenance.json"
    provenance = read(path)
    before = next(m["diff"]["before"] for m in provenance["mutations"] if m["diff"]["modified"])
    key = sorted(before)[0]
    if damage == "missing":
        del before[key]
    elif damage == "wrong_key":
        before[key]["row_id"] = "R-absent"
    else:
        col = "credit_minor" if before[key]["credit_minor"] else "debit_minor"
        before[key][col] += 1
    dump(path, provenance)
    with pytest.raises(InvalidEpisode, match="(before-image|I2)"):
        validate_episode(episode)


def test_i1_balanced_total_cannot_hide_unbalanced_groups(clean: Path) -> None:
    path = clean / "open/ledger.csv"
    rows = table(path)
    receivables = [r for r in rows if r["account"] == "1200" and int(r["debit_minor"]) > 1]
    first, second = receivables[:2]
    first["debit_minor"] = str(int(first["debit_minor"]) - 1)
    second["debit_minor"] = str(int(second["debit_minor"]) + 1)
    write_table(path, rows)
    with pytest.raises(InvalidEpisode, match="I1"):
        validate_episode(clean)


def test_i2_trial_balance(clean: Path) -> None:
    path = clean / "open/ledger.csv"
    rows = table(path)
    col = "credit_minor" if int(rows[0]["credit_minor"]) else "debit_minor"
    rows[0][col] = str(int(rows[0][col]) + 1)
    write_table(path, rows)
    with pytest.raises(InvalidEpisode, match="I2"):
        validate_episode(clean)


def test_i3_balanced_but_wrong_receivable(clean: Path) -> None:
    path = clean / "open/ledger.csv"
    rows = table(path)
    invoice = next(r["invoice_id"] for r in rows if r["account"] == "4000")
    for row in rows:
        if row["invoice_id"] == invoice and not row["payment_id"] and row["account"] in ("4000", "1200"):
            if (row["account"] == "1200" and int(row["debit_minor"])) or row["account"] == "4000":
                col = "debit_minor" if row["account"] == "1200" else "credit_minor"
                row[col] = str(int(row[col]) + 1)
    write_table(path, rows)
    with pytest.raises(InvalidEpisode, match="I3"):
        validate_episode(clean)


def test_i4_net_receipt_with_nonzero_fee(clean: Path) -> None:
    path = clean / "open/payments.jsonl"
    payments = table(path)
    payment = next(p for p in payments if p["fee_minor"] > 0)
    assert payment["amount_minor"] + payment["fee_minor"] == sum(r["amount_minor"] for r in payment["invoice_refs"])
    validate_episode(clean)
    payment["amount_minor"] -= 1
    write_table(path, payments)
    with pytest.raises(InvalidEpisode, match="I4"):
        validate_episode(clean)


@pytest.mark.parametrize("damage", ["timestamp", "twice"])
def test_i5_exact_posting(clean: Path, damage: str) -> None:
    path = clean / "open/ledger.csv"
    rows = table(path)
    selected = next(r for r in rows if r["account"] == "4100")
    if damage == "timestamp":
        selected["timestamp"] = "2026-05-01T01:00:00+05:00"
    else:
        # Introduce a duplicate offsetting pair; group and invoice balances stay unchanged.
        for index, account in enumerate(("6100", "7100")):
            rows.append(dict(selected, account=account, row_id=f"R-tests{index}",
                             debit_minor=str(1 - index), credit_minor=str(index)))
        manifest_path = clean / "open/manifest.json"
        manifest = read(manifest_path)
        manifest["counts"]["ledger"] += 2
        dump(manifest_path, manifest)
    write_table(path, rows)
    with pytest.raises(InvalidEpisode, match="I5"):
        validate_episode(clean)


def test_i6_unknown_reference(clean: Path) -> None:
    path = clean / "open/payments.jsonl"
    payments = table(path)
    known = {i["invoice_id"] for i in table(clean / "open/invoices.jsonl")}
    missing = next(n for i in range(10000) if (n := f"INV-{i:04}") not in known)
    payments[0]["invoice_refs"][0]["invoice_id"] = missing
    write_table(path, payments)
    with pytest.raises(InvalidEpisode, match="I6"):
        validate_episode(clean)


def test_i7_effective_quote_required(clean: Path) -> None:
    path = clean / "open/fx.csv"
    rows = table(path)
    currency = next(d["currency"] for d in table(clean / "open/invoices.jsonl") if d["currency"] != "KZT")
    remaining = [r for r in rows if r["source_currency"] != currency]
    write_table(path, remaining)
    manifest_path = clean / "open/manifest.json"
    manifest = read(manifest_path)
    manifest["counts"]["fx"] = len(remaining)
    dump(manifest_path, manifest)
    with pytest.raises(InvalidEpisode, match="I7"):
        validate_episode(clean)


def test_batch_report_picks_up_partial_episode(episode: Path, tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    episode.rename(data / "original")
    report = tmp_path / "reports/validation.json"
    assert validate_data(data, report) == 0
    assert read(report)["checked"] == 1
    partial = data / "partial"
    shutil.copytree(data / "original/open", partial / "open")
    assert validate_data(data, report) == 1
    output = read(report)
    assert output["checked"] == 2 and not output["valid"]
    assert validate_data(tmp_path / "empty", report) == 1
    assert read(report)["checked"] == 0


@pytest.mark.parametrize("seed", list(range(10_000, 10_010)) + list(range(20_000, 20_010)) + list(range(90_000, 90_100)))
def test_training_dev_and_fresh_episodes(seed: int, tmp_path: Path) -> None:
    """Independently validate every specified seed even when packaged data is absent."""
    config = preset("medium")
    episode = generate_episode(seed, config, tmp_path / "episode")
    validate_episode(episode)


@pytest.mark.parametrize("split", ["train", "dev"])
def test_every_packaged_episode(split: str) -> None:
    """Include existing split directories regardless of their names or generating configuration."""
    directory = ROOT / "data" / split
    if directory.exists():
        for episode in sorted(p for p in directory.iterdir() if p.is_dir()):
            validate_episode(episode)


def test_cutoff_proof_must_cross_the_declared_reporting_end(episode: Path) -> None:
    item = next(i for i in read(episode / "private/truth.json")["issues"]
                if i["type"] == "cutoff_error")
    ids = {k[7:] for k in item["evidence_sets"][0] if k.startswith("ledger:")}
    path = episode / "open/ledger.csv"
    rows = table(path)
    for row in rows:
        if row["row_id"] in ids:
            row["timestamp"] = "2026-05-01T01:00:00+05:00"
    write_table(path, rows)
    with pytest.raises(InvalidEpisode, match="cutoff outside reporting window"):
        validate_episode(episode)


def test_provenance_connects_each_issue_to_its_mutated_posting(episode: Path) -> None:
    path = episode / "private/provenance.json"
    provenance = read(path)
    duplicate = next(m for m in provenance["mutations"] if m["injector"] == "duplicate_with_payment")
    other = next(m for m in provenance["mutations"] if m["injector"] == "fx_partial_boundary")
    duplicate["diff"], other["diff"] = other["diff"], duplicate["diff"]
    dump(path, provenance)
    with pytest.raises(InvalidEpisode, match="cited posting unaffected"):
        validate_episode(episode)

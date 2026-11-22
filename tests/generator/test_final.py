"""On-disk regression tests using only the public API, files, and private rollback data."""

import csv
import json
import os
import subprocess
import sys
import unittest
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fintrace.books.discrepancies import verify
from fintrace.episode.serialize import generate_episode
from test_step0 import _preset, _read_json, _sha256_tree


def _table(folder: Path, name: str) -> list[dict]:
    with (folder / name).open(encoding="utf-8", newline="") as file:
        if name.endswith(".csv"):
            return list(csv.DictReader(file))
        return [json.loads(line) for line in file]


class FinalGeneratorTests(unittest.TestCase):
    def check_disk(self, root: Path) -> tuple[int, int, int]:
        public = root / "open"
        manifest = _read_json(public / "manifest.json")
        invoices = _table(public, "invoices.jsonl")
        payments = _table(public, "payments.jsonl")
        ledger = _table(public, "ledger.csv")
        fx = _table(public, "fx.csv")
        truth = _read_json(root / "private/truth.json")
        provenance = _read_json(root / "private/provenance.json")
        self.assertEqual(truth["episode_id"], manifest["episode_id"])
        self.assertEqual(manifest["counts"], dict(invoices=len(invoices), payments=len(payments),
                                                  ledger=len(ledger), fx=len(fx)))
        docs = {"doc:" + d["doc_id"]: d for d in invoices + payments}
        rows = {"ledger:" + r["row_id"]: r for r in ledger}
        quotes = {f"fx:{q['date']}:{q['source_currency']}:KZT" for q in fx}
        self.assertEqual(len(docs), len(invoices) + len(payments))
        self.assertEqual(len(rows), len(ledger))
        self.assertTrue(20 <= len(docs) <= 200)
        self.assertTrue(200 <= len(rows) <= 2000)
        self.assertTrue(0 <= len(truth["issues"]) <= 12)
        order = [(r["timestamp"], r["row_id"]) for r in ledger]
        self.assertEqual(order, sorted(order))
        ids = docs.keys() | rows.keys() | quotes
        for item in truth["issues"]:
            self.assertTrue(item["affected_ids"] and set(item["affected_ids"]) <= ids)
            self.assertTrue(item["evidence_sets"])
            for evidence in item["evidence_sets"]:
                self.assertTrue(evidence and set(evidence) <= ids)
        for d in provenance["distractors"]:
            self.assertTrue(set(d["ids"]) <= docs.keys())
        if not truth["issues"]:
            self.assertTrue(provenance["distractors"])

        # Roll back the serialized mutations, without relying on the generator's clean snapshot.
        for mutation in reversed(provenance["mutations"]):
            diff = mutation["diff"]
            self.assertEqual(set(diff["before"]), set(diff["modified"] + diff["removed"]))
            for h in diff["added"]:
                (docs if h.startswith("doc:") else rows).pop(h)
            rows.update(diff["before"])
        self.check_clean(manifest, list(docs.values()), list(rows.values()), fx)
        return len(invoices) + len(payments), len(ledger), len(truth["issues"])

    def check_clean(self, manifest: dict, docs: list[dict], ledger: list[dict], fx: list[dict]) -> None:
        zone = timezone(timedelta(hours=5))
        sales = {d["invoice_id"]: d for d in docs if d.get("kind") == "invoice"}
        exps = manifest["currency_exponents"]
        quotes = {(q["source_currency"], date.fromisoformat(q["date"])): Decimal(q["rate"]) for q in fx}
        expected, actual, balances, groups = Counter(), Counter(), Counter(), Counter()
        applied = Counter()
        for doc in docs:
            is_payment = "payment_id" in doc
            at = datetime.fromisoformat(doc["timestamp" if is_payment else "issued_at"]).astimezone(zone)
            src = doc["currency"]

            def base(amount: int, moment: datetime) -> int:
                day = moment.astimezone(zone).date()
                while day.weekday() > 4:
                    day -= timedelta(days=1)
                rate = Decimal("1") if src == "KZT" else quotes[src, day]
                return int((Decimal(amount).scaleb(2 - exps[src]) * rate).to_integral_value(rounding=ROUND_HALF_EVEN))

            def add(account: str, signed: int, invoice: str = "") -> None:
                if signed:
                    expected[at.isoformat(), account, signed, doc["entity_id"], invoice, doc.get("payment_id", "")] += 1

            if is_payment:
                cash, fee = base(doc["amount_minor"], at), base(doc["fee_minor"], at)
                unapplied = doc["amount_minor"] + doc["fee_minor"] - sum(r["amount_minor"] for r in doc["invoice_refs"])
                self.assertGreaterEqual(unapplied, 0)  # I4 in source minor units.
                excess = base(unapplied, at)
                add("1010", cash)
                add("6100", fee)
                add("2150", -excess)
                ar = 0
                for ref in doc["invoice_refs"]:
                    sale = sales[ref["invoice_id"]]
                    self.assertEqual((sale["entity_id"], sale["currency"]), (doc["entity_id"], src))
                    value = base(ref["amount_minor"], datetime.fromisoformat(sale["issued_at"]))
                    add("1200", -value, sale["invoice_id"])
                    applied[sale["invoice_id"]] += ref["amount_minor"]
                    ar += value
                add("7100", ar + excess - cash - fee)
            else:
                sale = sales[doc["original_invoice_id"] or doc["invoice_id"]]
                amount = base(doc["total_minor"], datetime.fromisoformat(sale["issued_at"]))
                self.assertEqual((sale["entity_id"], sale["currency"]), (doc["entity_id"], src))
                self.assertEqual(doc["total_minor"], sum(i["quantity"] * i["unit_price_minor"] for i in doc["line_items"]))
                if doc["kind"] == "invoice":
                    add("1200", amount, sale["invoice_id"])
                    add("4000", -amount, sale["invoice_id"])
                else:
                    add("4100", amount, sale["invoice_id"])
                    add("1200", -amount, sale["invoice_id"])
                    applied[sale["invoice_id"]] += doc["total_minor"]
        for r in ledger:
            debit, credit = int(r["debit_minor"]), int(r["credit_minor"])
            self.assertTrue(min(debit, credit) == 0 < max(debit, credit))
            self.assertEqual(r["currency"], manifest["base_currency"])
            inv, pay = r["invoice_id"] or "", r["payment_id"] or ""
            net = debit - credit
            actual[r["timestamp"], r["account"], net, r["entity_id"], inv, pay] += 1
            groups[pay or inv] += net
            if r["account"] == "1200":
                balances[inv] += net
        self.assertEqual(actual, expected)  # I3–I7: account, amount, period, links, effective FX, exactly once.
        self.assertTrue(all(groups[k] == 0 for k in sorted(groups)))  # I1
        self.assertEqual(sum(groups[k] for k in sorted(groups)), 0)  # I2
        self.assertTrue(all(balances[k] >= 0 and applied[k] <= sales[k]["total_minor"] for k in sorted(sales)))

    def test_1000_seeds(self) -> None:
        presets = [_preset(name) for name in ("easy", "medium", "hard")]
        for index in range(1000):
            seed = 90000 + index
            with self.subTest(seed=seed), TemporaryDirectory() as dest:
                out = generate_episode(seed, presets[index % 3], Path(dest) / "episode")
                self.check_disk(out)

    def test_size_regression_and_sha256(self) -> None:
        cfg = dict(_preset("hard"), n_issues=[4, 4], clean_probability=0,
                   interactions={"fx_partial_boundary": 1, "duplicate_with_payment": 1})
        with TemporaryDirectory() as dest, patch.dict("os.environ", {"FINTRACE_SALT": ""}):
            first = generate_episode(2, cfg, Path(dest) / "first")
            count, _, issues = self.check_disk(first)
            self.assertLessEqual(count, 200)
            self.assertEqual(issues, 4)
            replay = generate_episode(2, {k: cfg[k] for k in sorted(cfg, reverse=True)}, Path(dest) / "replay")
            self.assertEqual(_sha256_tree(first), _sha256_tree(replay))
            with patch.dict("os.environ", {"FINTRACE_SALT": "second-run"}):
                other = generate_episode(2, cfg, Path(dest) / "other")
            for name in ("ledger.csv", "invoices.jsonl", "payments.jsonl"):
                self.assertNotEqual((first / "open" / name).read_bytes(), (other / "open" / name).read_bytes())

    def test_incorrect_amount_is_rejected(self) -> None:
        doc = {"doc_id": "D-pay", "entity_id": "E-test", "payment_id": "P-test", "currency": "KZT",
               "timestamp": "2026-04-01T10:00:00+05:00"}
        ledger = [("R-test", "2026-03-31T10:00:00+05:00", "1010", 123, 0, "KZT", "E-test", None, "P-test")]
        issue = {"issue_id": "iss_01", "type": "cutoff_error", "currency": "KZT", "affected_ids": ["doc:D-pay"],
                 "expected_amount_minor": 124, "evidence_sets": [["doc:D-pay", "ledger:R-test"]]}
        with self.assertRaisesRegex(ValueError, "discrepancy"):
            verify([issue], [doc], ledger, [], "2026-03-31")

    def test_replay_in_separate_interpreters(self) -> None:
        project = Path(__file__).resolve().parents[2]
        code = ("import json,sys; from pathlib import Path; "
                "from fintrace.episode.serialize import generate_episode; "
                "generate_episode(13, json.loads(sys.argv[1]), Path(sys.argv[2]))")
        with TemporaryDirectory() as dest:
            outputs = []
            for hash_seed in ("19", "987"):
                out = Path(dest) / hash_seed
                env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONPATH=str(project / "src"),
                           FINTRACE_SALT="replay", PYTHONDONTWRITEBYTECODE="1")
                subprocess.run([sys.executable, "-c", code, json.dumps(_preset("medium")), str(out)],
                               env=env, check=True)
                outputs.append(_sha256_tree(out))
            self.assertEqual(outputs[0], outputs[1])

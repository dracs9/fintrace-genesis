import csv
import hashlib
import json
import unittest
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path
from tempfile import TemporaryDirectory
from zlib import crc32

from numpy.random import Generator, PCG64, SeedSequence

from auditgym.common.rng import Streams
from auditgym.episode.serialize import generate_episode
from test_step0 import _preset, _read_json, _sha256_tree


def read_table(folder: Path, name: str) -> list[dict]:
    path = folder / name
    with path.open(encoding="utf-8", newline="") as src:
        if path.suffix == ".csv":
            return list(csv.DictReader(src))
        return [json.loads(line) for line in src]


class InteractionTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.config = dict(_preset("hard"), n_issues=[4, 4], clean_probability=0,
                           interactions={"fx_partial_boundary": 1, "duplicate_with_payment": 1})

    def test_independent_named_streams(self) -> None:
        factory = Streams(42, "digest", "0.2.0", "salt")
        name = "inter.fx_partial_boundary"
        entropy = [42, *(crc32(s.encode()) for s in ("digest", "0.2.0", "salt"))]
        key = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")
        expected = Generator(PCG64(SeedSequence(entropy, spawn_key=(key,)))).bytes(32)
        self.assertEqual(factory.stream(name).bytes(32), expected)
        self.assertNotEqual(factory.stream("inter.duplicate_with_payment").bytes(32), expected)
        self.assertEqual(factory.stream(name).bytes(32), expected)

    def test_reconstruct_both_pairs_from_serialized_evidence(self) -> None:
        out = generate_episode(0, self.config, self.root / "first")
        self.assertEqual(_sha256_tree(out), _sha256_tree(generate_episode(0, self.config, self.root / "replay")))
        public = out / "open"
        manifest = _read_json(public / "manifest.json")
        provenance = _read_json(out / "private/provenance.json")
        issues = _read_json(out / "private/truth.json")["issues"]
        self.assertEqual(len({t["issue_id"] for t in issues}), 4)
        docs = {"doc:" + d["doc_id"]: d for f in ("invoices.jsonl", "payments.jsonl") for d in read_table(public, f)}
        ledger = {"ledger:" + r["row_id"]: r for r in read_table(public, "ledger.csv")}
        fx = {f"fx:{r['date']}:{r['source_currency']}:KZT": r for r in read_table(public, "fx.csv")}
        records = {**docs, **ledger, **fx}
        mutations = {m["injector"]: m for m in provenance["mutations"]}
        self.assertEqual(set(mutations), set(self.config["interactions"]))
        for name in sorted(mutations):
            mutation = mutations[name]
            self.assertIn("inter." + name, provenance["stream_keys"])
            self.assertEqual(len(mutation["issue_ids"]), 2)
            for issue_id in mutation["issue_ids"]:
                t, = [i for i in issues if i["issue_id"] == issue_id]
                self.assertTrue(t["affected_ids"])
                self.assertTrue(all(h in records for h in t["affected_ids"]))
                for group in t["evidence_sets"]:
                    self.assertTrue(all(h in records for h in group))
                self.assertIn(t["type"], mutation["decomposition"])
        by_type = {t["type"]: t for t in issues}
        cutoff, rate = by_type["cutoff_error"], by_type["fx_rate_error"]
        self.assertEqual(cutoff["affected_ids"], rate["affected_ids"])
        for t in (cutoff, rate):
            pay_id, cash_id, fx_id = t["evidence_sets"][0]
            pay, cash, quote = docs[pay_id], ledger[cash_id], fx[fx_id]
            self.assertEqual(t["affected_ids"], [pay_id])
            self.assertEqual(cash["payment_id"], pay["payment_id"])
            self.assertEqual(cash["account"], "1010")
            at = datetime.fromisoformat(pay["timestamp"])
            self.assertEqual(at.date(), date(2026, 3, 31))
            self.assertEqual(at.astimezone(timezone(timedelta(hours=5))).date(), date(2026, 4, 1))
            self.assertEqual(date.fromisoformat(cash["timestamp"][:10]), at.date())
            self.assertEqual(quote["date"], "2026-04-01")
            self.assertEqual(quote["source_currency"], pay["currency"])
            ref, = pay["invoice_refs"]
            sale, = [d for d in docs.values() if d.get("invoice_id") == ref["invoice_id"]]
            self.assertLess(ref["amount_minor"], sale["total_minor"])
            exp = manifest["currency_exponents"][pay["currency"]]
            units = Decimal(pay["amount_minor"]).scaleb(2 - exp)
            correct = int((units * Decimal(quote["rate"])).to_integral_value(rounding=ROUND_HALF_EVEN))
            old = fx[f"fx:2026-03-31:{pay['currency']}:KZT"]
            posted = int((units * Decimal(old["rate"])).to_integral_value(rounding=ROUND_HALF_EVEN))
            self.assertEqual(int(cash["debit_minor"]), posted)
            self.assertNotEqual(correct, posted)
            self.assertEqual(t["expected_amount_minor"], correct if t is cutoff else abs(posted - correct))
        dup, mis = by_type["duplicate_invoice"], by_type["payment_misallocation"]
        self.assertEqual(len(dup["evidence_sets"]), 2)
        for original_id, dup_id, row_id in dup["evidence_sets"]:
            original, copy = docs[original_id], docs[dup_id]
            self.assertEqual(copy["line_items"], original["line_items"])
            self.assertNotEqual(copy["invoice_id"], original["invoice_id"])
            self.assertEqual(ledger[row_id]["invoice_id"], copy["invoice_id"])
            self.assertEqual(dup["expected_amount_minor"],
                             int(ledger[row_id]["debit_minor"]) + int(ledger[row_id]["credit_minor"]))
        pay_id, sale_id, credit_id = mis["evidence_sets"][0]
        self.assertEqual(sale_id, original_id)
        credit = ledger[credit_id]
        self.assertEqual(credit["account"], "1200")
        self.assertEqual(credit["invoice_id"], copy["invoice_id"])
        self.assertEqual(credit["payment_id"], docs[pay_id]["payment_id"])
        self.assertIn(original["invoice_id"], [r["invoice_id"] for r in docs[pay_id]["invoice_refs"]])
        self.assertLessEqual(datetime.fromisoformat(copy["issued_at"]), datetime.fromisoformat(docs[pay_id]["timestamp"]))
        self.assertEqual(int(credit["credit_minor"]), mis["expected_amount_minor"])
        self.assertEqual(mis["affected_ids"], sorted([sale_id, pay_id]))
        self.assertEqual(dup["affected_ids"], [dup_id])
        self.assertFalse(any(b"decomposition" in p.read_bytes() for p in sorted(public.iterdir())))

    def test_budget_disabled_types_and_no_fx_difference_on_weekend(self) -> None:
        for label, overrides, expected, pairs in (
            ("clean", {"clean_probability": 1}, 0, 0),
            ("single", {"n_issues": [1, 1]}, 1, 0),
            ("disabled", {"enabled_types": ["duplicate_invoice"]}, 4, 0),
            ("off", {"interactions": {"duplicate_with_payment": 0, "fx_partial_boundary": 0}}, 4, 0),
            ("weekend", {"period": {"start": "2026-07-01", "end": "2026-07-31"}}, 4, 1),
            ("mixed", {"n_issues": [6, 6]}, 6, 2),
        ):
            with self.subTest(label=label):
                out = generate_episode(0, dict(self.config, **overrides), self.root / label)
                provenance = _read_json(out / "private/provenance.json")
                items = _read_json(out / "private/truth.json")["issues"]
                self.assertEqual(len(set(t["issue_id"] for t in items)), expected)
                self.assertEqual(sum(m["decomposition"] is not None for m in provenance["mutations"]), pairs)
                if label == "clean":
                    self.assertTrue(provenance["distractors"])
                if label == "weekend":
                    self.assertNotIn("fx_partial_boundary", [m["injector"] for m in provenance["mutations"]])

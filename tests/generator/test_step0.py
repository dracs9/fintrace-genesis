"""Integration checks for the step 0 generator, runnable with pytest or unittest."""

import csv
import hashlib
import json
import unittest
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import yaml

from auditgym.episode.serialize import generate_episode


_CONFIGS = Path(__file__).resolve().parents[2] / "configs"
_OPEN = ("fx.csv", "invoices.jsonl", "ledger.csv", "manifest.json", "payments.jsonl", "policy.json")
_PRIVATE = ("provenance.json", "truth.json")


def _preset(name: str) -> dict[str, Any]:
    path = _CONFIGS / ("default.yaml" if name == "default" else f"difficulty/{name}.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_bytes())


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


class StepZeroTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.config = _preset("default")

    def test_presets(self) -> None:
        self.assertEqual(self.config, _preset("medium"))
        for name, rate, interaction, max_issues in (
            ("easy", "0.1", "0", 3), ("medium", "0.25", "0.3", 8), ("hard", "0.4", "0.6", 12)
        ):
            with self.subTest(preset=name):
                config = _preset(name)
                self.assertEqual(config["config_version"], f"{name}-1.0")
                self.assertEqual(Decimal(str(config["distractor_rate"])), Decimal(rate))
                for scenario in ("fx_partial_boundary", "duplicate_with_payment"):
                    self.assertEqual(Decimal(str(config["interactions"][scenario])), Decimal(interaction))
                self.assertEqual(config["n_issues"], [0, max_issues])
                self.assertEqual(config["enabled_types"], [
                    "duplicate_invoice", "payment_misallocation", "fx_rate_error", "cutoff_error"
                ])
                generated = generate_episode(42, config, self.root / name)
                manifest = _read_json(generated / "open/manifest.json")
                self.assertEqual(manifest["config_version"], config["config_version"])
                self.assertEqual(manifest["counts"]["fx"], len(config["currencies"]) - 1)

    def test_manifest_policy_and_file_formats(self) -> None:
        episode = generate_episode(42, self.config, self.root / "schema")
        self.assertEqual(episode, self.root / "schema")
        public, private = episode / "open", episode / "private"
        self.assertEqual(sorted(p.name for p in episode.iterdir()), ["open", "private"])
        self.assertEqual(sorted(p.name for p in public.iterdir()), list(_OPEN))
        self.assertEqual(sorted(p.name for p in private.iterdir()), list(_PRIVATE))
        manifest = _read_json(public / "manifest.json")
        self.assertEqual(manifest.keys(), {
            "schema_version", "config_version", "generator_version", "episode_id", "base_currency",
            "currency_exponents", "period_start", "period_end", "timezone", "counts", "rounding",
        })
        self.assertRegex(manifest["episode_id"], r"^ep_[a-f0-9]{12}$")
        self.assertEqual(manifest["schema_version"], "1.0")
        self.assertEqual(manifest["base_currency"], "KZT")
        self.assertEqual(manifest["timezone"], "+05:00")
        self.assertEqual(manifest["currency_exponents"], {"KZT": 2, "USD": 2, "EUR": 2, "JPY": 0, "KWD": 3})
        self.assertEqual(manifest["rounding"], "ROUND_HALF_EVEN")
        policy = _read_json(public / "policy.json")
        self.assertEqual(policy.keys(), {
            "fx_rate_date", "fx_weekend_rule", "fx_direction", "period_boundary_tz", "tail_days",
            "ledger_currency", "finding_amount_currency", "duplicate_rule",
            "invoice_number_normalization", "rounding", "amount_tolerance_minor",
        })
        self.assertEqual(policy["fx_rate_date"], "document_date_in_company_tz")
        self.assertEqual(policy["fx_weekend_rule"], "last_available_on_or_before")
        self.assertEqual(policy["tail_days"], 3)
        self.assertEqual(policy["amount_tolerance_minor"], {"default": 0, "fx_rate_error": 1})
        self.assertEqual(policy["rounding"], manifest["rounding"])

        for path in sorted(episode.rglob("*")):
            if path.is_file():
                raw = path.read_bytes()
                self.assertNotIn(b"\r", raw)
                if path.suffix in (".json", ".jsonl"):
                    for line in raw.decode("utf-8").splitlines():
                        self.assertEqual(line, json.dumps(
                            json.loads(line), sort_keys=True, separators=(",", ":"), ensure_ascii=False
                        ))
                if path.suffix == ".jsonl":
                    self.assertTrue(raw.endswith(b"\n"))

    def test_fx_documents_and_postings(self) -> None:
        episode = generate_episode(42, self.config, self.root / "documents")
        public = episode / "open"
        ledger_path, fx_path = public / "ledger.csv", public / "fx.csv"
        ledger, fx = _rows(ledger_path), _rows(fx_path)
        self.assertEqual(
            ledger_path.read_bytes().splitlines()[0],
            b"row_id,timestamp,account,debit_minor,credit_minor,currency,entity_id,invoice_id,payment_id",
        )
        self.assertEqual(fx_path.read_bytes().splitlines()[0], b"date,source_currency,target_currency,rate")
        manifest = _read_json(public / "manifest.json")
        self.assertEqual(manifest["counts"], {
            "invoices": 1, "payments": 1, "ledger": len(ledger), "fx": len(fx),
        })
        self.assertEqual(len({r["row_id"] for r in ledger}), len(ledger))
        for row in fx:
            self.assertLess(date.fromisoformat(row["date"]).weekday(), 5)
            self.assertEqual(row["target_currency"], "KZT")
            self.assertRegex(row["rate"], r"^\d+\.\d{6}$")
        invoice = _read_json(public / "invoices.jsonl")
        payment = _read_json(public / "payments.jsonl")
        self.assertEqual(invoice.keys(), {
            "doc_id", "invoice_id", "entity_id", "issued_at", "due_at", "currency",
            "total_minor", "line_items", "kind", "original_invoice_id",
        })
        self.assertEqual(payment.keys(), {
            "doc_id", "payment_id", "entity_id", "timestamp", "currency",
            "amount_minor", "fee_minor", "invoice_refs",
        })
        for line in invoice["line_items"]:
            self.assertEqual(line.keys(), {"description", "quantity", "unit_price_minor", "service_period"})
            self.assertIs(type(line["quantity"]), int)
            self.assertIs(type(line["unit_price_minor"]), int)
        self.assertRegex(invoice["doc_id"], r"^D-[a-z0-9]{6}$")
        self.assertRegex(payment["doc_id"], r"^D-[a-z0-9]{6}$")
        self.assertNotEqual(invoice["doc_id"], payment["doc_id"])
        self.assertRegex(invoice["invoice_id"], r"^INV-[0-9]{4}$")
        self.assertRegex(payment["payment_id"], r"^P-[a-z0-9]{4}$")
        self.assertRegex(invoice["entity_id"], r"^E-[a-z0-9]{4}$")
        self.assertEqual(payment["entity_id"], invoice["entity_id"])
        self.assertIsNone(invoice["original_invoice_id"])
        self.assertEqual(invoice["kind"], "invoice")
        self.assertEqual(invoice["currency"], "KZT")
        self.assertEqual(payment["currency"], "KZT")
        self.assertIs(type(invoice["total_minor"]), int)
        self.assertIs(type(payment["amount_minor"]), int)
        self.assertEqual(payment["fee_minor"], 0)
        self.assertTrue(0 < payment["amount_minor"] < invoice["total_minor"])
        self.assertEqual(payment["invoice_refs"], [
            {"invoice_id": invoice["invoice_id"], "amount_minor": payment["amount_minor"]}
        ])
        line_total = sum(i["quantity"] * i["unit_price_minor"] for i in invoice["line_items"])
        self.assertEqual(line_total, invoice["total_minor"])
        for moment in (invoice["issued_at"], invoice["due_at"], payment["timestamp"]):
            self.assertIsNotNone(datetime.fromisoformat(moment).utcoffset())
        postings = []
        for row in ledger:
            self.assertRegex(row["row_id"], r"^R-[a-z0-9]{6}$")
            self.assertEqual(row["entity_id"], invoice["entity_id"])
            self.assertEqual(row["currency"], "KZT")
            postings.append((row["account"], int(row["debit_minor"]), int(row["credit_minor"]),
                             row["invoice_id"], row["payment_id"], row["timestamp"]))
        amount, inv, pay = invoice["total_minor"], invoice["invoice_id"], payment["payment_id"]
        self.assertCountEqual(postings, [
            ("1200", amount, 0, inv, "", invoice["issued_at"]), ("4000", 0, amount, inv, "", invoice["issued_at"]),
            ("1010", payment["amount_minor"], 0, "", pay, payment["timestamp"]),
            ("1200", 0, payment["amount_minor"], inv, pay, payment["timestamp"]),
        ])
    def test_private_contract(self) -> None:
        episode = generate_episode(42, self.config, self.root / "grader")
        private = episode / "private"
        manifest = _read_json(episode / "open/manifest.json")
        self.assertEqual(_read_json(private / "truth.json"), {
            "episode_id": manifest["episode_id"], "issues": [],
        })
        provenance = _read_json(private / "provenance.json")
        self.assertEqual(provenance.keys(), {
            "seed", "config", "config_hash", "generator_version", "stream_keys", "mutations", "distractors",
        })
        self.assertEqual(provenance["seed"], 42)
        self.assertEqual(provenance["config"], self.config)
        self.assertEqual(provenance["generator_version"], manifest["generator_version"])
        self.assertEqual(provenance["mutations"], [])
        docs = [_read_json(episode / "open" / name)["doc_id"] for name in ("invoices.jsonl", "payments.jsonl")]
        self.assertEqual(provenance["distractors"], [{"kind": "partial_open", "ids": ["doc:" + d for d in docs]}])
        self.assertEqual(provenance["stream_keys"], [])

    def test_example_for_team_matches_generator(self) -> None:
        with patch.dict("os.environ", {"AUDITGYM_SALT": ""}):
            fresh = generate_episode(42, self.config, self.root / "fresh")
        example = Path(__file__).parent / "fixtures/step0"
        self.assertEqual(_sha256_tree(fresh), _sha256_tree(example))

    def test_retry_rejects_unexpected_public_file(self) -> None:
        out = generate_episode(42, self.config, self.root / "existing")
        (out / "open/unexpected.json").write_bytes(b"{}")
        with self.assertRaises(OSError):
            generate_episode(42, self.config, out)

    def test_replay_sha256_and_episode_identity(self) -> None:
        first = self.root / "first"
        first.mkdir()
        with patch.dict("os.environ", {"AUDITGYM_SALT": "private-test-salt"}):
            generate_episode(42, self.config, first)
            original = _sha256_tree(first)
            reordered = {k: self.config[k] for k in sorted(self.config, reverse=True)}
            replay = generate_episode(42, reordered, self.root / "replay")
            self.assertEqual(_sha256_tree(replay), original)
            generate_episode(42, self.config, first)
            self.assertEqual(_sha256_tree(first), original)

            manifest = _read_json(first / "open/manifest.json")
            encoded = json.dumps(self.config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            config_hash = hashlib.sha256(encoded.encode()).hexdigest()
            self.assertEqual(config_hash, _read_json(first / "private/provenance.json")["config_hash"])
            identity = ("42" + config_hash + manifest["generator_version"] + "private-test-salt").encode()
            self.assertEqual(manifest["episode_id"], "ep_" + hashlib.sha256(identity).hexdigest()[:12])
            for path in sorted((first / "open").iterdir()):
                for marker in (b'"seed"', b'"issues"', b'"mutations"', b'"config_hash"', b"private-test-salt"):
                    self.assertNotIn(marker, path.read_bytes())
            other_seed = generate_episode(43, self.config, self.root / "other-seed")
            changed = dict(self.config, period={"start": "2024-02-01", "end": "2024-02-29"})
            other_config = generate_episode(42, changed, self.root / "other-config")
            other_manifest = _read_json(other_config / "open/manifest.json")
            self.assertTrue(other_manifest["period_end"].startswith("2024-02-29T"))
            self.assertTrue(_read_json(other_config / "open/invoices.jsonl")["issued_at"].startswith("2024-02-"))
        with patch.dict("os.environ", {"AUDITGYM_SALT": ""}):
            other_salt = generate_episode(42, self.config, self.root / "other-salt")
        for path in (other_seed, other_config, other_salt):
            self.assertNotEqual(_read_json(path / "open/manifest.json")["episode_id"], manifest["episode_id"])

    def test_invalid_period_leaves_no_episode(self) -> None:
        config = dict(self.config, period={"start": "2026-03-01", "end": "2026-03-30"})
        output = self.root / "incomplete"
        with self.assertRaises(ValueError):
            generate_episode(42, config, output)
        self.assertFalse(output.exists())

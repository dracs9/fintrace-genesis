"""Offline independent checks of serialized AuditGym episodes (SPEC §§3–7, 12)."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from auditgym.common.money import convert, parse_rate

Record = dict[str, Any]
Post = tuple[datetime, str, int, str, str | None, str | None]
LEDGER = ("row_id", "timestamp", "account", "debit_minor", "credit_minor", "currency",
          "entity_id", "invoice_id", "payment_id")
FX = ("date", "source_currency", "target_currency", "rate")
TYPES = ("duplicate_invoice", "payment_misallocation", "fx_rate_error", "cutoff_error")
TRAPS = ("recurring_subscription", "split_payment", "multi_invoice_payment", "partial_open",
         "overpayment", "bank_fee", "credit_note", "boundary_correct", "exotic_exponent")
CURRENCIES = ("EUR", "JPY", "KWD", "KZT", "USD")
ZONE = timezone(timedelta(hours=5))


class InvalidEpisode(ValueError):
    """An offline diagnostic for the operator, independent of the agent interface."""


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise InvalidEpisode(reason)


def _fields(row: Any, names: str) -> None:
    _require(isinstance(row, dict) and sorted(row) == sorted(names.split()), "schema: object fields")


def _list(value: Any) -> list[Any]:
    _require(isinstance(value, list), "schema: expected array")
    return value


def _strings(value: Any) -> list[str]:
    items = _list(value)
    _require(all(isinstance(s, str) and s for s in items), "schema: string array")
    _require(len(items) == len(set(items)), "schema: repeated identifier")
    return items


def _integer(value: Any, minimum: int = 0) -> None:
    _require(type(value) is int and value >= minimum, "schema: integer out of range")


def _pattern(value: Any, pattern: str) -> None:
    _require(isinstance(value, str) and re.fullmatch(pattern, value) is not None, "schema: invalid text")


def _stamp(value: Any) -> datetime:
    _pattern(value, r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?[+-]\d{2}:\d{2}")
    return datetime.fromisoformat(value)


def _business_day(moment: datetime) -> date:
    day = moment.astimezone(ZONE).date()
    return day - timedelta(days=max(0, day.weekday() - 4))


def _pairs(items: list[tuple[str, Any]]) -> Record:
    row = dict(items)
    _require(len(row) == len(items), "schema: duplicate JSON key")
    return row


def _json(text: str) -> Any:
    def reject(token: str) -> None:
        raise InvalidEpisode("schema: invalid JSON number")

    return json.loads(text, object_pairs_hook=_pairs, parse_constant=reject)


def _table(public: Path, table: str) -> list[Record]:
    if table in ("invoices", "payments"):
        with (public / f"{table}.jsonl").open(encoding="utf-8") as stream:
            return [_json(line) for line in stream]
    columns = LEDGER if table == "ledger" else FX
    with (public / f"{table}.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        _require(reader.fieldnames == list(columns), "schema: CSV columns")
        records = []
        for row in reader:
            _require(None not in row and all(row[col] is not None for col in columns), "schema: CSV width")
            record: Record = dict(row)
            if table == "ledger":
                for col in ("debit_minor", "credit_minor"):
                    _pattern(record[col], r"(?:0|[1-9][0-9]*)")
                    record[col] = int(record[col])
                for col in ("invoice_id", "payment_id", "entity_id"):
                    record[col] = record[col] or None
            records.append(record)
    return records


def _public_record(table: str, row: Record) -> str:
    if table == "fx":
        _fields(row, " ".join(FX))
        _pattern(row["date"], r"\d{4}-\d{2}-\d{2}")
        _require(date.fromisoformat(row["date"]).weekday() < 5, "schema: weekend FX quote")
        _require(row["target_currency"] == "KZT" and row["source_currency"] in ("EUR", "USD", "JPY", "KWD"),
                 "schema: FX currency pair")
        _pattern(row["rate"], r"[0-9]+\.[0-9]{6}")
        parse_rate(row["rate"])
        return "fx:{date}:{source_currency}:{target_currency}".format_map(row)
    if table == "ledger":
        _fields(row, " ".join(LEDGER))
        _pattern(row["row_id"], r"R-[a-z0-9]{6}")
        _require(row["account"] in ("1010", "1200", "2150", "4000", "4100", "6100", "7100")
                 and row["currency"] == "KZT", "schema: ledger account or currency")
        _require(_stamp(row["timestamp"]).utcoffset() == ZONE.utcoffset(None), "schema: ledger offset")
        for col in ("debit_minor", "credit_minor"):
            _integer(row[col])
        _require((row["debit_minor"] > 0) != (row["credit_minor"] > 0), "schema: one-sided posting")
        for col, pattern in (("payment_id", r"P-[a-z0-9]{4}"), ("entity_id", r"E-[a-z0-9]{4}"),
                             ("invoice_id", r"(?i:inv[-_ ]*0*\d{4})")):
            if row[col] is not None:
                _pattern(row[col], pattern)
        return "ledger:" + row["row_id"]
    if table == "invoices":
        _fields(row, "doc_id invoice_id entity_id issued_at due_at currency total_minor line_items kind original_invoice_id")
        _pattern(row["invoice_id"], r"(?i:inv[-_ ]*0*\d{4})")
        _stamp(row["issued_at"])
        _stamp(row["due_at"])
        _integer(row["total_minor"], 1)
        _require(row["kind"] in ("invoice", "credit_note"), "schema: invoice kind")
        _require((row["kind"] == "invoice") == (row["original_invoice_id"] is None), "schema: original invoice")
        if row["original_invoice_id"] is not None:
            _pattern(row["original_invoice_id"], r"INV-\d{4}")
        total = 0
        for item in _list(row["line_items"]):
            _fields(item, "description quantity unit_price_minor service_period")
            _require(isinstance(item["description"], str), "schema: line description")
            _pattern(item["service_period"], r"\d{4}-(?:0[1-9]|1[0-2])")
            _integer(item["quantity"], 1)
            _integer(item["unit_price_minor"])
            total += item["quantity"] * item["unit_price_minor"]
        _require(total == row["total_minor"], "schema: line item total")
    else:
        _fields(row, "doc_id payment_id entity_id timestamp currency amount_minor fee_minor invoice_refs")
        _pattern(row["payment_id"], r"P-[a-z0-9]{4}")
        _stamp(row["timestamp"])
        _integer(row["amount_minor"])
        _integer(row["fee_minor"])
        for ref in _list(row["invoice_refs"]):
            _fields(ref, "invoice_id amount_minor")
            _pattern(ref["invoice_id"], r"INV-\d{4}")
            _integer(ref["amount_minor"], 1)
    _pattern(row["doc_id"], r"D-[a-z0-9]{6}")
    _pattern(row["entity_id"], r"E-[a-z0-9]{4}")
    _require(row["currency"] in CURRENCIES, "schema: document currency")
    return "doc:" + row["doc_id"]


def _headers(manifest: Record, policy: Record) -> None:
    _fields(manifest, "schema_version config_version generator_version episode_id base_currency "
            "currency_exponents period_start period_end timezone counts rounding")
    _pattern(manifest["episode_id"], r"ep_[0-9a-f]{12}")
    _require(manifest["schema_version"] == "1.0" and manifest["rounding"] == "ROUND_HALF_EVEN"
             and manifest["base_currency"] == "KZT" and manifest["timezone"] == "+05:00",
             "schema: manifest constants")
    for name in ("generator_version", "config_version"):
        _pattern(manifest[name], r".+")
    exponents = manifest["currency_exponents"]
    _fields(exponents, " ".join(CURRENCIES))
    for src in CURRENCIES:
        _integer(exponents[src])
        _require(exponents[src] == {"JPY": 0, "KWD": 3}.get(src, 2), "schema: currency exponent")
    start, end = (_stamp(manifest[col]) for col in ("period_start", "period_end"))
    _require(start.utcoffset() == end.utcoffset() == ZONE.utcoffset(None)
             and start.day == 1 and start.time().isoformat() == "00:00:00"
             and end.date() >= start.date() and (end.date() + timedelta(days=1)).day == 1
             and (start.year, start.month) == (end.year, end.month)
             and (end.hour, end.minute, end.second) == (23, 59, 59), "schema: reporting month")
    _fields(manifest["counts"], "ledger invoices payments fx")
    for name in sorted(manifest["counts"]):
        _integer(manifest["counts"][name])
    _fields(policy, "fx_rate_date fx_weekend_rule fx_direction period_boundary_tz tail_days ledger_currency "
            "finding_amount_currency duplicate_rule invoice_number_normalization rounding amount_tolerance_minor")
    expected = {
        "fx_rate_date": "document_date_in_company_tz", "fx_weekend_rule": "last_available_on_or_before",
        "fx_direction": "rate = units of target per 1 unit of source", "period_boundary_tz": "+05:00",
        "ledger_currency": "base", "finding_amount_currency": "base", "rounding": "ROUND_HALF_EVEN",
        "duplicate_rule": "the later-issued of two invoices with same entity, normalized number and amount is the duplicate",
        "invoice_number_normalization": "uppercase, remove spaces/dashes/underscores, strip leading zeros of numeric part",
    }
    _require(all(policy[k] == expected[k] for k in sorted(expected)), "schema: unsupported audit policy")
    _integer(policy["tail_days"])
    tolerances = policy["amount_tolerance_minor"]
    _fields(tolerances, "default fx_rate_error")
    for key in sorted(tolerances):
        _integer(tolerances[key])
    _require(tolerances == {"default": 0, "fx_rate_error": 1}, "schema: finding tolerance")


@dataclass(frozen=True)
class _Episode:
    manifest: Record
    policy: Record
    records: dict[str, Record]
    truth: Record
    provenance: Record


def _read(path: Path) -> _Episode:
    public, private = path / "open", path / "private"
    _require(sorted(p.name for p in public.iterdir()) == sorted(
        ("manifest.json", "policy.json", "ledger.csv", "fx.csv", "invoices.jsonl", "payments.jsonl")),
        "schema: public file set")
    _require(sorted(p.name for p in private.iterdir()) == ["provenance.json", "truth.json"], "schema: private file set")
    manifest = _json((public / "manifest.json").read_text(encoding="utf-8"))
    policy = _json((public / "policy.json").read_text(encoding="utf-8"))
    _headers(manifest, policy)
    records = {}
    for table in ("ledger", "invoices", "payments", "fx"):
        rows = _table(public, table)
        _require(len(rows) == manifest["counts"][table], "manifest: table count")
        for row in rows:
            key = _public_record(table, row)
            _require(key not in records, "schema: public identifier collision")
            records[key] = row
    counts = manifest["counts"]
    _require(200 <= counts["ledger"] <= 2000 and 20 <= counts["invoices"] + counts["payments"] <= 200,
             "manifest: episode size")
    truth = _json((private / "truth.json").read_text(encoding="utf-8"))
    provenance = _json((private / "provenance.json").read_text(encoding="utf-8"))
    return _Episode(manifest, policy, records, truth, provenance)

def _normalized_number(number: str) -> str:
    compact = re.sub(r"[-_ ]", "", number.upper())
    return re.sub(r"[0-9]+", lambda match: str(int(match[0])), compact)


def _evidence_amount(kind: str, affected: list[str], records: list[Record], policy: Record) -> int:
    """Calculate one claim from exactly one evidence set, without a book or answer key."""
    docs = [r for r in records if "doc_id" in r]
    rows = [r for r in records if "row_id" in r]
    quotes = [r for r in records if "rate" in r]
    _require(len(rows) == 1, "evidence: missing or ambiguous posting")
    row = rows[0]
    amount = row["debit_minor"] + row["credit_minor"]
    if kind == "duplicate_invoice":
        _require(len(docs) == 2 and not quotes and all(d.get("kind") == "invoice" for d in docs),
                 "evidence: duplicate documents")
        first, second = sorted(docs, key=lambda d: _stamp(d["issued_at"]))
        _require(affected == ["doc:" + second["doc_id"]] and
                 _stamp(first["issued_at"]) < _stamp(second["issued_at"]) and
                 _normalized_number(first["invoice_id"]) == _normalized_number(second["invoice_id"]) and
                 all(first[k] == second[k] for k in ("currency", "entity_id", "total_minor", "line_items")),
                 "evidence: duplicate identity")
        _require(row["invoice_id"] == second["invoice_id"] and row["payment_id"] is None and
                 row["entity_id"] in (None, second["entity_id"]) and
                 ((row["account"] == "1200" and row["debit_minor"]) or
                  (row["account"] == "4000" and row["credit_minor"])), "evidence: duplicate posting")
        return amount
    if kind == "payment_misallocation":
        payments = [d for d in docs if "payment_id" in d]
        sales = [d for d in docs if d.get("kind") == "invoice"]
        _require(len(payments) == len(sales) == 1 and len(docs) == 2 and not quotes,
                 "evidence: remittance documents")
        payment, sale = payments[0], sales[0]
        _require(sorted(affected) == sorted("doc:" + d["doc_id"] for d in docs) and
                 row["payment_id"] == payment["payment_id"] and
                 sale["entity_id"] == payment["entity_id"] and row["entity_id"] in (None, sale["entity_id"]) and
                 sale["currency"] == payment["currency"], "evidence: remittance identity")
        refs = [ref["invoice_id"] for ref in payment["invoice_refs"]]
        _require(sale["invoice_id"] in refs and row["credit_minor"] > 0, "evidence: remittance allocation")
        if row["account"] == "2150":
            # An unapplied-cash posting has no invoice number. One intended
            # invoice makes the target identifiable from this proof alone.
            _require(len(set(refs)) == 1, "evidence: ambiguous unapplied cash target")
            if payment["currency"] == row["currency"]:
                _require(amount <= sum(ref["amount_minor"] for ref in payment["invoice_refs"]),
                         "evidence: diverted credit exceeds remittance")
            _require(row["invoice_id"] is None and payment["amount_minor"] + payment["fee_minor"] ==
                     sum(ref["amount_minor"] for ref in payment["invoice_refs"]),
                     "evidence: unexplained unapplied cash")
            if payment["currency"] == "KZT":
                _require(row["credit_minor"] <= sum(ref["amount_minor"] for ref in payment["invoice_refs"]),
                         "evidence: diversion exceeds remittance")
        else:
            _require(row["account"] == "1200" and row["invoice_id"] not in refs and
                     isinstance(row["invoice_id"], str) and
                     _normalized_number(row["invoice_id"]) == _normalized_number(sale["invoice_id"]),
                     "evidence: redirected allocation")
        return row["credit_minor"]

    _require(len(docs) == 1 and affected == ["doc:" + docs[0]["doc_id"]], "evidence: affected document")
    doc = docs[0]
    payment_doc = "payment_id" in doc
    if payment_doc:
        linked = row["payment_id"] == doc["payment_id"] and row["invoice_id"] is None
        account = "1010"
    else:
        linked = doc["kind"] == "invoice" and row["invoice_id"] == doc["invoice_id"] and row["payment_id"] is None
        account = "1200" if kind == "fx_rate_error" else "4000"
    _require(linked and row["entity_id"] in (None, doc["entity_id"]) and row["account"] == account and
             row["credit_minor" if account == "4000" else "debit_minor"] > 0, "evidence: posting link or side")
    at = _stamp(doc["timestamp" if payment_doc else "issued_at"])
    if kind == "cutoff_error":
        zone = _stamp("2000-01-01T00:00:00" + policy["period_boundary_tz"]).tzinfo
        a, b = at.astimezone(zone), _stamp(row["timestamp"]).astimezone(zone)
        _require((a.year, a.month) != (b.year, b.month), "evidence: cutoff did not cross month end")
        if not quotes:
            return amount
    _require(len(quotes) == 1, "evidence: effective FX quote required")
    q = quotes[0]
    _require(q["source_currency"] == doc["currency"] and q["target_currency"] == row["currency"]
             and q["date"] == _business_day(at).isoformat(), "evidence: incorrect effective quote")
    correct = convert(doc["amount_minor" if payment_doc else "total_minor"], doc["currency"],
                      q["target_currency"], parse_rate(q["rate"]))
    _require(correct != amount, "evidence: no FX discrepancy")
    return correct if kind == "cutoff_error" else abs(correct - amount)


def _issues(ep: _Episode) -> dict[str, Record]:
    t = ep.truth
    _fields(t, "episode_id issues")
    _require(t["episode_id"] == ep.manifest["episode_id"], "truth: episode identity")
    items = _list(t["issues"])
    _require(len(items) <= 12, "truth: issue count")
    by_id = {}
    fx_docs = set()
    for item in items:
        _fields(item, "issue_id type affected_ids expected_amount_minor currency evidence_sets")
        _pattern(item["issue_id"], r".+")
        _require(item["issue_id"] not in by_id and item["type"] in TYPES and item["currency"] == "KZT",
                 "truth: issue identity, type or currency")
        _integer(item["expected_amount_minor"], 1)
        affected = _strings(item["affected_ids"])
        _require(1 <= len(affected) <= 10 and affected == sorted(affected) and
                 all(k in ep.records for k in affected), "truth: affected IDs")
        if item["type"] == "fx_rate_error":
            fx_docs.update(affected)
        by_id[item["issue_id"]] = item
    start, end = (_stamp(ep.manifest[key]).date() for key in ("period_start", "period_end"))
    tail_end = end + timedelta(days=ep.policy["tail_days"])
    for item in items:
        groups = _list(item["evidence_sets"])
        _require(bool(groups), "truth: missing evidence sets")
        accounts = set()
        for group in groups:
            keys = _strings(group)
            _require(1 <= len(keys) <= 10 and all(k in ep.records for k in keys), "truth: evidence IDs")
            records = [ep.records[k] for k in keys]
            if item["type"] == "cutoff_error":
                coupled = item["affected_ids"][0] in fx_docs
                _require(any("rate" in r for r in records) == coupled, "evidence: cutoff quote coverage")
            computed = _evidence_amount(item["type"], item["affected_ids"], records, ep.policy)
            _require(computed == item["expected_amount_minor"], "truth: evidence amount mismatch")
            if item["type"] == "cutoff_error":
                dates = sorted(_stamp(r.get("issued_at") or r["timestamp"]).astimezone(ZONE).date()
                               for r in records if "doc_id" in r or "row_id" in r)
                _require(start <= dates[0] <= end < dates[1] <= tail_end,
                         "evidence: cutoff outside reporting window")
            accounts.update(r["account"] for r in records if "row_id" in r)
        if item["type"] == "duplicate_invoice":
            _require(accounts == {"1200", "4000"}, "truth: duplicate needs both alternative proofs")
    return by_id


def _rollback(ep: _Episode, issues: dict[str, Record]) -> dict[str, Record]:
    p = ep.provenance
    _fields(p, "seed config config_hash generator_version stream_keys mutations distractors")
    _integer(p["seed"])
    _require(isinstance(p["config"], dict) and p["generator_version"] == ep.manifest["generator_version"],
             "provenance: generation metadata")
    canonical_config = json.dumps(p["config"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    _require(hashlib.sha256(canonical_config.encode()).hexdigest() == p["config_hash"], "provenance: config hash")
    _strings(p["stream_keys"])
    traps = _list(p["distractors"])
    _require(bool(traps) or bool(issues), "provenance: missing clean distractors")
    for trap in traps:
        _fields(trap, "kind ids")
        ids = _strings(trap["ids"])
        _require(trap["kind"] in TRAPS and bool(ids) and all(k.startswith("doc:") and k in ep.records for k in ids),
                 "provenance: distractor")
    clean = ep.records.copy()
    owned: list[str] = []
    pairs = {"fx_partial_boundary": ["cutoff_error", "fx_rate_error"],
             "duplicate_with_payment": ["duplicate_invoice", "payment_misallocation"]}
    for mutation in reversed(_list(p["mutations"])):
        _fields(mutation, "issue_ids injector causal_note diff decomposition")
        keys = _strings(mutation["issue_ids"])
        name = mutation["injector"]
        _require(isinstance(name, str) and (name in TYPES or name in pairs) and
                 all(k in issues for k in keys), "provenance: mutation identity")
        _require(sorted(issues[k]["type"] for k in keys) == pairs.get(name, [name]), "provenance: mutation types")
        _require(isinstance(mutation["causal_note"], str) and bool(mutation["causal_note"]) and
                 ((name in pairs and isinstance(mutation["decomposition"], dict) and bool(mutation["decomposition"]))
                  or (name in TYPES and mutation["decomposition"] is None)), "provenance: decomposition")
        owned.extend(keys)
        diff = mutation["diff"]
        _require(isinstance(diff, dict) and sorted(diff) in (["added", "modified", "removed"],
                 ["added", "before", "modified", "removed"]), "provenance: diff fields")
        added, removed, modified = (_strings(diff[k]) for k in ("added", "removed", "modified"))
        changed = added + removed + modified
        _require(bool(changed) and len(changed) == len(set(changed)), "provenance: conflicting diff")
        visible_postings = {k for k in added + modified if k.startswith("ledger:")}
        for key in keys:
            issue = issues[key]
            _require(all(visible_postings.intersection(group) for group in issue["evidence_sets"]),
                     "provenance: cited posting unaffected by mutation")
            if issue["type"] == "duplicate_invoice":
                _require(all(k in added for k in issue["affected_ids"]), "provenance: duplicate must be added")
        before = diff.get("before", {})
        _require(isinstance(before, dict) and sorted(before) == sorted(modified + removed),
                 "provenance: before-image coverage")
        _require(all(k in clean for k in added + modified) and all(k not in clean for k in removed),
                 "provenance: diff does not apply")
        for k in added:
            del clean[k]
        for k in sorted(before):
            record = before[k]
            _require(isinstance(record, dict), "provenance: before-image")
            if k.startswith("ledger:"):
                table = "ledger"
            elif k.startswith("fx:"):
                table = "fx"
            else:
                table = "invoices" if "invoice_id" in record else "payments"
            _require(_public_record(table, record) == k and clean.get(k) != record, "provenance: before-image identity")
            clean[k] = record
    _require(sorted(owned) == sorted(issues), "provenance: issue coverage")
    return clean

def _clean_book(ep: _Episode, clean: dict[str, Record]) -> None:
    rows = [clean[k] for k in sorted(clean) if k.startswith("ledger:")]
    docs = [clean[k] for k in sorted(clean) if k.startswith("doc:")]
    sales = {d["invoice_id"]: d for d in docs if d.get("kind") == "invoice"}
    invoices = [d["invoice_id"] for d in docs if "invoice_id" in d]
    receipts = {d["payment_id"]: d for d in docs if "payment_id" in d}
    _require(len(set(invoices)) == len(invoices) and len(receipts) + len(invoices) == len(docs), "I5: duplicate document")
    actual: Counter[Post] = Counter()
    groups: Counter[str] = Counter()
    ar: Counter[str] = Counter()
    for row in rows:
        sale, payment = row["invoice_id"], row["payment_id"]
        _require((sale is not None or payment is not None) and (sale is None or sale in sales)
                 and (payment is None or payment in receipts), "I6: unknown ledger reference")
        entity = (receipts[payment] if payment is not None else sales[sale])["entity_id"]
        _require(row["entity_id"] in (None, entity), "I6: inconsistent ledger counterparty")
        signed = row["debit_minor"] - row["credit_minor"]
        groups[payment or sale] += signed
        actual[_stamp(row["timestamp"]), row["account"], signed, entity, sale, payment] += 1
        if row["account"] == "1200":
            _require(sale is not None, "I6: receivable without invoice")
            ar[sale] += signed
    _require(sum(groups[k] for k in sorted(groups)) == 0, "I2: trial balance")
    _require(all(groups[k] == 0 for k in sorted(groups)), "I1: unbalanced document group")
    expected: Counter[Post] = Counter()
    carrying: Counter[str] = Counter()
    applied: Counter[str] = Counter()
    start, end = (_stamp(ep.manifest[k]).date() for k in ("period_start", "period_end"))
    end += timedelta(days=ep.policy["tail_days"])

    def in_base(amount: int, doc: Record, moment: datetime) -> int:
        src = doc["currency"]
        if src == "KZT":
            return amount
        quote = clean.get(f"fx:{_business_day(moment)}:{src}:KZT")
        _require(quote is not None, "I7: missing effective business-day quote")
        return convert(amount, src, "KZT", parse_rate(quote["rate"]))

    for doc in docs:
        payment = doc.get("payment_id")
        at = _stamp(doc["timestamp" if payment else "issued_at"])
        _require(start <= at.astimezone(ZONE).date() <= end, "I5: document outside reporting window")

        def sale_for(number: str) -> Record:
            sale = sales.get(number)
            _require(sale is not None and sale["entity_id"] == doc["entity_id"]
                     and sale["currency"] == doc["currency"] and _stamp(sale["issued_at"]) <= at,
                     "I6: invalid document reference")
            return sale

        def post(account: str, signed: int, sale: str | None = None) -> None:
            if signed:
                expected[at, account, signed, doc["entity_id"], sale, payment] += 1
                if account == "1200" and sale is not None:
                    carrying[sale] += signed

        if payment is None:
            number = doc["original_invoice_id"] or doc["invoice_id"]
            original = sale_for(number)
            value = in_base(doc["total_minor"], doc, _stamp(original["issued_at"]))
            if doc["kind"] == "invoice":
                post("1200", value, number)
                post("4000", -value, number)
            else:
                applied[number] += doc["total_minor"]
                post("4100", value, number)
                post("1200", -value, number)
            continue
        # §4.5 defines the posted bank receipt net of its fee.
        refs = doc["invoice_refs"]
        unapplied = doc["amount_minor"] + doc["fee_minor"] - sum(ref["amount_minor"] for ref in refs)
        _require(unapplied >= 0, "I4: remittance exceeds gross receipt")
        cash = in_base(doc["amount_minor"], doc, at)
        fee = in_base(doc["fee_minor"], doc, at)
        excess = in_base(unapplied, doc, at)
        post("1010", cash)
        post("6100", fee)
        post("2150", -excess)
        allocated = 0
        for ref in refs:
            number = ref["invoice_id"]
            sale = sale_for(number)
            value = in_base(ref["amount_minor"], doc, _stamp(sale["issued_at"]))
            allocated += value
            applied[number] += ref["amount_minor"]
            post("1200", -value, number)
        post("7100", allocated + excess - cash - fee)
    _require(ar == carrying and all(ar[k] >= 0 and applied[k] <= sales[k]["total_minor"] for k in sorted(sales)),
             "I3: receivable balance")
    _require(actual == expected, "I5: document posting multiplicity, time or amount")


def _checked(path: Path) -> str:
    try:
        ep = _read(path)
        issues = _issues(ep)
        clean = _rollback(ep, issues)
        _clean_book(ep, clean)
        return str(ep.manifest["episode_id"])
    except InvalidEpisode:
        raise
    except (OSError, ValueError, TypeError, KeyError, csv.Error, OverflowError, RecursionError) as exc:
        raise InvalidEpisode("schema: missing or malformed episode data") from exc


def validate_episode(path: Path) -> None:
    """Check one serialized episode deterministically. Raise ValueError on its first discrepancy."""
    _checked(path)


def _episode_dirs(root: Path) -> list[Path]:
    if (root / "open").exists() or (root / "private").exists():
        return [root]
    if not root.is_dir():
        return []
    # Detect partial episodes even when one of their two domains is absent.
    paths = {p.parent for name in ("open", "private") for p in root.rglob(name)}
    for split in ("train", "dev"):
        part = root / split
        if part.is_dir():
            paths.update(p for p in part.iterdir() if p.is_dir())
    return sorted(paths)


def validate_data(data: Path, out: Path = Path("reports/validation.json")) -> int:
    """Validate each episode and atomically publish canonical results. Return 0 only on success."""
    entries = []
    for path in _episode_dirs(data):
        try:
            episode_id = _checked(path)
            entries.append({"episode_id": episode_id, "valid": True})
        except InvalidEpisode as exc:
            entries.append({"episode": path.relative_to(data).as_posix(), "valid": False, "error": str(exc)})
    valid = bool(entries) and all(entry["valid"] for entry in entries)
    report = {"episodes": entries, "valid": valid, "checked": len(entries)}
    if not entries:
        report["error"] = "no episodes found"
    out.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=out.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            json.dump(report, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    try:
        temp.replace(out)
    finally:
        temp.unlink(missing_ok=True)
    return int(not valid)

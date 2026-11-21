"""Tools, pagination and error codes (SPEC §8.3, §8.4)."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from env_helpers import check_obs
from auditgym.env import AuditEnv
from auditgym.env.errors import MESSAGES
from auditgym.env.pagination import encode_cursor


def _call(env: AuditEnv, tool: str, **args: object) -> dict:
    obs, *_ = env.step({"tool": tool, "args": args})
    check_obs(obs)
    return obs


def _error(obs: dict) -> str:
    assert obs["ok"] is False and obs["result"] is None
    assert obs["error"]["message"] == MESSAGES[obs["error"]["code"]]
    return obs["error"]["code"]


def _walk(env: AuditEnv, table: str, limit: int | None = None) -> list[dict]:
    items, cursor = [], None
    while True:
        args: dict = {"table": table, "cursor": cursor}
        if limit is not None:
            args["limit"] = limit
        result = _call(env, "list_records", **args)["result"]
        items.extend(result["items"])
        cursor = result["next_cursor"]
        if cursor is None:
            return items


# --- list_records / pagination -----------------------------------------------------------

def test_default_page_is_50_and_reports_total(started: AuditEnv) -> None:
    result = _call(started, "list_records", table="fx")["result"]
    assert len(result["items"]) == 50 and result["total"] == 60
    assert isinstance(result["next_cursor"], str)


def test_limit_above_50_is_truncated(started: AuditEnv) -> None:
    result = _call(started, "list_records", table="fx", limit=500)["result"]
    assert len(result["items"]) == 50


@pytest.mark.parametrize("limit", [0, -1, "5", 2.0, True, None])
def test_bad_limit_is_bad_args(started: AuditEnv, limit: object) -> None:
    assert _error(_call(started, "list_records", table="fx", limit=limit)) == "BAD_ARGS"


@pytest.mark.parametrize("table,limit", [("fx", None), ("fx", 7), ("ledger", 4), ("invoices", 1)])
def test_full_walk_returns_every_record_once_in_file_order(started: AuditEnv, tiny_dir: Path,
                                                         table: str, limit: int | None) -> None:
    items = _walk(started, table, limit)
    if table in ("fx", "ledger"):
        lines = (tiny_dir / f"open/{table}.csv").read_text("utf-8").splitlines()[1:]
        keys = [line.split(",")[0] + (":" + ":".join(line.split(",")[1:3]) if table == "fx" else "")
                for line in lines]
        got = [f"{r['date']}:{r['source_currency']}:{r['target_currency']}" if table == "fx" else r["row_id"]
               for r in items]
    else:
        keys = [json.loads(line)["doc_id"] for line in
                (tiny_dir / f"open/{table}.jsonl").read_text("utf-8").splitlines()]
        got = [r["doc_id"] for r in items]
    assert got == keys and len(set(got)) == len(got)


def test_cursor_is_stable(started: AuditEnv) -> None:
    first = _call(started, "list_records", table="fx", limit=10)["result"]
    again = _call(started, "list_records", table="fx", limit=10)["result"]
    assert first == again
    page2a = _call(started, "list_records", table="fx", cursor=first["next_cursor"], limit=10)["result"]
    page2b = _call(started, "list_records", table="fx", cursor=first["next_cursor"], limit=10)["result"]
    assert page2a == page2b and page2a["items"] != first["items"]


def test_last_page_has_null_cursor(started: AuditEnv) -> None:
    first = _call(started, "list_records", table="fx")["result"]
    last = _call(started, "list_records", table="fx", cursor=first["next_cursor"])["result"]
    assert len(last["items"]) == 10 and last["next_cursor"] is None
    exact = _call(started, "list_records", table="payments")["result"]
    assert exact == {"items": exact["items"], "next_cursor": None, "total": 1}


def _b64(obj: object) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


@pytest.mark.parametrize("cursor", [
    "", "!!!", "not-a-cursor", "e30", _b64([1, 2]), _b64({"t": "fx", "o": 10}),
    _b64({"t": "fx", "o": "10", "e": "ep_01234"}), _b64({"t": "fx", "o": 10, "e": "ep_99999"}),
    _b64({"t": "fx", "o": -1, "e": "ep_01234"}), _b64({"t": "fx", "o": 60, "e": "ep_01234"}),
    _b64({"t": "fx", "o": 10, "e": "ep_01234", "x": 1}), _b64({"t": "ledger", "o": 1, "e": "ep_01234"}),
    _b64({"t": "fx", "o": True, "e": "ep_01234"}), "A" * 1000,
    encode_cursor("fx", 10, "ep_0123456789ab") + "==",
])
def test_bad_cursor(started: AuditEnv, cursor: str) -> None:
    assert _error(_call(started, "list_records", table="fx", cursor=cursor)) == "BAD_CURSOR"


def test_cursor_from_other_table_is_rejected(started: AuditEnv) -> None:
    cursor = _call(started, "list_records", table="fx", limit=1)["result"]["next_cursor"]
    assert _error(_call(started, "list_records", table="ledger", cursor=cursor)) == "BAD_CURSOR"


def test_cursor_from_other_episode_is_rejected(started: AuditEnv) -> None:
    foreign = encode_cursor("fx", 10, "ep_ffffffffffff")
    assert _error(_call(started, "list_records", table="fx", cursor=foreign)) == "BAD_CURSOR"


def test_ledger_record_types(started: AuditEnv) -> None:
    row = _call(started, "get_record", table="ledger", id="R-4p8x1t")["result"]["record"]
    assert row == {"row_id": "R-4p8x1t", "timestamp": "2026-03-10T18:00:00+05:00", "account": "1010",
                   "debit_minor": 50004500, "credit_minor": 0, "currency": "KZT", "entity_id": "E-7k2p",
                   "invoice_id": None, "payment_id": "P-3x7a"}


# --- get_record / get_document ------------------------------------------------------------

def test_get_record_each_table(started: AuditEnv) -> None:
    assert _call(started, "get_record", table="invoices", id="D-7q2x9a")["result"]["record"]["invoice_id"] == "INV-1042"
    assert _call(started, "get_record", table="payments", id="D-k1p0zz")["result"]["record"]["payment_id"] == "P-3x7a"
    fx = _call(started, "get_record", table="fx", id="2026-03-02:USD:KZT")["result"]["record"]
    assert fx["rate"] == "500.003000" and isinstance(fx["rate"], str)


def test_get_record_wrong_table_is_unknown_id(started: AuditEnv) -> None:
    assert _error(_call(started, "get_record", table="invoices", id="D-k1p0zz")) == "UNKNOWN_ID"
    assert _error(_call(started, "get_record", table="fx", id="fx:2026-03-02:USD:KZT")) == "UNKNOWN_ID"


def test_get_document(started: AuditEnv) -> None:
    inv = _call(started, "get_document", doc_id="D-u5s0b3")["result"]
    assert inv["table"] == "invoices" and inv["document"]["currency"] == "USD"
    pay = _call(started, "get_document", doc_id="D-k1p0zz")["result"]
    assert pay["table"] == "payments" and pay["document"]["invoice_refs"][0]["invoice_id"] == "INV-2001"


@pytest.mark.parametrize("fake", ["D-k9mq1c", "d-k9mq1b", "D-K9MQ1B", "doc:D-k9mq1b", "D-zzzzzz", " D-k9mq1b", ""])
def test_unknown_id_message_is_uniform(started: AuditEnv, fake: str) -> None:
    obs_doc = _call(started, "get_document", doc_id=fake)
    obs_rec = _call(started, "get_record", table="invoices", id=fake)
    assert _error(obs_doc) == _error(obs_rec) == "UNKNOWN_ID"
    assert obs_doc["error"] == obs_rec["error"] == {"code": "UNKNOWN_ID", "message": "record not found"}


# --- every error code ---------------------------------------------------------------------

def test_unknown_tool(started: AuditEnv) -> None:
    for action in ({"tool": "rm_rf"}, {"tool": "__import__", "args": {}}, {"tool": "LIST_RECORDS"}):
        obs, *_ = started.step(action)
        check_obs(obs)
        assert _error(obs) == "UNKNOWN_TOOL"


@pytest.mark.parametrize("action", [
    None, [], "list_records", {}, {"args": {}}, {"tool": 1}, {"tool": "list_records", "args": []},
    {"tool": "list_records", "args": {}}, {"tool": "list_records", "args": {"table": "users"}},
    {"tool": "list_records", "args": {"table": "fx", "extra": 1}},
    {"tool": "list_records", "args": {"table": "fx"}, "x": 1},
    {"tool": "get_record", "args": {"table": "fx"}}, {"tool": "get_record", "args": {"table": "fx", "id": 5}},
    {"tool": "get_document", "args": {"doc_id": None}}, {"tool": "get_document", "args": {"doc_id": "x" * 500}},
    {"tool": "calculator", "args": {"expression": 5}}, {"tool": "submit", "args": {"findings": []}},
    {"tool": "submit"}, {"tool": "list_records", "args": {"table": "fx", "limit": float("nan")}},
    {"tool": "list_records", "args": {"table": {"fx"}}},
])
def test_bad_args(started: AuditEnv, action: object) -> None:
    obs, reward, term, trunc, _ = started.step(action)  # type: ignore[arg-type]
    check_obs(obs)
    assert _error(obs) == "BAD_ARGS"
    assert obs["steps_used"] == 1 and (reward, term, trunc) == (0.0, False, False)


def test_unknown_id(started: AuditEnv) -> None:
    assert _error(_call(started, "get_record", table="ledger", id="R-000000")) == "UNKNOWN_ID"


def test_bad_cursor_code(started: AuditEnv) -> None:
    assert _error(_call(started, "list_records", table="fx", cursor="zzz")) == "BAD_CURSOR"


def test_calc_error_code(started: AuditEnv) -> None:
    assert _error(_call(started, "calculator", expression="__import__('os')")) == "CALC_ERROR"


def test_episode_done_code(env: AuditEnv) -> None:
    obs, *_ = env.step({"tool": "list_records", "args": {"table": "fx"}})
    assert _error(obs) == "EPISODE_DONE"


def test_every_error_consumes_a_step(started: AuditEnv) -> None:
    actions = [{"tool": "x"}, {"tool": "list_records", "args": {}},
               {"tool": "get_document", "args": {"doc_id": "D-000000"}},
               {"tool": "list_records", "args": {"table": "fx", "cursor": "bad"}},
               {"tool": "calculator", "args": {"expression": "1/0"}}]
    for n, action in enumerate(actions, 1):
        obs, *_ = started.step(action)
        assert obs["ok"] is False and obs["steps_used"] == n


def test_internal_exception_becomes_bad_args(started: AuditEnv, monkeypatch: pytest.MonkeyPatch,
                                             caplog: pytest.LogCaptureFixture) -> None:
    import auditgym.env.core as core

    def boom(data, args):  # noqa: ANN001
        raise KeyError("/secret/private/truth.json")

    monkeypatch.setattr(core, "READ_TOOLS", {**core.READ_TOOLS, "get_document": boom})
    with caplog.at_level("ERROR", logger="auditgym.env"):
        obs, *_ = started.step({"tool": "get_document", "args": {"doc_id": "D-7q2x9a"}})
    assert obs["error"] == {"code": "BAD_ARGS", "message": "invalid arguments"}
    assert "secret" not in json.dumps(obs)
    assert any("internal error" in r.getMessage() for r in caplog.records)

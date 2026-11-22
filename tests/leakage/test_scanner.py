import json
from pathlib import Path

import pytest

from fintrace.grader.leakage import LeakIndex, scan_trajectory


@pytest.fixture
def episode(tmp_path: Path) -> Path:
    public = tmp_path / "open"
    public.mkdir()
    (public / "manifest.json").write_text('{"episode_id":"ep_000000000001","base_currency":"KZT"}')
    (public / "policy.json").write_text('{"rule":"remove spaces/dashes/underscores"}')
    (public / "invoices.jsonl").write_text('{"doc_id":"D-123456"}\n')
    (public / "payments.jsonl").touch()
    (public / "ledger.csv").write_text("row_id,currency\nR-123456,KZT\n")
    (public / "fx.csv").write_text("date,source_currency,target_currency,rate\n2026-03-31,USD,KZT,500\n")
    private = tmp_path / "private"
    private.mkdir()
    truth = {"episode_id": "ep_000000000001", "issues": [{"issue_id": "opaque-key",
             "affected_ids": ["doc:D-123456", "doc:D-999999"],
             "evidence_sets": [["ledger:R-123456", "fx:2026-03-31:USD:KZT"]], "currency": "KZT"}]}
    (private / "truth.json").write_text(json.dumps(truth))
    (private / "provenance.json").write_text('{"seed":90001,"mutations":[{"causal_note":"hidden memo"}]}')
    return tmp_path


@pytest.mark.parametrize("leak", [
    {"issue_id": "x"}, "opaque-key", "value doc:D-999999", "iss_01", "Injection",
    "truth.json", {"provenance": []}, "private/open", 90001, "debug=90001",
    {"seed": 123}, "trace C:\\data\\ep_00", "/home/agent/data", "../logs", "data/ep_00",
    "hidden memo", "local-test-salt", {"expected_amount_minor": 42},
])
def test_nested_disclosure(episode: Path, leak: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINTRACE_SALT", "local-test-salt")
    index = LeakIndex.from_episode(episode)
    with pytest.raises(ValueError, match="^restricted data in trajectory$"):
        index.check({"result": [{"nested": leak}]})


def test_open_values_are_legitimate(episode: Path) -> None:
    LeakIndex.from_episode(episode).check({"finding_types": ["fx_rate_error"], "result": [
        "doc:D-123456", "ledger:R-123456", "fx:2026-03-31:USD:KZT", "ep_000000000001",
        "KZT", "remove spaces/dashes/underscores", "record not found", 42]})


def test_scan_all_visible_channels(episode: Path) -> None:
    record = {"episode_id": "ep_000000000001", "observation": {}, "info": {}, "error": None}
    path = episode / "run.jsonl"
    clean = json.dumps(record) + "\n"
    path.write_text(clean * 50)
    assert scan_trajectory(episode, path) == 50
    for field in ("observation", "obs", "info", "error"):
        path.write_text(clean + json.dumps({**record, field: [{"nested": "opaque-key"}]}))
        with pytest.raises(ValueError, match=rf"^restricted data in {field} on line 2$"):
            scan_trajectory(episode, path)

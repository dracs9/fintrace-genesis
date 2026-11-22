"""Exercise the local HTTP bridge with generated books, real policies and spawned graders."""
from __future__ import annotations

import json
from collections.abc import Iterator
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread
from typing import Any

import pytest

from fintrace.env.core import ensure_episode, load_config
from fintrace.cli import run_cached_episodes
from fintrace.grader.leakage import LeakIndex
from fintrace.web_server import server

Json = dict[str, Any]
CONFIG = {"seed": 20000, "config": "medium"}


@pytest.fixture
def api(tmp_path: Path) -> Iterator[HTTPConnection]:
    with server(tmp_path, 0) as http:
        thread = Thread(target=http.serve_forever, kwargs={"poll_interval": 0.02})
        thread.start()
        client = HTTPConnection(*http.server_address, timeout=90)
        try:
            yield client
        finally:
            client.close()
            http.shutdown()
            thread.join()


def request(api: HTTPConnection, path: str, body: Json | None = None,
            status: int = 200, **headers: str) -> Json:
    """Round-trip JSON through the actual server and verify the HTTP response code."""
    api.request("GET" if body is None else "POST", path,
                None if body is None else json.dumps(body),
                {"Content-Type": "application/json", **headers})
    response = api.getresponse()
    data = json.loads(response.read())
    assert response.status == status, data
    return data


def step(api: HTTPConnection, tool: str, **args: object) -> Json:
    """Send an action envelope through the public step route."""
    return request(api, "/api/env/step", {"tool": tool, "args": args})


def test_active_api_and_public_tools(api: HTTPConnection, tmp_path: Path) -> None:
    reference = ensure_episode(tmp_path / "reference", CONFIG["seed"], load_config("medium"))
    scanner = LeakIndex.from_episode(reference)
    reset = request(api, "/api/env/reset", CONFIG)
    scanner.check(reset)
    assert set(reset) == {"obs", "info"}
    assert reset["obs"]["steps_used"] == 0
    first = step(api, "list_records", table="ledger", cursor=None, limit=50)
    page = first["obs"]["result"]
    second = step(api, "list_records", table="ledger", cursor=page["next_cursor"], limit=50)
    assert len(page["items"]) == 50
    assert page["items"][0] != second["obs"]["result"]["items"][0]
    invoices = step(api, "list_records", table="invoices", cursor=None, limit=50)
    invoice = invoices["obs"]["result"]["items"][0]
    document = step(api, "get_document", doc_id=invoice["doc_id"])
    assert document["obs"]["result"]["document"] == invoice
    calc = step(api, "calculator", expression="100/8")
    assert calc["obs"]["result"]["value"] == "12.5"
    bad = step(api, "get_document", doc_id="private/truth.json")
    assert bad["obs"]["error"]["message"] in {"invalid arguments", "record not found"}
    for frame in (first, second, invoices, document, calc, bad):
        scanner.check(frame)
        assert frame["info"] == {} and frame["reward"] == 0
        assert not frame["terminated"] and not frame["truncated"]
        assert "grade" not in frame
    end = step(api, "submit", findings={"episode_id": reset["info"]["episode_id"], "findings": []})
    assert end["terminated"] and end["grade"]["valid"]
    assert end["grade"]["inspection_steps"] == 6
    assert end["grade"]["reward"] == end["reward"] == end["info"]["grade"]["reward"]
    after = step(api, "calculator", expression="2")
    assert after["obs"]["error"]["code"] == "EPISODE_DONE" and "grade" not in after


def test_truncation_and_eval_mask(api: HTTPConnection) -> None:
    request(api, "/api/env/reset", {**CONFIG, "mode": "eval"})
    for i in range(1, 101):
        frame = step(api, "calculator", expression="0")
        assert frame["obs"]["steps_used"] == i
        assert frame["reward"] == 0 and frame["info"] == {}
        assert frame["truncated"] == (i == 100)
        assert ("grade" in frame) == (i == 100)
    assert frame["grade"]["inspection_steps"] == 100
    assert frame["grade"]["valid"] and frame["grade"]["matches"] == []


@pytest.mark.parametrize("policy", ["random", "flag_all", "reconciler", "oracle"])
def test_policy_replay_and_summary(api: HTTPConnection, policy: str, tmp_path: Path) -> None:
    result = request(api, "/api/env/run-policy", {**CONFIG, "policy": policy})
    frames, grade = result["trajectory"], result["grade"]
    scanner = LeakIndex.from_episode(ensure_episode(tmp_path / "ref", CONFIG["seed"], load_config("medium")))
    assert frames[0]["action"] is None
    for index, frame in enumerate(frames):
        assert frame["obs"]["steps_used"] == index
        scanner.check(frame["obs"])
        if index < len(frames) - 1:
            scanner.check(frame["info"])
            assert "grade" not in frame and not frame["terminated"] and not frame["truncated"]
    assert frames[-1]["terminated"] and frames[-1]["reward"] == grade["reward"]
    assert len(grade["matches"]) == grade["tp"]
    if policy == "oracle":
        assert grade["tp"] > 0 and grade["f1"] == 1
    rows = request(api, "/api/runs")["runs"]
    assert len(rows) == 1
    assert rows[0]["episode_id"] == grade["episode_id"]
    assert rows[0]["config"] == "medium" and rows[0]["reward"] == grade["reward"]
    assert rows[0]["steps"] == len(frames) - 1


def test_isolation_reset_errors_and_static_boundary(api: HTTPConnection) -> None:
    assert request(api, "/api/runs") == {"runs": []}
    for path in ("/private/truth.json", "/logs/grader.jsonl", "/../SPEC.md"):
        assert request(api, path, status=404)["error"]["code"] == "UNKNOWN_TOOL"
    request(api, "/api/env/reset", CONFIG, 403, Origin="http://example.org")
    request(api, "/api/env/run-policy", {**CONFIG, "policy": "module:Class"}, 400)
    request(api, "/api/env/reset", {**CONFIG, "config": "../private"}, 400)
    request(api, "/api/env/reset", CONFIG)
    step(api, "calculator", expression="1")
    reset = request(api, "/api/env/reset", {**CONFIG, "seed": 90002, "mode": "eval"})
    assert reset["obs"]["steps_used"] == 0
    request(api, "/api/env/step", {}, 409, **{"If-Match": '"1"'})
    end = step(api, "submit", findings={"episode_id": "wrong", "findings": []})
    assert end["terminated"] and not end["grade"]["valid"]
    assert end["reward"] == 0 and end["grade"]["reward"] == -1 and end["info"] == {}


def test_cli_summary_joins_public_manifest(
    api: HTTPConnection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data/dev"
    episode = ensure_episode(data, 20000, load_config("easy"))
    run_cached_episodes(data, [episode], "flag_all", tmp_path / "runs", split="dev")
    assert (tmp_path / "logs/grader.jsonl").is_file()
    row, = request(api, "/api/runs")["runs"]
    assert row["episode_id"] == episode.name and row["policy"] == "flag_all"
    assert row["config"] == load_config("easy")["config_version"] and row["split"] == "dev"
    assert set(row) == {"episode_id", "policy", "steps", "f1", "reward", "precision",
                        "recall", "split", "config"}


def test_local_frontend_assets(api: HTTPConnection) -> None:
    for path in ("/", "/studio.css", "/studio.js", "/ui.js", "/findings.js", "/explorer.js",
                 "/inter.woff2", "/jetbrains-mono.woff2"):
        api.request("GET", path)
        response = api.getresponse()
        content = response.read()
        assert response.status == 200 and content
        if path.endswith(".woff2"):
            assert content.startswith(b"wOF2")


def test_public_host_mode_and_health(tmp_path: Path) -> None:
    with server(tmp_path, 0, public_hosts=("fintrace.example.com",)) as http:
        thread = Thread(target=http.serve_forever, kwargs={"poll_interval": 0.02})
        thread.start()
        client = HTTPConnection(*http.server_address, timeout=90)
        try:
            public = {"Host": "fintrace.example.com"}
            assert request(client, "/api/runs", **public) == {"runs": []}
            # Same-origin HTTPS call passes the boundary and reaches argument validation.
            request(client, "/api/env/reset", {**CONFIG, "config": "x"}, 400,
                    **public, Origin="https://fintrace.example.com")
            request(client, "/api/runs", status=403, Host="evil.example.com")
            request(client, "/api/env/reset", CONFIG, 403, **public, Origin="http://fintrace.example.com")
            request(client, "/api/env/reset", CONFIG, 403, **public, Origin="https://evil.example.com")
            client.request("GET", "/healthz", headers={"Host": "10.0.0.1:10000"})
            response = client.getresponse()
            assert response.status == 200 and response.read() == b"ok"
        finally:
            client.close()
            http.shutdown()
            thread.join()


def test_unconfigured_server_rejects_foreign_host(api: HTTPConnection) -> None:
    request(api, "/api/runs", status=403, Host="fintrace.example.com")

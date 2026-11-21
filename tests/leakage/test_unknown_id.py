"""Run with the real environment and the Role 1 generator (commit 0498514c)."""

import json
from pathlib import Path

import pytest

from auditgym.grader import GraderClient


def test_deleted_and_invented_id_have_identical_errors(tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    core = pytest.importorskip("auditgym.env.core", reason="requires the Role 2 environment",
                               exc_type=ModuleNotFoundError)
    monkeypatch.setenv("AUDITGYM_SALT", "test-eval-salt")
    # Seed 90017 in the reference generator actually removes a ledger row.
    with GraderClient(tmp_path, log_dir=tmp_path / "logs") as client:
        env = core.AuditEnv(data_root=tmp_path, mode="train", grader=client)
        try:
            _, info = env.reset(90017, "medium")
            episode = tmp_path / info["episode_id"]
            provenance = json.loads((episode / "private/provenance.json").read_bytes())
            deleted = next(key.removeprefix("ledger:") for mutation in provenance["mutations"]
                           for key in mutation["diff"]["removed"] if key.startswith("ledger:"))
            public = (episode / "open/ledger.csv").read_text(encoding="utf-8")
            invented = next(f"R-{i:06d}" for i in range(1_000_000) if f"R-{i:06d}" not in public
                            and f"R-{i:06d}" != deleted)
            assert deleted not in public
            for key in (deleted, invented):
                obs, _, _, _, _ = env.step({"tool": "get_record", "args": {"table": "ledger", "id": key}})
                assert obs["error"] == {"code": "UNKNOWN_ID", "message": "record not found"}
        finally:
            env.close()

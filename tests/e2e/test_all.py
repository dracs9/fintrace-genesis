"""E2E: ``auditgym all`` on a small config finishes in < 120 s (SPEC §14)."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


def test_all_small_config_under_120s(tmp_path: Path) -> None:
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "auditgym.cli", "all", "--config", "easy", "--n", "2", "--skip-tests"],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
    )
    elapsed = time.perf_counter() - started
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "all: done" in proc.stdout
    assert elapsed < 120
    assert len(list((tmp_path / "data" / "train").iterdir())) == 2
    assert len(list((tmp_path / "data" / "dev").iterdir())) == 2

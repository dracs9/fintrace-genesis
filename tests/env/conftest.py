from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_helpers import TINY, FakeGraderClient  # noqa: E402

from auditgym.env import AuditEnv  # noqa: E402


@pytest.fixture
def tiny_dir(tmp_path: Path) -> Path:
    """A private copy of the hand-made episode."""
    dst = tmp_path / "data" / "ep_0123456789ab"
    shutil.copytree(TINY, dst)
    return dst


@pytest.fixture
def grader() -> FakeGraderClient:
    return FakeGraderClient()


@pytest.fixture
def env(tiny_dir: Path, grader: FakeGraderClient) -> AuditEnv:
    return AuditEnv(tiny_dir.parent, mode="train", grader=grader)


@pytest.fixture
def started(env: AuditEnv, tiny_dir: Path) -> AuditEnv:
    env.reset_from_dir(tiny_dir)
    return env

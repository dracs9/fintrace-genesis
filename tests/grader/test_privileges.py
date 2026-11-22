"""Privilege drop for the Docker UID split (SPEC §9.5), with os/pwd stubbed on every platform."""

import os
import sys
from types import SimpleNamespace

import pytest

from fintrace.grader.privileges import drop_to


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Fake POSIX process starting as root; `fake.calls` records syscalls in order."""
    proc = SimpleNamespace(uid=0, calls=[])
    accounts = {"grader": SimpleNamespace(pw_uid=1002, pw_gid=1002)}

    def setuid(uid: int) -> None:
        proc.calls.append(("setuid", uid))
        proc.uid = uid

    monkeypatch.setitem(sys.modules, "pwd", SimpleNamespace(getpwnam=accounts.__getitem__))
    monkeypatch.setattr(os, "getuid", lambda: proc.uid, raising=False)
    monkeypatch.setattr(os, "geteuid", lambda: proc.uid, raising=False)
    monkeypatch.setattr(os, "setgroups", lambda g: proc.calls.append(("setgroups", tuple(g))),
                        raising=False)
    monkeypatch.setattr(os, "setgid", lambda gid: proc.calls.append(("setgid", gid)), raising=False)
    monkeypatch.setattr(os, "setuid", setuid, raising=False)
    return proc


def test_root_drops_groups_then_gid_then_uid(fake: SimpleNamespace) -> None:
    drop_to("grader")
    assert fake.calls == [("setgroups", ()), ("setgid", 1002), ("setuid", 1002)]
    assert fake.uid == 1002


def test_already_target_account_is_noop(fake: SimpleNamespace) -> None:
    fake.uid = 1002
    drop_to("grader")
    assert fake.calls == []


def test_other_unprivileged_account_fails_closed(fake: SimpleNamespace) -> None:
    fake.uid = 1001
    with pytest.raises(RuntimeError, match="^privilege separation requires root$"):
        drop_to("grader")
    assert fake.calls == []


def test_unknown_account(fake: SimpleNamespace) -> None:
    with pytest.raises(RuntimeError, match="^unknown account$"):
        drop_to("nobody-here")

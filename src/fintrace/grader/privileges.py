"""Irreversible switch of the current process to an unprivileged POSIX account (§9.5).

The trusted runner starts as root; the grader worker drops to the `grader` account and the
runner drops to `agent` before any policy code runs, so only the worker can read `private/`.
"""

import os


def drop_to(user: str) -> None:
    """Switch the real and effective UID/GID to `user`, clearing supplementary groups.

    Invariants: after return the process cannot regain root. A non-root process that already
    runs as `user` is left unchanged; any other non-root process raises RuntimeError, because
    it cannot honour the requested separation. The message never contains paths or IDs.
    """
    import pwd  # POSIX only; callers enable this through environment variables in Docker.

    try:
        entry = pwd.getpwnam(user)
    except KeyError:
        raise RuntimeError("unknown account") from None
    if os.geteuid() != 0:
        if os.getuid() == os.geteuid() == entry.pw_uid:
            return
        raise RuntimeError("privilege separation requires root")
    os.setgroups([])
    os.setgid(entry.pw_gid)
    os.setuid(entry.pw_uid)
    if os.getuid() != entry.pw_uid or os.geteuid() != entry.pw_uid:
        raise RuntimeError("privilege separation failed")

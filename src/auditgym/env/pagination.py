"""Opaque pagination cursors (SPEC §8.3).

Cursor = base64url (no padding) of canonical JSON ``{"e": episode_id[:8], "o": offset, "t": table}``.
Decoding is strict: the cursor must re-encode to exactly the same string, belong to the
current episode and table, and point strictly inside the table. Deterministic, stateless.
"""

from __future__ import annotations

import base64
import binascii
import json

from .errors import EnvError

DEFAULT_LIMIT = 50
MAX_LIMIT = 50
_MAX_CURSOR_LEN = 256


def _canonical(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def encode_cursor(table: str, offset: int, episode_id: str) -> str:
    """Encode a cursor for ``table`` starting at ``offset`` in the given episode."""
    raw = _canonical({"e": episode_id[:8], "o": offset, "t": table}).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(cursor: str, table: str, episode_id: str, total: int) -> int:
    """Return the offset of a valid cursor; raise ``EnvError("BAD_CURSOR")`` otherwise."""
    if not isinstance(cursor, str) or not 0 < len(cursor) <= _MAX_CURSOR_LEN:
        raise EnvError("BAD_CURSOR")
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        data = json.loads(raw.decode("utf-8"))
    except (binascii.Error, ValueError, UnicodeDecodeError):
        raise EnvError("BAD_CURSOR") from None
    if not isinstance(data, dict) or sorted(data) != ["e", "o", "t"]:
        raise EnvError("BAD_CURSOR")
    offset = data["o"]
    if type(offset) is not int or data["t"] != table or data["e"] != episode_id[:8]:
        raise EnvError("BAD_CURSOR")
    if not 0 <= offset < total or encode_cursor(table, offset, episode_id) != cursor:
        raise EnvError("BAD_CURSOR")
    return offset


def clamp_limit(limit: object) -> int:
    """Default 50, values above 50 are truncated to 50; non-int or < 1 → ``BAD_ARGS``."""
    if limit is None:
        return DEFAULT_LIMIT
    if type(limit) is not int or limit < 1:
        raise EnvError("BAD_ARGS")
    return min(limit, MAX_LIMIT)


def page(items: tuple, table: str, cursor: str | None, limit: object, episode_id: str) -> dict:
    """Return ``{"items", "next_cursor", "total"}`` in file order; last page has ``next_cursor=None``."""
    size = clamp_limit(limit)
    total = len(items)
    start = 0 if cursor is None else decode_cursor(cursor, table, episode_id, total)
    end = min(start + size, total)
    next_cursor = encode_cursor(table, end, episode_id) if end < total else None
    return {"items": list(items[start:end]), "next_cursor": next_cursor, "total": total}

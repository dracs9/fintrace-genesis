"""Draft 2020-12 validators for findings and trusted answer keys."""

import json
from functools import lru_cache
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from .normalize import Finding, Issue, canonical_id, clean, finding_from, issue_from


@lru_cache(maxsize=2)
def _validator(name: str) -> Draft202012Validator:
    path = Path(__file__).resolve().parents[3] / "schemas" / f"{name}.schema.json"
    return Draft202012Validator(json.loads(path.read_text(encoding="utf-8")))


def parse_findings(payload: object, episode_id: str) -> tuple[Finding, ...]:
    """Validate the submitted envelope and original limits before deduplicating IDs."""
    if not isinstance(payload, dict) or payload.keys() != {"episode_id", "findings"}:
        raise ValueError("invalid arguments")
    ep, rows = payload["episode_id"], payload["findings"]
    if not isinstance(ep, str) or clean(ep) != episode_id or not isinstance(rows, list) or len(rows) > 30:
        raise ValueError("invalid arguments")
    canonical = []
    for row in rows:
        if not isinstance(row, dict) or type(row.get("claimed_amount_minor")) is not int:
            raise ValueError("invalid arguments")
        item = row.copy()
        for key in ("type", "currency", "explanation"):
            value = row.get(key)
            if not isinstance(value, str) or (key == "explanation" and len(value) > 1000):
                raise ValueError("invalid arguments")
            item[key] = clean(value)
        for key in ("affected_ids", "evidence_ids"):
            values = row.get(key)
            if not isinstance(values, list) or len(values) > 10 or not all(isinstance(v, str) for v in values):
                raise ValueError("invalid arguments")
            item[key] = [canonical_id(v) for v in values]
        canonical.append(item)
    if not _validator("findings").is_valid({"episode_id": episode_id, "findings": canonical}):
        raise ValueError("invalid arguments")
    return tuple(finding_from(item) for item in canonical)


def parse_truth(payload: object, episode_id: str) -> tuple[Issue, ...]:
    """Reject corrupt answer keys, including duplicate issue IDs and non-int money."""
    try:
        _validator("truth").validate(payload)
    except ValidationError:
        raise ValueError("invalid arguments") from None
    assert isinstance(payload, dict)
    rows = payload["issues"]
    if (payload["episode_id"] != episode_id or len({row["issue_id"] for row in rows}) != len(rows)
            or any(type(row["expected_amount_minor"]) is not int for row in rows)):
        raise ValueError("invalid arguments")
    return tuple(issue_from(row) for row in rows)

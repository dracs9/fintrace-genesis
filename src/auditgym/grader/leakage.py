"""Offline verifier for agent-visible trajectory data. Run under the trusted evaluator UID."""

import argparse
import csv
import json
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .normalize import IssueType


def _leaves(value: object) -> Iterator[str | int]:
    if isinstance(value, dict):
        for key in sorted(value):
            yield key
            yield from _leaves(value[key])
    elif isinstance(value, list):
        for child in value:
            yield from _leaves(child)
    elif isinstance(value, str) or type(value) is int:
        yield value


def _public_values(directory: Path) -> set[str | int]:
    values: set[str | int] = {kind.value for kind in IssueType}
    for filename in ("manifest.json", "policy.json", "invoices.jsonl", "payments.jsonl",
                     "ledger.csv", "fx.csv"):
        with (directory / filename).open(encoding="utf-8", newline="") as handle:
            if filename.endswith(".json"):
                values.update(_leaves(json.load(handle)))
            elif filename.endswith(".jsonl"):
                for line in handle:
                    values.update(_leaves(json.loads(line)))
            else:
                for row in csv.DictReader(handle):
                    values.update(_leaves(row))
                    if filename == "fx.csv":
                        values.add("fx:{date}:{source_currency}:{target_currency}".format(**row))
    for value in sorted(v for v in values if isinstance(v, str)):
        if value.startswith("D-"):
            values.add("doc:" + value)
        elif value.startswith("R-"):
            values.add("ledger:" + value)
    return values


_MARKER = re.compile(r"issue_id|iss_|inject|truth|private|provenance|seed|auditgym_salt", re.I)
_PATH = re.compile(r"(?:[a-z]:[\\/]|\\\\|(?:^|[\s(=\x22'])/(?:[\w.-]+)|"
                   r"(?:^|[\s(=\x22'])\.\.?[\\/][\w.-]+)", re.I)
_RELATIVE = re.compile(r"(?:^|[\s(=\x22'])[a-z_][\w.-]*[\\/][\w.-]+", re.I)
_PRIVATE_FIELDS = frozenset(("expected_amount_minor", "evidence_sets", "affected_ids",
                            "causal_note", "stream_keys", "mutations", "matches", "config_hash"))


@dataclass(frozen=True)
class LeakIndex:
    secrets: re.Pattern[str] | None
    seed: int
    public: frozenset[str | int]

    @classmethod
    def from_episode(cls, directory: Path) -> "LeakIndex":
        """Index truth-only strings using the six public files as the comparison corpus."""
        public = _public_values(directory / "open")
        truth = json.loads((directory / "private/truth.json").read_bytes())
        provenance = json.loads((directory / "private/provenance.json").read_bytes())
        seed = provenance["seed"]
        private = {value for data in (truth, provenance) for value in _strings(data)
                   if value and value not in public}
        private.update(row["issue_id"] for row in truth["issues"])
        if salt := os.environ.get("AUDITGYM_SALT"):
            private.add(salt)
        pattern = "|".join(re.escape(value) for value in sorted(private, key=lambda s: (-len(s), s)))
        return cls(re.compile(pattern) if pattern else None, seed, frozenset(public))

    def check(self, value: object) -> None:
        """Scan nested keys and values without printing secret material on failure."""
        for leaf in _leaves(value):
            if isinstance(leaf, str):
                if _MARKER.search(leaf) or leaf.casefold() in _PRIVATE_FIELDS or _PATH.search(leaf):
                    raise ValueError("restricted data in trajectory")
                if leaf in self.public:
                    continue
                if (_RELATIVE.search(leaf) or (self.secrets and self.secrets.search(leaf))
                        or re.search(rf"(?<!\w){self.seed}(?!\w)", leaf)):
                    raise ValueError("restricted data in trajectory")
            elif leaf == self.seed and leaf not in self.public and str(leaf) not in self.public:
                raise ValueError("restricted data in trajectory")


def _strings(value: object) -> Iterator[str]:
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _strings(value[key])
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


def scan_trajectory(episode: Path, trajectory: Path) -> int:
    """Scan every visible observation/info/error for this episode in a JSONL run."""
    episode_id = json.loads((episode / "open/manifest.json").read_bytes())["episode_id"]
    index = LeakIndex.from_episode(episode)
    count = 0
    with trajectory.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            record = json.loads(line)
            if record["episode_id"] != episode_id:
                continue
            if not ("observation" in record or "obs" in record):
                raise ValueError(f"missing observation on line {number}")
            for field in ("observation", "obs", "info", "error"):
                if field in record:
                    try:
                        index.check(record[field])
                    except ValueError:
                        raise ValueError(f"restricted data in {field} on line {number}") from None
            count += 1
    if not count:
        raise ValueError("no matching trajectory records")
    return count


def main() -> None:
    """Verify one episode against a trajectory file, reporting only the row count."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("trajectory", type=Path)
    args = parser.parse_args()
    print(scan_trajectory(args.episode, args.trajectory))


if __name__ == "__main__":
    main()

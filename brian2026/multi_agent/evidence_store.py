"""Append-only, hash-chained JSONL store for decision records (local; no database writes yet).

Each line carries the SHA-256 of the previous line, so an edited or removed record breaks ``verify``.
Records are never rewritten; a correction is a new record.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

from .schemas import DecisionRecord

GENESIS = "0" * 64


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


class EvidenceStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _last_hash(self) -> str:
        last = GENESIS
        for entry in self.entries():
            last = entry["hash"]
        return last

    def append(self, record: DecisionRecord) -> str:
        body = record.to_dict()
        prev = self._last_hash()
        digest = hashlib.sha256((prev + _canonical(body)).encode("utf-8")).hexdigest()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(_canonical({"prev": prev, "hash": digest, "record": body}) + "\n")
        return digest

    def entries(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)

    def verify(self) -> bool:
        prev = GENESIS
        for entry in self.entries():
            expected = hashlib.sha256((prev + _canonical(entry["record"])).encode("utf-8")).hexdigest()
            if entry.get("prev") != prev or entry.get("hash") != expected:
                return False
            prev = entry["hash"]
        return True

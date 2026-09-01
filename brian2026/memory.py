from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable
import json
import math
import time


class EpisodicMemory:
    """Append-only experience memory plus simple specialist reliability.

    JSONL keeps the first version transparent and recoverable.  A future vector
    store can sit behind this interface without changing the decision engine.
    """

    def __init__(self, root: str | Path = "runtime/brian2026") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root / "events.jsonl"
        self.reliability_path = self.root / "reliability.json"

    def append(self, kind: str, payload: Dict[str, Any]) -> None:
        rec = {"ts": time.time(), "kind": kind, "payload": payload}
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")

    def tail(self, limit: int = 500, kind: str | None = None) -> list[Dict[str, Any]]:
        if not self.events_path.exists():
            return []
        rows: list[Dict[str, Any]] = []
        with self.events_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if kind is None or rec.get("kind") == kind:
                    rows.append(rec)
        return rows[-max(0, int(limit)):]

    def similar(self, symbol: str, regime: str, limit: int = 50) -> list[Dict[str, Any]]:
        scored: list[tuple[float, Dict[str, Any]]] = []
        now = time.time()
        for rec in self.tail(limit=3000, kind="decision_outcome"):
            p = rec.get("payload", {})
            score = 0.0
            if p.get("symbol") == symbol:
                score += 2.0
            if str(p.get("regime", "")).upper() == str(regime).upper():
                score += 1.5
            age_days = max(0.0, (now - float(rec.get("ts", now))) / 86400.0)
            score += math.exp(-age_days / 30.0)
            scored.append((score, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[:limit]]

    def _load_reliability(self) -> Dict[str, Dict[str, float]]:
        if not self.reliability_path.exists():
            return {}
        try:
            data = json.loads(self.reliability_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_reliability(self, data: Dict[str, Dict[str, float]]) -> None:
        tmp = self.reliability_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.reliability_path)

    def specialist_weight(self, name: str, default: float = 1.0) -> float:
        row = self._load_reliability().get(name)
        if not row:
            return default
        n = float(row.get("n", 0.0))
        wins = float(row.get("wins", 0.0))
        # Beta(3,3) shrinkage: avoids overreacting to a handful of lucky trades.
        wr = (wins + 3.0) / (n + 6.0)
        return max(0.55, min(1.45, default * (0.5 + wr)))

    def update_specialists(self, votes: Iterable[Dict[str, Any]], won: bool, executed_action: str = "BUY") -> None:
        data = self._load_reliability()
        for vote in votes:
            name = str(vote.get("name", "unknown"))
            action = str(vote.get("action", "WAIT"))
            if action == "WAIT":
                continue
            row = data.setdefault(name, {"n": 0.0, "wins": 0.0})
            row["n"] = float(row.get("n", 0.0)) + 1.0
            vote_correct = (won and action == executed_action) or ((not won) and action != executed_action)
            if vote_correct:
                row["wins"] = float(row.get("wins", 0.0)) + 1.0
        self._save_reliability(data)

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict
import json
import time

from .memory import EpisodicMemory


@dataclass(slots=True)
class ExperimentSpec:
    hypothesis: str
    target: str
    change: Dict[str, Any]
    evidence_count: int
    expected_effect: str

    def to_dict(self):
        return asdict(self)


class Researcher:
    """RD-Agent-inspired bounded research loop.

    Foundation phase proposes measurable parameter experiments from real loss
    clusters. Arbitrary self-modifying code is intentionally deferred until a
    sandbox + test + promotion pipeline exists.
    """

    def __init__(self, memory: EpisodicMemory, root: str | Path = "runtime/brian2026") -> None:
        self.memory = memory
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.queue = self.root / "experiments.jsonl"

    def propose(self, min_cluster: int = 4) -> list[ExperimentSpec]:
        rows = self.memory.tail(limit=2000, kind="decision_outcome")
        clusters: Counter[tuple[str, str]] = Counter()
        for rec in rows:
            p = rec.get("payload", {})
            if float(p.get("pnl_usd", 0.0)) >= 0:
                continue
            regime = str(p.get("regime", "UNKNOWN")).upper()
            reason = str(p.get("exit_reason", "loss")).upper()
            clusters[(regime, reason)] += 1

        specs: list[ExperimentSpec] = []
        for (regime, reason), n in clusters.most_common(8):
            if n < min_cluster:
                continue
            if "SL" in reason:
                change = {"entry_confidence_delta": +0.03, "size_scale": 0.90}
                effect = "fewer weak entries and smaller loss exposure"
            elif "TIME" in reason:
                change = {"entry_delay_bars": +1, "min_consensus_delta": +0.03}
                effect = "reduce early entries that fail to follow through"
            else:
                change = {"min_consensus_delta": +0.02}
                effect = "demand stronger agreement in this loss cluster"
            specs.append(ExperimentSpec(
                hypothesis=f"{regime}/{reason} losses may be reduced by stricter entry quality",
                target=f"regime:{regime}",
                change=change,
                evidence_count=n,
                expected_effect=effect,
            ))

        for spec in specs:
            rec = {"ts": time.time(), "status": "PROPOSED", **spec.to_dict()}
            with self.queue.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return specs

"""End-to-end dry run with mock providers (free, offline).

    python -m brian2026.multi_agent.dry_run [--store PATH]

Two synthetic candidates: one whose measured edge clears cost (exercises all six roles and the debate)
and one with the real-world situation of 2026-11-05 (every measured segment below cost), which must
end in NO_TRADE without any model call. All values are synthetic test fixtures, not market data.

    python -m brian2026.multi_agent.dry_run --live-canary

Live canary: one small structured call per provider. It runs only when BRIAN_MULTI_AGENT_ENABLED=true,
BRIAN_AGENT_COST_LIMIT_USD>0 and both keys are set; otherwise it reports why it did not run.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .agents import facts_for, run_model_role, SYSTEM_STRATEGIST
from .config import MultiAgentConfig
from .coordinator import Coordinator
from .evidence_store import EvidenceStore
from .providers import CallCounter, ClaudeProvider, CostLedger, GrokProvider, MockProvider
from .schemas import Candidate, MeasuredEdge, Source, utc_iso

DECISION_TIME = datetime(2026, 11, 5, 15, 0, tzinfo=timezone.utc)


def fixtures() -> list[tuple[Candidate, MeasuredEdge]]:
    t = DECISION_TIME
    source = Source("https://www.sec.gov/example-8k", "OFFICIAL_PRIMARY", utc_iso(t - timedelta(minutes=12)), None,
                    "synthetic: company files an 8-K announcing a buyback")
    clears = Candidate("dry-run-1", "equity:EXAMPLE", "equity", utc_iso(t - timedelta(minutes=4)), 101.5,
                       utc_iso(t - timedelta(seconds=40)), 180.0, 1, news_class="OFFICIAL", volume_ratio=3.1,
                       spread_bps=4.0, sources=(source,), metadata={"move_started_at": utc_iso(t - timedelta(minutes=10))})
    clears_edge = MeasuredEdge("SYNTHETIC/equity/1/OFFICIAL", 400, 40.0, 4.0, 18.0, utc_iso(t - timedelta(minutes=40)))
    below = Candidate("dry-run-2", "commodity:BRENT", "commodity", utc_iso(t - timedelta(minutes=3)), 74.2,
                      utc_iso(t - timedelta(seconds=30)), -120.0, -1, news_class="OFFICIAL", volume_ratio=2.4,
                      spread_bps=3.0)
    # Shape of the best real segment on 2026-11-05 (net about -7 bps after shrinkage, SE and cost).
    below_edge = MeasuredEdge("MULTIASSET_EVENT_REACTION/commodity/-1/OFFICIAL", 159, 8.0, 4.0, 8.0,
                              utc_iso(t - timedelta(minutes=40)))
    return [(clears, clears_edge), (below, below_edge)]


def _mock(name: str, stance: str) -> MockProvider:
    def answer(request):
        return {"stance": stance, "reason": f"mock {name} round answer",
                "evidence": ["synthetic evidence"], "counter_evidence": ["synthetic counter-evidence"],
                "sources": [{"url": "https://www.sec.gov/example-8k", "source_type": "OFFICIAL_PRIMARY",
                             "published_at": utc_iso(DECISION_TIME - timedelta(minutes=12)),
                             "claim": "synthetic"}]}
    return MockProvider(name, answer)


def dry_run(store_path: Path) -> list[dict]:
    config = MultiAgentConfig(enabled=True, cost_limit_usd=0.0)
    coordinator = Coordinator(config, _mock("claude", "bullish"), _mock("grok", "bullish"))
    store = EvidenceStore(store_path)
    out = []
    for candidate, edge in fixtures():
        record = coordinator.run(candidate, DECISION_TIME, edge, opens_today=0, day_return_pct=0.0)
        store.append(record)
        out.append({"candidate_id": record.candidate_id, "decision": record.decision,
                    "reasons": list(record.decision_reasons), "model_calls": len(record.usage),
                    "debate_rounds": record.debate_rounds, "expected_net_edge_bps": record.risk.expected_net_edge_bps,
                    "provider_cost_usd": record.provider_cost_usd, "mock": record.mock,
                    "execution_allowed": record.execution_allowed})
    if not store.verify():
        raise RuntimeError("evidence chain failed verification")
    return out


def live_canary() -> dict:
    config = MultiAgentConfig.from_env()
    if not config.paid_calls_allowed:
        return {"status": "NOT_RUN", "reason": "needs BRIAN_MULTI_AGENT_ENABLED=true, no STOP and BRIAN_AGENT_COST_LIMIT_USD>0"}
    ledger = CostLedger(config.cost_limit_usd)
    candidate, edge = fixtures()[0]
    facts = facts_for(candidate, DECISION_TIME, edge)
    result = {}
    for provider in (ClaudeProvider(config.claude_model, ledger=ledger), GrokProvider(config.grok_model, ledger=ledger)):
        assessment, response = run_model_role(f"canary_{provider.name}", provider, SYSTEM_STRATEGIST, facts,
                                              DECISION_TIME, CallCounter(1), True)
        result[provider.name] = {"status": response.status, "error_class": response.error_class,
                                 "model": response.model, "cost_usd": response.estimated_cost_usd,
                                 "schema_valid_stance": assessment.stance}
    result["spent_usd"] = round(ledger.spent_usd, 6)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", type=Path, default=None)
    parser.add_argument("--live-canary", action="store_true")
    args = parser.parse_args(argv)
    if args.live_canary:
        print(json.dumps(live_canary(), indent=2))
        return 0
    store = args.store or Path(tempfile.mkdtemp(prefix="brian-ma-")) / "decisions.jsonl"
    print(json.dumps({"store": str(store), "results": dry_run(store)}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

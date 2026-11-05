"""Live shadow runner: CORE candidates -> six roles -> append-only decision table.

    python -m brian2026.multi_agent.live [--dry-run] [--window-minutes 15]

Reads (PostgREST, service key from the environment, never logged):
  brian_multiasset_alpha_decisions   OPEN_* event-reaction signals of the last window, no veto
  brian_intel_events                 the linked news (URL, publication time, trust class)
  brian_treasury_segment_edge_cache  measured after-cost evidence per segment
  brian_treasury_shadow_*            today's opens and equity, for the daily risk limits
  brian_multi_agent_decisions        candidates already decided (cross-run de-duplication)
Writes one row per decided candidate to brian_multi_agent_decisions (insert-only table), unless
--dry-run. Crypto ALPHA signals are not routed here: they are microstructure signals without a news
catalyst, so a news analyst adds nothing to them.

Model calls follow MultiAgentConfig: off / budget 0 by default, so a default run records the
deterministic verdicts only and spends nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence

from . import agents
from .config import MultiAgentConfig
from .coordinator import Coordinator
from .providers import ClaudeProvider, CostLedger, GrokProvider
from .schemas import Candidate, DecisionRecord, MeasuredEdge, Source, finite_or_none, parse_utc, utc_iso

LANE = "MULTIASSET_EVENT_REACTION"
TABLE = "brian_multi_agent_decisions"
TRUST_TO_SOURCE_TYPE = {"OFFICIAL_PRIMARY": "OFFICIAL_PRIMARY", "INDEPENDENT_PROFESSIONAL": "INDEPENDENT_PROFESSIONAL"}


# --- pure mapping (shared by the live source and offline snapshots) --------------------------------
def news_class(event_ids: Sequence[str], trust_by_id: Mapping[str, str]) -> str:
    """Mirror of newsClassOf in evolution_treasury_unified.ts / public.brian_news_class_v1."""
    if not event_ids:
        return "NONE"
    classes = [trust_by_id[i] for i in event_ids if trust_by_id.get(i)]
    if "OFFICIAL_PRIMARY" in classes:
        return "OFFICIAL"
    if "INDEPENDENT_PROFESSIONAL" in classes:
        return "PROFESSIONAL"
    return "UNVERIFIED" if classes else "UNKNOWN"


def candidates_from_rows(decisions: Iterable[Mapping[str, Any]], events: Iterable[Mapping[str, Any]]) -> list[Candidate]:
    by_id = {str(e.get("event_id")): e for e in events}
    trust = {k: str(v.get("trust_class") or "") for k, v in by_id.items()}
    out: list[Candidate] = []
    for row in decisions:
        action, direction = str(row.get("action") or ""), int(row.get("direction") or 0)
        if row.get("veto_reason") or (action, direction) not in {("OPEN_LONG", 1), ("OPEN_SHORT", -1)}:
            continue
        meta = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
        ret_1h = finite_or_none(meta.get("return_1h"))
        linked = [str(i) for i in (row.get("linked_event_ids") or [])]
        sources = []
        for event_id in linked[:20]:
            event = by_id.get(event_id)
            if not event or not event.get("provenance_uri"):
                continue
            sources.append(Source(
                url=str(event["provenance_uri"]),
                source_type=TRUST_TO_SOURCE_TYPE.get(str(event.get("trust_class") or ""), "UNKNOWN"),
                published_at=_iso(event.get("published_at") or event.get("first_observed_at")),
                accessed_at=_iso(event.get("captured_at")),
                claim=str(event.get("claim") or "")[:500],
            ))
        out.append(Candidate(
            candidate_id=str(row["decision_id"]), asset=str(row["asset_id"]), asset_class=str(row["asset_class"]),
            observed_at=_iso(row.get("observed_at")) or "", reference_price=finite_or_none(row.get("observed_reference_price")),
            price_observed_at=_iso(row.get("provider_time")), move_bps=None if ret_1h is None else ret_1h * 1e4,
            direction=direction, signal_lane=LANE, news_class=news_class(linked, trust), volume_ratio=None,
            spread_bps=None, sources=tuple(sources),
            metadata={"engine_version": meta.get("version"), "reaction_score": meta.get("reaction_score"),
                      "linked_themes": meta.get("linked_themes"), "session_state": row.get("session_state")},
        ))
    return out


def edges_from_rows(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str, int, str], MeasuredEdge]:
    out = {}
    for r in rows:
        try:
            key = (str(r["lane"]), str(r["asset_class"]), int(r["direction"]), str(r["news_class"]))
        except (KeyError, TypeError, ValueError):
            continue
        out[key] = MeasuredEdge("/".join(map(str, key)), int(r.get("n") or 0), finite_or_none(r.get("gross_bps")),
                                finite_or_none(r.get("se_bps")), finite_or_none(r.get("cost_bps")), _iso(r.get("computed_at")))
    return out


def edge_for(candidate: Candidate, edges: Mapping[tuple[str, str, int, str], MeasuredEdge]) -> MeasuredEdge | None:
    return edges.get((candidate.signal_lane, candidate.asset_class, candidate.direction, candidate.news_class))


def row_for(record: DecisionRecord) -> dict[str, Any]:
    return {
        "candidate_id": record.candidate_id, "decision_time": record.decision_time_utc, "asset": record.asset,
        "asset_class": record.asset_class, "decision": record.decision,
        "expected_net_edge_bps": record.risk.expected_net_edge_bps, "provider_cost_usd": record.provider_cost_usd,
        "debate_rounds": record.debate_rounds, "mock": record.mock, "schema_version": record.schema_version,
        "record": json.loads(json.dumps(record.to_dict(), default=str)),
    }


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    parsed = parse_utc(str(value).replace(" ", "T"))
    return utc_iso(parsed) if parsed else None


# --- PostgREST access --------------------------------------------------------------------------------
class CoreRest:
    def __init__(self, url: str, key: str, session: Any = None, timeout: float = 20.0) -> None:
        import requests  # runtime dependency already in requirements.txt
        self.base = url.rstrip("/") + "/rest/v1/"
        self.session = session or requests.Session()
        self.session.headers.update({"apikey": key, "Authorization": f"Bearer {key}"})
        self.timeout = timeout

    def get(self, table: str, params: Mapping[str, str]) -> list[dict[str, Any]]:
        response = self.session.get(self.base + table, params=dict(params), timeout=self.timeout)
        if response.status_code >= 400:
            raise RuntimeError(f"CORE read {table} failed: HTTP {response.status_code}")
        return response.json()

    def insert(self, table: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        response = self.session.post(
            self.base + table, params={"on_conflict": "candidate_id"}, json=rows, timeout=self.timeout,
            headers={"Prefer": "resolution=ignore-duplicates,return=minimal"},
        )
        if response.status_code >= 400:
            raise RuntimeError(f"CORE insert {table} failed: HTTP {response.status_code}")


def _in(values: Iterable[str]) -> str:
    return "in.(" + ",".join(values) + ")"


def load_inputs(core: CoreRest, now: datetime, window_minutes: int) -> dict[str, Any]:
    since = utc_iso(now - timedelta(minutes=window_minutes))
    decisions = core.get("brian_multiasset_alpha_decisions", {
        "select": "decision_id,observed_at,provider_time,asset_id,asset_class,action,direction,"
                  "observed_reference_price,veto_reason,session_state,linked_event_ids,metadata",
        "observed_at": f"gte.{since}", "action": "like.OPEN*", "veto_reason": "is.null",
        "order": "observed_at.desc", "limit": "200"})
    event_ids = sorted({str(i) for d in decisions for i in (d.get("linked_event_ids") or [])})[:400]
    events = core.get("brian_intel_events", {
        "select": "event_id,published_at,first_observed_at,captured_at,trust_class,provenance_uri,claim",
        "event_id": _in(event_ids)}) if event_ids else []
    edges = core.get("brian_treasury_segment_edge_cache", {"select": "*", "lane": f"eq.{LANE}"})
    day_start = now.strftime("%Y-%m-%dT00:00:00Z")
    opens = core.get("brian_treasury_shadow_actions", {"select": "action_id", "kind": "eq.OPEN",
                                                        "observed_at": f"gte.{day_start}", "limit": "1000"})
    start = core.get("brian_treasury_shadow_snapshots", {"select": "equity_usd", "observed_at": f"lt.{day_start}",
                                                          "order": "observed_at.desc", "limit": "1"})
    latest = core.get("brian_treasury_shadow_latest", {"select": "equity_usd", "limit": "1"})
    ids = [str(d["decision_id"]) for d in decisions]
    done = core.get(TABLE, {"select": "candidate_id", "candidate_id": _in(ids)}) if ids else []
    return {"decisions": decisions, "events": events, "edges": edges, "opens_today": len(opens),
            "day_start_equity": finite_or_none(start[0]["equity_usd"]) if start else None,
            "equity": finite_or_none(latest[0]["equity_usd"]) if latest else None,
            "already_decided": {str(r["candidate_id"]) for r in done}}


def decide_all(inputs: Mapping[str, Any], now: datetime, coordinator: Coordinator) -> tuple[list[DecisionRecord], dict[str, int]]:
    candidates = [c for c in candidates_from_rows(inputs["decisions"], inputs["events"])
                  if c.candidate_id not in inputs.get("already_decided", set())]
    # one candidate per asset and direction: the newest signal (older duplicates add calls, not information)
    newest: dict[tuple[str, int], Candidate] = {}
    for c in sorted(candidates, key=lambda c: c.observed_at):
        newest[(c.asset, c.direction)] = c
    kept, dropped = agents.scan(newest.values(), now)
    edges = edges_from_rows(inputs["edges"])
    start, equity = inputs.get("day_start_equity"), inputs.get("equity")
    day_return = equity / start - 1 if start and equity and start > 0 else None
    records = [coordinator.run(c, now, edge_for(c, edges), opens_today=inputs.get("opens_today"),
                               day_return_pct=day_return) for c in kept]
    stats = {"signals": len(inputs["decisions"]), "new_candidates": len(candidates), "scanned": len(kept),
             **{f"scanner_{reason}": sum(1 for _, r in dropped if r == reason) for reason in {r for _, r in dropped}}}
    return records, stats


def build_coordinator(config: MultiAgentConfig) -> Coordinator:
    ledger = CostLedger(config.cost_limit_usd)
    return Coordinator(config, ClaudeProvider(config.claude_model, ledger=ledger, timeout_seconds=config.provider_timeout_seconds),
                       GrokProvider(config.grok_model, ledger=ledger, timeout_seconds=config.provider_timeout_seconds))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BRIAN multi-agent live shadow runner")
    parser.add_argument("--dry-run", action="store_true", help="read CORE, decide, print; write nothing")
    parser.add_argument("--window-minutes", type=int, default=15)
    args = parser.parse_args(argv)
    config = MultiAgentConfig.from_env()
    if config.operator_stop:
        print(json.dumps({"status": "STOPPED", "reason": "BRIAN_MULTI_AGENT_STOP"}))
        return 0
    url, key = os.environ.get("BRIAN_CORE_SUPABASE_URL", ""), os.environ.get("BRIAN_CORE_SUPABASE_SECRET_KEY", "")
    if not url.startswith("https://") or not key.strip():
        print(json.dumps({"status": "NOT_CONFIGURED", "reason": "BRIAN_CORE_SUPABASE_URL / _SECRET_KEY missing"}))
        return 1
    now = datetime.now(timezone.utc)
    core = CoreRest(url, key)
    records, stats = decide_all(load_inputs(core, now, max(1, min(60, args.window_minutes))), now, build_coordinator(config))
    if not args.dry_run:
        core.insert(TABLE, [row_for(r) for r in records])
    summary = {"status": "OK", "dry_run": args.dry_run, "decided_at": utc_iso(now), **stats,
               "decisions": {d: sum(1 for r in records if r.decision == d) for d in {r.decision for r in records}},
               "provider_cost_usd": round(sum(r.provider_cost_usd for r in records), 6),
               "paid_calls_allowed": config.paid_calls_allowed}
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

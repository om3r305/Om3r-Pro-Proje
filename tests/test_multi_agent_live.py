import json
from datetime import datetime, timedelta, timezone

import pytest

from brian2026.multi_agent import MultiAgentConfig
from brian2026.multi_agent import live
from brian2026.multi_agent.coordinator import Coordinator
from brian2026.multi_agent.providers import MockProvider
from brian2026.multi_agent.schemas import utc_iso

NOW = datetime(2026, 11, 5, 22, 20, tzinfo=timezone.utc)


def signal(i="d1", asset="commodity:BRENT", direction=1, ret_1h=0.0125, minutes_ago=4, events=("e1", "e2"), **kw):
    row = {"decision_id": i, "observed_at": utc_iso(NOW - timedelta(minutes=minutes_ago)).replace("T", " "),
           "provider_time": utc_iso(NOW - timedelta(minutes=minutes_ago + 2)), "asset_id": asset,
           "asset_class": asset.split(":")[0], "action": "OPEN_LONG" if direction == 1 else "OPEN_SHORT",
           "direction": direction, "observed_reference_price": "99.25", "veto_reason": None,
           "session_state": "REGULAR", "linked_event_ids": list(events),
           "metadata": {"return_1h": ret_1h, "version": "v6", "reaction_score": 1}}
    row.update(kw)
    return row


EVENTS = [
    {"event_id": "e1", "published_at": utc_iso(NOW - timedelta(minutes=30)), "trust_class": "OFFICIAL_PRIMARY",
     "provenance_uri": "https://www.opec.org/press", "claim": "statement", "captured_at": utc_iso(NOW - timedelta(minutes=29))},
    {"event_id": "e2", "published_at": utc_iso(NOW - timedelta(minutes=20)), "trust_class": "SOCIAL",
     "provenance_uri": "https://x.com/a/status/1", "claim": "post"},
]
EDGE_ROW = {"lane": live.LANE, "asset_class": "commodity", "direction": 1, "news_class": "OFFICIAL", "n": 400,
            "gross_bps": 40.0, "se_bps": 4.0, "cost_bps": 14.0, "hit_rate": 0.55,
            "computed_at": utc_iso(NOW - timedelta(minutes=50))}


def inputs(decisions, edges=(EDGE_ROW,), **kw):
    base = {"decisions": list(decisions), "events": EVENTS, "edges": list(edges), "opens_today": 0,
            "day_start_equity": 10_000.0, "equity": 10_000.0, "already_decided": set()}
    base.update(kw)
    return base


def mocks(stance="bullish"):
    answer = {"stance": stance, "reason": "r", "evidence": [], "counter_evidence": [], "sources": []}
    return MockProvider("claude", lambda r: answer), MockProvider("grok", lambda r: answer)


def test_news_class_mirrors_the_treasury_rule():
    trust = {"a": "OFFICIAL_PRIMARY", "b": "INDEPENDENT_PROFESSIONAL", "c": "SOCIAL"}
    assert live.news_class([], trust) == "NONE"
    assert live.news_class(["c", "a"], trust) == "OFFICIAL"
    assert live.news_class(["b", "c"], trust) == "PROFESSIONAL"
    assert live.news_class(["c"], trust) == "UNVERIFIED"
    assert live.news_class(["zzz"], trust) == "UNKNOWN"


def test_candidate_mapping_keeps_point_in_time_fields_and_sources():
    (c,) = live.candidates_from_rows([signal()], EVENTS)
    assert c.candidate_id == "d1" and c.move_bps == pytest.approx(125) and c.news_class == "OFFICIAL"
    assert c.observed_at.endswith("Z") and c.price_observed_at == utc_iso(NOW - timedelta(minutes=6))
    assert [s.source_type for s in c.sources] == ["OFFICIAL_PRIMARY", "UNKNOWN"]
    assert live.candidates_from_rows([signal(veto_reason="FEED_CONTRACT_ROLL")], EVENTS) == []
    assert live.candidates_from_rows([signal(action="OPEN_SHORT")], EVENTS) == []  # action/direction mismatch


def test_decide_all_dedupes_scans_and_uses_the_measured_segment():
    claude, grok = mocks()
    coord = Coordinator(MultiAgentConfig(enabled=True), claude, grok)
    rows = [signal("old", minutes_ago=9), signal("new", minutes_ago=3), signal("quiet", asset="etf:XLE", ret_1h=0.003),
            signal("done", asset="commodity:WTI")]
    records, stats = live.decide_all(inputs(rows, already_decided={"done"}), NOW, coord)
    assert [r.candidate_id for r in records] == ["new"]
    assert records[0].decision == "BUY_CANDIDATE" and records[0].risk.expected_net_edge_bps == pytest.approx(14.0)
    assert stats["scanner_MOVE_NOT_UNUSUAL"] == 1 and stats["new_candidates"] == 3


def test_below_cost_segment_and_missing_segment_spend_nothing():
    claude, grok = mocks()
    coord = Coordinator(MultiAgentConfig(enabled=True), claude, grok)
    losing = dict(EDGE_ROW, gross_bps=8.0)
    (rec,), _ = live.decide_all(inputs([signal()], edges=[losing]), NOW, coord)
    assert rec.decision == "NO_TRADE" and claude.requests == []
    (rec,), _ = live.decide_all(inputs([signal("x2")], edges=[]), NOW, coord)
    assert rec.decision == "INSUFFICIENT_DATA" and claude.requests == []


def test_row_for_is_json_and_never_allows_execution():
    claude, grok = mocks()
    (rec,), _ = live.decide_all(inputs([signal()]), NOW, Coordinator(MultiAgentConfig(enabled=True), claude, grok))
    row = live.row_for(rec)
    json.dumps(row)
    assert row["record"]["execution_allowed"] is False and row["record"]["shadow_only"] is True
    assert row["decision"] == "BUY_CANDIDATE" and row["mock"] is True


class FakeResponse:
    def __init__(self, status, body=None):
        self.status_code, self._body = status, body

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, status=200, body=()):
        self.headers, self.calls, self.status, self.body = {}, [], status, list(body)

    def get(self, url, params, timeout):
        self.calls.append(("GET", url, params))
        return FakeResponse(self.status, self.body)

    def post(self, url, params, json, timeout, headers):
        self.calls.append(("POST", url, params, json, headers))
        return FakeResponse(self.status)


def test_core_rest_inserts_idempotently_and_errors_do_not_echo_the_key():
    session = FakeSession()
    core = live.CoreRest("https://example.supabase.co/", "service-key-not-real", session=session)
    core.insert(live.TABLE, [{"candidate_id": "a"}])
    _, url, params, _, headers = session.calls[0]
    assert url.endswith("/rest/v1/brian_multi_agent_decisions") and params == {"on_conflict": "candidate_id"}
    assert "ignore-duplicates" in headers["Prefer"]
    bad = live.CoreRest("https://example.supabase.co", "service-key-not-real", session=FakeSession(status=401))
    with pytest.raises(RuntimeError) as err:
        bad.get("t", {})
    assert "not-real" not in str(err.value)


def test_main_refuses_without_core_configuration(monkeypatch, capsys):
    monkeypatch.delenv("BRIAN_CORE_SUPABASE_URL", raising=False)
    monkeypatch.delenv("BRIAN_CORE_SUPABASE_SECRET_KEY", raising=False)
    assert live.main(["--dry-run"]) == 1
    assert "NOT_CONFIGURED" in capsys.readouterr().out
    monkeypatch.setenv("BRIAN_MULTI_AGENT_STOP", "true")
    assert live.main([]) == 0
    assert "STOPPED" in capsys.readouterr().out

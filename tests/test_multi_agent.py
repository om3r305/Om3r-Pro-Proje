import dataclasses
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from brian2026.multi_agent import Coordinator, Decision, MultiAgentConfig, UnsafeConfigError
from brian2026.multi_agent import agents
from brian2026.multi_agent.dry_run import DECISION_TIME, dry_run, fixtures, live_canary
from brian2026.multi_agent.evaluation import compare_groups
from brian2026.multi_agent.evidence_store import EvidenceStore
from brian2026.multi_agent.providers import (
    CallCounter, ClaudeProvider, CostLedger, GrokProvider, MockProvider, ProviderRequest,
)
from brian2026.multi_agent.schemas import (
    ASSESSMENT_JSON_SCHEMA, DecisionRecord, MeasuredEdge, RiskVerdict, Source, Verification, utc_iso,
)

T = DECISION_TIME
GOOD = {"stance": "bullish", "reason": "r", "evidence": ["e"], "counter_evidence": ["c"],
        "sources": [{"url": "https://example.gov/a", "source_type": "OFFICIAL_PRIMARY",
                     "published_at": utc_iso(T - timedelta(minutes=5)), "claim": "x"}]}


def request(**kw):
    return ProviderRequest(agent="t", system="s", prompt="p", json_schema=ASSESSMENT_JSON_SCHEMA, **kw)


def mock(name, answer=GOOD):
    return MockProvider(name, answer if callable(answer) else (lambda r: answer))


def coordinator(claude=None, grok=None, **cfg):
    return Coordinator(MultiAgentConfig(enabled=True, **cfg), claude or mock("claude"), grok or mock("grok"))


def good_case():
    return fixtures()[0]


# --- config ----------------------------------------------------------------------------------------
def test_config_defaults_are_off_shadow_only_and_free():
    c = MultiAgentConfig.from_env({})
    assert (c.enabled, c.shadow_only, c.real_trading_enabled, c.cost_limit_usd) == (False, True, False, 0.0)
    assert c.paid_calls_allowed is False
    assert c.max_debate_rounds == 2 and c.max_provider_calls_per_candidate == 6


@pytest.mark.parametrize("env", [{"BRIAN_REAL_TRADING_ENABLED": "true"}, {"BRIAN_MULTI_AGENT_SHADOW_ONLY": "false"},
                                 {"BRIAN_AGENT_COST_LIMIT_USD": "-1"}, {"BRIAN_AGENT_COST_LIMIT_USD": "nan"},
                                 {"BRIAN_MULTI_AGENT_ENABLED": "maybe"}])
def test_config_refuses_unsafe_or_malformed_values(env):
    with pytest.raises(UnsafeConfigError):
        MultiAgentConfig.from_env(env)


def test_config_clamps_rounds_and_calls_and_stop_blocks_paid_calls():
    c = MultiAgentConfig.from_env({"BRIAN_MAX_DEBATE_ROUNDS": "9", "BRIAN_MAX_PROVIDER_CALLS_PER_CANDIDATE": "50",
                                   "BRIAN_MULTI_AGENT_ENABLED": "true", "BRIAN_AGENT_COST_LIMIT_USD": "2",
                                   "BRIAN_MULTI_AGENT_STOP": "true"})
    assert c.max_debate_rounds == 2 and c.max_provider_calls_per_candidate == 6
    assert c.paid_calls_allowed is False
    with pytest.raises(UnsafeConfigError):
        MultiAgentConfig(real_trading_enabled=True)


# --- provider guard --------------------------------------------------------------------------------
class FailingSdk:
    def __getattr__(self, name):  # any SDK use would be a test failure
        raise AssertionError("SDK must not be touched")


def test_paid_provider_with_zero_budget_never_reaches_the_sdk(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    p = ClaudeProvider("claude-opus-5-5", sdk=FailingSdk(), ledger=CostLedger(0.0))
    r = p.call(request(), calls=CallCounter(6), allow_paid=False)
    assert (r.status, r.error_class, r.estimated_cost_usd) == ("SKIPPED", "PAID_CALLS_DISABLED", 0.0)
    r = p.call(request(), calls=CallCounter(6), allow_paid=True)  # allowed, but the budget is zero
    assert (r.status, r.error_class) == ("SKIPPED", "BUDGET_EXCEEDED")


def test_unknown_model_price_and_missing_key_are_refused(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    sdk = SimpleNamespace()
    assert ClaudeProvider("claude-opus-5-5", sdk=sdk, ledger=CostLedger(5)).call(
        request(), calls=CallCounter(6), allow_paid=True).error_class == "PROVIDER_UNAVAILABLE"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    assert ClaudeProvider("claude-made-up", sdk=sdk, ledger=CostLedger(5)).call(
        request(), calls=CallCounter(6), allow_paid=True).error_class == "UNKNOWN_MODEL_PRICE"


def test_call_cap_cache_and_rate_limit():
    sleeps, now = [], [0.0]
    p = MockProvider("m", lambda r: GOOD)
    p.min_interval_seconds, p._clock, p._sleep = 2.0, (lambda: now[0]), sleeps.append
    calls = CallCounter(2)
    assert p.call(request(), calls=calls, allow_paid=False).status == "OK"
    cached = p.call(request(), calls=calls, allow_paid=False)
    assert cached.cached and len(p.requests) == 1 and calls.used == 1
    assert p.call(request(max_output_tokens=10), calls=calls, allow_paid=False).status == "OK"
    assert sleeps == [2.0]
    assert p.call(request(max_output_tokens=11), calls=calls, allow_paid=False).error_class == "CALL_CAP_REACHED"


def test_provider_errors_are_reduced_to_a_class_name_without_secrets():
    class APITimeoutError(Exception):
        pass
    leak = mock("m", lambda r: RuntimeError("Authorization: Bearer sk-secret-123 prompt text"))
    r = leak.call(request(), calls=CallCounter(6), allow_paid=False)
    assert r.status == "FAILED" and r.error_class == "PROVIDER_ERROR:RuntimeError"
    assert "secret" not in json.dumps(dataclasses.asdict(r))
    r = mock("m", lambda r: APITimeoutError("x")).call(request(), calls=CallCounter(6), allow_paid=False)
    assert r.error_class == "TIMEOUT"


# --- real adapters against fake SDKs ---------------------------------------------------------------
class FakeAnthropic:
    def __init__(self, text, stop_reason="end_turn"):
        self.text, self.stop_reason, self.created = text, stop_reason, []
        self.Anthropic = self._client

    def _client(self, timeout, max_retries):
        self.client_args = (timeout, max_retries)
        return SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.created.append(kwargs)
        return SimpleNamespace(stop_reason=self.stop_reason, usage=SimpleNamespace(input_tokens=1_000, output_tokens=200),
                               content=[SimpleNamespace(type="text", text=self.text)])


def test_claude_adapter_uses_structured_output_and_records_cost(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    sdk = FakeAnthropic(json.dumps(GOOD))
    ledger = CostLedger(1.0)
    r = ClaudeProvider("claude-opus-5-5", sdk=sdk, ledger=ledger, min_interval_seconds=0).call(
        request(), calls=CallCounter(6), allow_paid=True)
    assert r.status == "OK" and r.payload["stance"] == "bullish"
    sent = sdk.created[0]
    assert sent["model"] == "claude-opus-5-5" and sent["output_config"]["format"]["type"] == "json_schema"
    assert r.estimated_cost_usd == pytest.approx(1_000 * 4 / 1e6 + 200 * 20 / 1e6)
    assert ledger.spent_usd == pytest.approx(r.estimated_cost_usd) and ledger.reserved_usd == 0


@pytest.mark.parametrize("text,stop,error", [("not json", "end_turn", "INVALID_JSON"),
                                             (json.dumps({"stance": "moon"}), "end_turn", "SCHEMA_INVALID"),
                                             (json.dumps(GOOD), "refusal", "MODEL_REFUSED"),
                                             (json.dumps(GOOD), "max_tokens", "TRUNCATED_OUTPUT")])
def test_claude_adapter_fails_closed_on_bad_answers(monkeypatch, text, stop, error):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    r = ClaudeProvider("claude-opus-5-5", sdk=FakeAnthropic(text, stop), ledger=CostLedger(1.0),
                       min_interval_seconds=0).call(request(), calls=CallCounter(6), allow_paid=True)
    assert (r.status, r.error_class) == ("FAILED", error)


def test_grok_adapter_cuts_x_search_at_the_decision_date(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-test-not-real")
    seen = {}

    class Chat:
        def __init__(self, **kw):
            seen["create"], self.msgs = kw, []

        def append(self, m):
            self.msgs.append(m)

        def sample(self):
            return SimpleNamespace(content=json.dumps(GOOD), citations=["https://x.com/a/status/1"],
                                   usage=SimpleNamespace(prompt_tokens=500, completion_tokens=100), cost_usd=0.012)

    sdk = SimpleNamespace(Client=lambda timeout: SimpleNamespace(chat=SimpleNamespace(create=lambda **kw: Chat(**kw))))
    tools = SimpleNamespace(x_search=lambda **kw: ("x_search", kw))
    chat = SimpleNamespace(system=lambda t: ("system", t), user=lambda t: ("user", t))
    p = GrokProvider("grok-4.7", sdk=sdk, chat_helpers=chat, tools_module=tools, ledger=CostLedger(2.0),
                     min_interval_seconds=0)
    r = p.call(request(x_search=True, x_search_to_date="2026-11-06"), calls=CallCounter(6), allow_paid=True)
    assert r.status == "OK" and r.reported_cost_usd == 0.012 and r.estimated_cost_usd == 0.012
    assert seen["create"]["tools"][0][1]["to_date"] == datetime(2026, 11, 6, tzinfo=timezone.utc)
    assert r.citations == ("https://x.com/a/status/1",)
    r = p.call(request(x_search=True), calls=CallCounter(6), allow_paid=True)
    assert r.error_class == "X_SEARCH_WITHOUT_CUTOFF"


# --- deterministic roles ---------------------------------------------------------------------------
def test_verifier_fails_on_future_sources_and_flags_stale_or_social_only_data():
    c, e = good_case()
    future = Source("https://example.com/later", "INDEPENDENT_PROFESSIONAL", utc_iso(T + timedelta(minutes=1)), None)
    v = agents.verify(dataclasses.replace(c, sources=c.sources + (future,)), T, e, [])
    assert v.status == "FAILED" and any(p.startswith("SOURCES_PUBLISHED_AFTER_DECISION") for p in v.problems)
    stale = dataclasses.replace(c, price_observed_at=utc_iso(T - timedelta(hours=1)))
    assert "PRICE_STALE" in agents.verify(stale, T, e, []).problems
    social = Source("https://x.com/a/status/1", "SOCIAL", utc_iso(T - timedelta(minutes=3)), None)
    assert "SOCIAL_ONLY_SOURCES" in agents.verify(dataclasses.replace(c, sources=(social,)), T, e, []).problems
    late_edge = dataclasses.replace(e, computed_at=utc_iso(T + timedelta(minutes=5)))
    assert agents.verify(c, T, late_edge, []).status == "FAILED"


def test_scanner_drops_ordinary_stale_and_future_moves():
    c, _ = good_case()
    quiet = dataclasses.replace(c, candidate_id="q", move_bps=20.0, volume_ratio=1.1)
    old = dataclasses.replace(c, candidate_id="o", observed_at=utc_iso(T - timedelta(hours=1)))
    ahead = dataclasses.replace(c, candidate_id="f", observed_at=utc_iso(T + timedelta(minutes=1)))
    kept, dropped = agents.scan([c, quiet, old, ahead], T)
    assert [k.candidate_id for k in kept] == [c.candidate_id]
    assert [r for _, r in dropped] == ["MOVE_NOT_UNUSUAL", "STALE_CANDIDATE", "FUTURE_CANDIDATE"]


def test_risk_limits_are_frozen_and_block_without_measured_edge_or_daily_budget():
    with pytest.raises(dataclasses.FrozenInstanceError):
        agents.RISK_LIMITS.min_net_edge_bps = -100
    c, e = good_case()
    assert agents.risk_check(c, e, opens_today=0, day_return_pct=0.0).status == "PASS"
    assert "EXPECTED_EDGE_NOT_MEASURED" in agents.risk_check(c, None, opens_today=0).reasons
    assert "DAILY_TRADE_LIMIT_OR_UNKNOWN" in agents.risk_check(c, e).reasons
    assert "DAILY_LOSS_LIMIT" in agents.risk_check(c, e, opens_today=0, day_return_pct=-0.03).reasons
    wide = dataclasses.replace(c, spread_bps=40.0)
    assert "SPREAD_TOO_WIDE" in agents.risk_check(wide, e, opens_today=0).reasons


def test_skeptic_objections_carry_evidence():
    c, e = good_case()
    losing = dataclasses.replace(e, gross_bps=5.0)
    late = dataclasses.replace(c, observed_at=utc_iso(T - timedelta(minutes=30)))
    codes = {f.code: f for f in agents.skeptic(late, T, losing, [])}
    assert codes["LATE_ENTRY"].blocking and codes["COST_EXCEEDS_MEASURED_EDGE"].blocking
    assert "net=" in codes["COST_EXCEEDS_MEASURED_EDGE"].evidence


# --- coordinator -----------------------------------------------------------------------------------
def run(coord, case=None, **kw):
    c, e = case or good_case()
    kw.setdefault("opens_today", 0)
    kw.setdefault("day_return_pct", 0.0)
    return coord.run(c, T, e, **kw)


def test_full_path_runs_two_rounds_and_never_allows_execution():
    claude, grok = mock("claude"), mock("grok")
    rec = run(coordinator(claude, grok))
    assert rec.decision == "BUY_CANDIDATE" and rec.debate_rounds == 2 and len(rec.usage) == 4
    assert rec.execution_allowed is False and rec.mock is True
    # round 1 is independent: Claude never sees Grok's answer before its own first assessment
    assert "Independent assessment" not in claude.requests[0].prompt
    assert "Independent assessment" in claude.requests[1].prompt
    assert grok.requests[0].x_search and grok.requests[0].x_search_to_date == "2026-11-06"


def test_measured_edge_below_cost_ends_without_model_calls():
    claude, grok = mock("claude"), mock("grok")
    rec = run(coordinator(claude, grok), fixtures()[1])
    assert rec.decision == "NO_TRADE" and rec.usage == () and claude.requests == [] == grok.requests


def test_failures_disagreement_stop_and_duplicates_never_trade():
    failing = mock("grok", lambda r: RuntimeError("boom"))
    assert run(coordinator(grok=failing)).decision == "NO_TRADE"
    bear = dict(GOOD, stance="bearish")
    assert run(coordinator(grok=mock("grok", bear))).decision == "WAIT"
    stopped = coordinator(operator_stop=True)
    rec = run(stopped)
    assert rec.decision == "NO_TRADE" and rec.usage == ()
    coord = coordinator()
    run(coord)
    assert run(coord).decision == "REJECT"


def test_call_cap_and_round_limit_are_respected():
    claude, grok = mock("claude"), mock("grok")
    rec = run(coordinator(claude, grok, max_debate_rounds=1))
    assert rec.debate_rounds == 1 and len(claude.requests) + len(grok.requests) == 2
    capped = run(coordinator(max_provider_calls_per_candidate=3))
    assert sum(1 for u in capped.usage if u.status != "SKIPPED") == 3
    assert capped.decision == "INSUFFICIENT_DATA"


def test_default_config_with_real_providers_costs_nothing_and_does_not_trade(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("XAI_API_KEY", "xai-test-not-real")
    ledger = CostLedger(0.0)
    coord = Coordinator(MultiAgentConfig(enabled=True), ClaudeProvider("claude-opus-5-5", sdk=FailingSdk(), ledger=ledger),
                        GrokProvider("grok-4.7", sdk=FailingSdk(), ledger=ledger))
    rec = run(coord)
    assert rec.decision == "INSUFFICIENT_DATA" and rec.provider_cost_usd == 0 and ledger.spent_usd == 0
    assert {u.error_class for u in rec.usage} == {"PAID_CALLS_DISABLED"}
    assert "not-real" not in json.dumps(rec.to_dict(), default=str)


def test_decision_record_cannot_allow_execution():
    rec = run(coordinator())
    with pytest.raises(ValueError):
        dataclasses.replace(rec, execution_allowed=True)
    with pytest.raises(ValueError):
        dataclasses.replace(rec, decision="BUY_NOW")


# --- evidence store / evaluation / dry run ---------------------------------------------------------
def test_evidence_store_is_append_only_and_detects_tampering(tmp_path):
    store = EvidenceStore(tmp_path / "d.jsonl")
    store.append(run(coordinator()))
    store.append(run(coordinator(), fixtures()[1]))
    assert store.verify()
    lines = store.path.read_text(encoding="utf-8").splitlines()
    store.path.write_text("\n".join([lines[0].replace("BUY_CANDIDATE", "WAIT"), lines[1]]) + "\n", encoding="utf-8")
    assert not store.verify()


def test_compare_groups_uses_only_outcomes_learned_after_the_decision():
    d = lambda cid, dec, cost=0.0: {"candidate_id": cid, "decision": dec, "decision_time_utc": utc_iso(T),
                                    "provider_cost_usd": cost}
    later, before = utc_iso(T + timedelta(hours=1)), utc_iso(T - timedelta(minutes=1))
    outcomes = {"a": {"direction": 1, "gross_bps": 30, "cost_bps": 18, "known_at": later},
                "b": {"direction": 1, "gross_bps": 5, "cost_bps": 18, "known_at": later},
                "c": {"direction": -1, "gross_bps": 50, "cost_bps": 18, "known_at": before},
                "d": {"direction": 1, "gross_bps": 40, "cost_bps": 18, "known_at": later}}
    (rep,) = compare_groups({"C": [d("a", "BUY_CANDIDATE", 0.1), d("b", "BUY_CANDIDATE"),
                                   d("c", "SELL_CANDIDATE"), d("d", "WAIT"), d("e", "NO_TRADE")]}, outcomes)
    assert (rep.outcomes_used, rep.outcomes_leaky, rep.outcomes_missing) == (3, 1, 1)
    assert (rep.true_positives, rep.false_positives, rep.missed_opportunities) == (1, 1, 1)
    assert rep.net_bps_total == pytest.approx(12 - 13) and rep.max_drawdown_bps == pytest.approx(13)
    assert rep.provider_cost_usd == pytest.approx(0.1)


def test_dry_run_end_to_end(tmp_path):
    results = {r["candidate_id"]: r for r in dry_run(tmp_path / "d.jsonl")}
    assert results["dry-run-1"]["decision"] == "BUY_CANDIDATE" and results["dry-run-1"]["model_calls"] == 4
    assert results["dry-run-2"]["decision"] == "NO_TRADE" and results["dry-run-2"]["model_calls"] == 0
    assert all(r["execution_allowed"] is False and r["provider_cost_usd"] == 0 for r in results.values())


def test_live_canary_does_not_run_without_budget(monkeypatch):
    monkeypatch.delenv("BRIAN_AGENT_COST_LIMIT_USD", raising=False)
    assert live_canary()["status"] == "NOT_RUN"

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAMPLER = ROOT / "supabase" / "functions" / "brian-realtime-readiness-cost-sampler" / "index.ts"
SCHEDULER = ROOT / "supabase" / "functions" / "brian-realtime-core-scheduler" / "index.ts"
EDGE_READER = ROOT / "brian2026" / "phase108_supabase_lagged_edge_reader.py"


def test_readiness_cost_sampler_is_public_l2_shadow_only() -> None:
    source = SAMPLER.read_text(encoding="utf-8")

    assert 'requireRealtimeInternal(req)' in source
    assert 'https://api.binance.com/api/v3/depth' in source
    assert 'compileL2Cost({' in source
    assert 'side: "BUY"' in source
    assert 'side: "SELL"' in source
    assert 'evidence_class: EVIDENCE' in source
    assert 'shadow_only: true' in source
    assert 'live_execution: false' in source


def test_readiness_cost_sampler_has_no_private_exchange_order_surface() -> None:
    source = SAMPLER.read_text(encoding="utf-8").lower()

    for forbidden in (
        "/api/v3/order",
        "/api/v3/account",
        "x-mbx-apikey",
        "exchange_api_key",
        "place_order",
        "create_order",
        "submit_order",
        "live_execution: true",
    ):
        assert forbidden not in source


def test_readiness_cost_uses_conservative_buy_sell_max_and_fixed_provenance() -> None:
    source = SAMPLER.read_text(encoding="utf-8")
    reader = EDGE_READER.read_text(encoding="utf-8")

    assert "buy.estimatedRoundTripCostBps >= sell.estimatedRoundTripCostBps" in source
    assert 'const VERSION = "brian.readiness-cost-sampler.v1"' in source
    assert 'READINESS_COST_COMPILER_VERSION = "brian.readiness-cost-sampler.v1"' in reader
    assert '"compiler_version": f"eq.{READINESS_COST_COMPILER_VERSION}"' in reader


def test_readiness_cost_sampler_is_scheduled_every_three_minutes() -> None:
    source = SCHEDULER.read_text(encoding="utf-8")

    assert 'action==="readiness_cost"' in source
    assert 'brian-realtime-readiness-cost-sampler' in source
    assert 'if(minute%3===0) actions.push("readiness_cost");' in source

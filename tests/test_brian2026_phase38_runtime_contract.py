from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EDGE = ROOT / "supabase" / "functions" / "brian-sensor-mesh" / "index.ts"
SCHEMA = ROOT / "supabase" / "migrations" / "202609030004_brian_phase38_sensor_mesh.sql"
RUNTIME = ROOT / "supabase" / "migrations" / "202609030005_brian_phase38_sensor_mesh_runtime.sql"


def test_phase38_edge_is_public_market_read_only_and_shadow_only():
    text = EDGE.read_text(encoding="utf-8")
    assert "https://api.binance.com/api/v3/klines" in text
    assert "https://api.binance.com/api/v3/ticker/bookTicker" in text
    forbidden = (
        "/api/v3/order",
        "createOrder",
        "create_order",
        "placeOrder",
        "place_order",
        "submitOrder",
        "submit_order",
        "MARKET_BUY",
        "MARKET_SELL",
    )
    assert not any(token in text for token in forbidden)
    assert 'live_execution: false' in text
    assert 'learning_enabled: false' in text
    assert "FAILED_CLOSED" in text


def test_phase38_missing_non_price_families_are_explicitly_unavailable():
    text = EDGE.read_text(encoding="utf-8")
    for family in ("derivatives", "onchain", "news", "social_psychology", "cross_asset", "macro"):
        assert family in text
    assert "unavailable_families" in text


def test_phase38_schema_is_append_only_rls_and_micro_book_is_virtual():
    text = SCHEMA.read_text(encoding="utf-8")
    for table in (
        "brian_sensor_observations",
        "brian_micro_book_ticks",
        "brian_micro_book_receipts",
        "brian_opportunity_tournament_rounds",
        "brian_missed_opportunity_receipts",
    ):
        assert f"alter table public.{table} enable row level security" in text
        assert table in text
    assert "starting_equity in (2,3,5,10,20)" in text
    assert "live_execution boolean not null default false check (not live_execution)" in text
    assert "brian_reject_mutation" in text


def test_phase38_scheduler_is_five_minute_vault_backed_and_separate_from_phase37():
    text = RUNTIME.read_text(encoding="utf-8")
    assert "brian-sensor-mesh-5m" in text
    assert "2-59/5 * * * *" in text
    assert "brian_project_url" in text
    assert "brian_anon_jwt" in text
    assert "/functions/v1/brian-sensor-mesh" in text
    assert "brian-live-shadow" not in text

from __future__ import annotations

from pathlib import Path
import re

from brian2026.universe_radar import UniverseConfig


ROOT = Path(__file__).resolve().parents[1]
MIGRATION_MEMORY = ROOT / "supabase/migrations/202609030001_brian_intelligence_memory.sql"
MIGRATION_RUNTIME = ROOT / "supabase/migrations/202609030002_brian_collector_runtime.sql"
COLLECTOR = ROOT / "supabase/functions/brian-universe-collector/index.ts"
DOC = ROOT / "docs/BRIAN_PHASE32_SUPABASE_RUNTIME.md"


def _read(path: Path) -> str:
    assert path.exists(), f"missing runtime artifact: {path}"
    return path.read_text(encoding="utf-8")


def _ts_number(source: str, name: str) -> float:
    match = re.search(rf"\b{re.escape(name)}:\s*([0-9_+.eE-]+)", source)
    assert match, f"missing TS config value {name}"
    return float(match.group(1).replace("_", ""))


def test_runtime_files_do_not_commit_deployed_supabase_credentials() -> None:
    combined = "\n".join(_read(path) for path in (MIGRATION_MEMORY, MIGRATION_RUNTIME, COLLECTOR, DOC))
    forbidden = (
        "qbcjuxhvhwagvqbjyemo",
        "eyJhbGciOi",
        "sb_publishable_",
        "SUPABASE_SERVICE_ROLE_KEY=",
    )
    for token in forbidden:
        assert token not in combined


def test_collector_is_public_data_shadow_only_and_has_no_exchange_order_surface() -> None:
    source = _read(COLLECTOR)
    assert "https://api.binance.com/api/v3/exchangeInfo" in source
    assert "https://api.binance.com/api/v3/ticker/24hr" in source
    assert "https://api.binance.com/api/v3/ticker/bookTicker" in source
    assert 'shadow_only: true' in source
    assert 'contentType: "application/gzip"' in source
    assert 'MIN_INTERVAL_SECONDS = 780' in source

    forbidden = (
        "/api/v3/order",
        "create_order",
        "place_order",
        "submit_order",
        "execute_order",
        "BINANCE_API_KEY",
        "BINANCE_SECRET",
    )
    lowered = source.lower()
    for token in forbidden:
        assert token.lower() not in lowered


def test_supabase_tables_are_client_closed_and_service_append_only() -> None:
    migration = _read(MIGRATION_MEMORY).lower()
    assert "enable row level security" in migration
    assert "revoke all on public.brian_raw_captures from anon, authenticated" in migration
    assert "revoke update, delete, truncate" in migration
    assert "from service_role" in migration
    assert "brian_reject_mutation" in migration
    assert "brian_validate_source_outcome_time" in migration
    assert "brian_validate_opportunity_outcome_time" in migration


def test_scheduler_uses_vault_names_not_literal_project_credentials() -> None:
    migration = _read(MIGRATION_RUNTIME)
    assert "brian_project_url" in migration
    assert "brian_anon_jwt" in migration
    assert "vault.decrypted_secrets" in migration
    assert "*/15 * * * *" in migration
    assert "supabase.co" not in migration
    assert "eyJ" not in migration


def test_typescript_universe_thresholds_match_python_defaults() -> None:
    source = _read(COLLECTOR)
    config = UniverseConfig()
    assert _ts_number(source, "min_quote_volume") == config.min_quote_volume
    assert int(_ts_number(source, "min_trades_24h")) == config.min_trades_24h
    assert _ts_number(source, "min_price") == config.min_price
    assert int(_ts_number(source, "top_n")) == config.top_n
    assert _ts_number(source, "max_abs_change_pct") == config.max_abs_change_pct


def test_book_ticker_degrades_instead_of_becoming_required() -> None:
    source = _read(COLLECTOR)
    assert "fetchJson(BOOK_TICKER, false)" in source
    assert 'degraded_sources: book.degraded ? ["book_ticker"] : []' in source
    assert "spread === null ? 0.50" in source

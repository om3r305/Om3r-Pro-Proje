from pathlib import Path

import pytest

from brian2026.global_eyes import GLOBAL_EYE_PROVIDERS, EyeProviderSpec, live_provider_ids, unavailable_domains

ROOT = Path(__file__).resolve().parents[1]


def test_only_public_no_key_providers_are_live():
    assert set(live_provider_ids()) == {"binance-usdm-public", "gdelt-doc2", "ecb-reference-fx"}
    assert all(row.access_mode == "PUBLIC_NO_KEY" for row in GLOBAL_EYE_PROVIDERS if row.enabled_live)


def test_credentialed_and_licensed_domains_fail_closed():
    missing = set(unavailable_domains())
    assert {"social_psychology", "onchain", "equities", "metals"} <= missing


def test_provider_cannot_enable_live_execution():
    with pytest.raises(ValueError, match="cannot execute live trades"):
        EyeProviderSpec("x", "x", "PUBLIC_NO_KEY", "FAST_5M", "https://example.com", True, "x", live_execution=True)


def test_credentialed_provider_cannot_be_silently_enabled():
    with pytest.raises(ValueError, match="cannot be silently enabled"):
        EyeProviderSpec("x", "x", "API_KEY_REQUIRED", "EVENT_DRIVEN", "https://example.com", True, "x")


def test_derivatives_eye_uses_public_market_data_only():
    text = (ROOT / "supabase/functions/brian-derivatives-eye/index.ts").read_text()
    assert "fapi.binance.com" in text
    assert "openInterestHist" in text
    assert "takerlongshortRatio" in text
    assert "premiumIndex" in text
    forbidden = ["create_order", "place_order", "/fapi/v1/order", "X-MBX-APIKEY", "secretKey"]
    assert not any(token in text for token in forbidden)
    assert "live_execution:false" in text


def test_news_eye_marks_discovery_as_not_source_truth():
    text = (ROOT / "supabase/functions/brian-news-eye/index.ts").read_text()
    assert "GDELT_DISCOVERY" in text
    assert "UNVERIFIED_DISCOVERY" in text
    assert "not_source_truth:true" in text
    assert "headline_pressure_only:true" in text
    assert "live_execution:false" in text


def test_fx_eye_is_daily_and_does_not_fake_intraday():
    text = (ROOT / "supabase/functions/brian-fx-eye/index.ts").read_text()
    assert "ecb.europa.eu" in text
    assert 'horizon:"DAILY"' in text
    assert 'native_frequency:"daily"' in text
    assert "not_intraday:true" in text


def test_collector_health_is_append_only_and_rls_protected():
    text = (ROOT / "supabase/migrations/202609030006_brian_phase39_global_eyes.sql").read_text()
    assert "brian_collector_runs" in text
    assert "enable row level security" in text
    assert "brian_collector_runs_append_only" in text
    assert "revoke update, delete, truncate" in text


def test_global_eye_cron_keeps_collectors_separate():
    text = (ROOT / "supabase/migrations/202609030006_brian_phase39_global_eyes.sql").read_text()
    assert "brian-derivatives-eye-5m" in text
    assert "brian-news-eye-10m" in text
    assert "brian-fx-eye-hourly" in text
    assert "brian-live-shadow-5m" not in text


def test_phase39_does_not_claim_live_capital():
    for path in (
        ROOT / "supabase/functions/brian-derivatives-eye/index.ts",
        ROOT / "supabase/functions/brian-news-eye/index.ts",
        ROOT / "supabase/functions/brian-fx-eye/index.ts",
    ):
        text = path.read_text().lower()
        assert "shadow_only:true" in text
        assert "learning_enabled:false" in text or "learning_enabled" not in text
        assert "live_execution:false" in text

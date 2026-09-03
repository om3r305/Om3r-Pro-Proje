from __future__ import annotations

from datetime import datetime, timezone
import pytest

from brian2026.multi_asset_history import (
    CRYPTO_CURRICULUM_SYMBOLS,
    OFFICIAL_CROSS_MARKET_SERIES,
    CurriculumArchiveSpec,
    CurriculumHistoryManifest,
    PointInTimeValue,
    asof_value,
    default_curriculum_history_manifest,
    monthly_crypto_specs,
    provider_requirements,
)
from brian2026.portfolio import DEVELOPMENT_CUTOFF


def test_crypto_curriculum_is_broad_unique_and_contains_majors_and_fan_tokens() -> None:
    assert len(CRYPTO_CURRICULUM_SYMBOLS) == 30
    assert len(set(CRYPTO_CURRICULUM_SYMBOLS)) == len(CRYPTO_CURRICULUM_SYMBOLS)
    for symbol in ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT", "LINKUSDT"):
        assert symbol in CRYPTO_CURRICULUM_SYMBOLS
    for symbol in ("BARUSDT", "PSGUSDT", "CITYUSDT", "PORTOUSDT", "ALPINEUSDT"):
        assert symbol in CRYPTO_CURRICULUM_SYMBOLS


def test_curriculum_archive_uses_only_official_binance_vision_and_checksum_path() -> None:
    spec = CurriculumArchiveSpec("XRPUSDT", 2025, 12)
    assert spec.url == "https://data.binance.vision/data/spot/monthly/klines/XRPUSDT/1m/XRPUSDT-1m-2025-12.zip"
    assert spec.checksum_url == spec.url + ".CHECKSUM"
    assert spec.end == DEVELOPMENT_CUTOFF


def test_curriculum_archive_hard_rejects_2026_historical_development() -> None:
    with pytest.raises(ValueError, match="INVALID_CONTAMINATED"):
        CurriculumArchiveSpec("BTCUSDT", 2026, 1)


def test_monthly_crypto_plan_is_cutoff_safe() -> None:
    items = monthly_crypto_specs(
        "ETHUSDT",
        start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert len(items) == 72
    assert items[0].filename == "ETHUSDT-1m-2020-01.zip"
    assert items[-1].filename == "ETHUSDT-1m-2025-12.zip"
    assert all(item.start < DEVELOPMENT_CUTOFF for item in items)

    with pytest.raises(ValueError, match="cannot cross"):
        monthly_crypto_specs(
            "ETHUSDT",
            start=datetime(2025, 12, 1, tzinfo=timezone.utc),
            end=datetime(2026, 2, 1, tzinfo=timezone.utc),
        )


def test_cross_market_registry_keeps_native_clocks_and_provider_requirements_explicit() -> None:
    by_id = {row.canonical_id: row for row in OFFICIAL_CROSS_MARKET_SERIES}
    assert by_id["fx-usd-eur"].native_frequency == "1d"
    assert by_id["fx-usd-eur"].access_mode == "PUBLIC_NO_KEY"
    assert by_id["oil-wti-spot"].provider_series_id == "PET.RWTC.D"
    assert by_id["oil-brent-spot"].provider_series_id == "PET.RBRTE.D"
    assert by_id["macro-us-cpi"].native_frequency == "1mo"
    assert by_id["macro-us-cpi"].revision_model == "REVISION_AWARE"
    assert by_id["equity-us-broad-universe"].access_mode == "LICENSED_REQUIRED"
    assert by_id["gold-tradable-history"].access_mode == "LICENSED_REQUIRED"
    req = provider_requirements()
    assert "fx-usd-eur" in req["PUBLIC_NO_KEY"]
    assert "macro-us-cpi" in req["API_KEY_REQUIRED"]
    assert "equity-us-broad-universe" in req["LICENSED_REQUIRED"]


def test_asof_value_never_uses_future_release_or_future_revision() -> None:
    rows = (
        PointInTimeValue("CPIAUCSL", 100.0, 120.0, 300.0, "v1"),
        PointInTimeValue("CPIAUCSL", 100.0, 200.0, 301.0, "v2-revision"),
        PointInTimeValue("CPIAUCSL", 180.0, 220.0, 302.0, "v3-new-month"),
    )
    assert asof_value(rows, decision_time=119.0) is None
    at_150 = asof_value(rows, decision_time=150.0)
    assert at_150 is not None
    assert at_150.value == 300.0
    assert at_150.vintage_id == "v1"
    at_210 = asof_value(rows, decision_time=210.0)
    assert at_210 is not None
    assert at_210.value == 301.0
    assert at_210.vintage_id == "v2-revision"
    at_230 = asof_value(rows, decision_time=230.0)
    assert at_230 is not None
    assert at_230.value == 302.0
    assert at_230.source_observation_time == 180.0


def test_asof_value_rejects_mixed_series_and_contaminated_decision_window() -> None:
    rows = (
        PointInTimeValue("CPIAUCSL", 100.0, 120.0, 300.0, "cpi-v1"),
        PointInTimeValue("UNRATE", 100.0, 120.0, 4.0, "unrate-v1"),
    )
    with pytest.raises(ValueError, match="cannot mix"):
        asof_value(rows, decision_time=150.0)
    with pytest.raises(ValueError, match="INVALID_CONTAMINATED"):
        asof_value((rows[0],), decision_time=DEVELOPMENT_CUTOFF)


def test_point_in_time_value_cannot_be_available_before_observation_or_in_2026() -> None:
    with pytest.raises(ValueError, match="before"):
        PointInTimeValue("bad", 100.0, 99.0, 1.0)
    with pytest.raises(ValueError, match="INVALID_CONTAMINATED"):
        PointInTimeValue("bad", DEVELOPMENT_CUTOFF - 60.0, DEVELOPMENT_CUTOFF, 1.0)


def test_history_manifest_forbids_fake_intraday_upsampling_and_final_holdout_claim() -> None:
    base = default_curriculum_history_manifest()
    assert base.historical_cutoff_exclusive == DEVELOPMENT_CUTOFF
    assert base.allow_fake_intraday_upsampling is False
    assert base.final_holdout_evaluated is False
    assert base.training_only is True
    assert len(base.manifest_id) == 64

    with pytest.raises(ValueError, match="fabricated"):
        CurriculumHistoryManifest(
            DEVELOPMENT_CUTOFF, CRYPTO_CURRICULUM_SYMBOLS, OFFICIAL_CROSS_MARKET_SERIES,
            allow_fake_intraday_upsampling=True,
        )
    with pytest.raises(ValueError, match="final holdout"):
        CurriculumHistoryManifest(
            DEVELOPMENT_CUTOFF, CRYPTO_CURRICULUM_SYMBOLS, OFFICIAL_CROSS_MARKET_SERIES,
            final_holdout_evaluated=True,
        )

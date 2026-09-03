from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEWS = ROOT / "supabase/functions/brian-news-eye/index.ts"


def test_official_news_feeds_are_primary_backbone():
    text = NEWS.read_text()
    assert "https://www.federalreserve.gov/feeds/press_all.xml" in text
    assert "https://www.ecb.europa.eu/rss/press.html" in text
    assert "https://www.sec.gov/news/pressreleases.rss" in text
    assert 'trust_class:item.official ? "OFFICIAL_PRIMARY"' in text


def test_gdelt_is_optional_and_failure_degrades_not_blinds():
    text = NEWS.read_text()
    assert "async function fetchGdelt" in text
    assert 'degraded.push("GDELT")' in text
    assert "all official RSS sources unavailable" in text


def test_news_does_not_turn_headline_sentiment_into_trade_direction():
    text = NEWS.read_text()
    assert "direction:0" in text
    assert "directional_inference_disabled:true" in text
    assert "directional inference intentionally disabled" in text
    assert "live_execution:false" in text

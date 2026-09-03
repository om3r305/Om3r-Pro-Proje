from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_phase39_pg_net_timeouts_are_explicit_and_news_is_longer():
    text = (ROOT / "supabase/migrations/202609030007_brian_phase39_http_timeouts.sql").read_text()
    assert "timeout_milliseconds := 20000" in text
    assert text.count("timeout_milliseconds := 10000") == 2
    assert "brian-news-eye-10m" in text
    assert "select brian_private.schedule_global_eyes();" in text

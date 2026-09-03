from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EDGE = ROOT / "supabase" / "functions" / "brian-intrabar-eye" / "index.ts"


def test_intrabar_rate_guard_is_anchored_to_previous_run_start():
    text = EDGE.read_text(encoding="utf-8")
    assert '.select("started_at")' in text
    assert '.order("started_at", { ascending: false })' in text
    assert 'lastRun.data?.started_at' in text
    assert 'Date.parse(lastRun.data.started_at)' in text
    assert '.select("finished_at")' not in text
    assert 'Date.parse(lastRun.data.finished_at)' not in text

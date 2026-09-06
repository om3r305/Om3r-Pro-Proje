from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one target, found {count}")
    p.write_text(text.replace(old, new, 1))


alpha = "supabase/functions/brian-alpha-decision-compiler/index.ts"
intrabar = "supabase/functions/brian-intrabar-eye/index.ts"
sensor = "supabase/functions/brian-sensor-mesh/index.ts"

replace_once(
    alpha,
    'const MAX_MACRO_CONTEXT_EVENTS = 24;\n',
    'const MAX_MACRO_CONTEXT_EVENTS = 24;\nconst ENABLE_DIP_DIRECTIONAL_EVIDENCE = false; // MAIN and DIP remain separate brains by default.\nconst RADAR_MAX_AGE_MS = 30 * 60_000; // 2x the canonical 15m universe cadence.\n',
)

old_radar = '''async function latestRadarAssets(): Promise<string[]> {
  const out = new Set<string>(CORE_ASSETS);
  const frame = await supabase.from("brian_emergent_mover_frames")
    .select("observed_at,report")
    .eq("provider", "binance_public")
    .order("observed_at", { ascending: false }).limit(1).maybeSingle();
  if (!frame.error && frame.data?.report && typeof frame.data.report === "object") {
    const candidates = (frame.data.report as Record<string, unknown>).candidates;
    if (Array.isArray(candidates)) {
      for (const raw of candidates.slice(0, MAX_ASSETS)) {
        if (!raw || typeof raw !== "object") continue;
        const symbol = String((raw as Record<string, unknown>).symbol ?? "").trim().toUpperCase();
        if (/^[A-Z0-9]+USDT$/.test(symbol)) out.add(`crypto:${symbol}`);
      }
    }
  }
  return [...out].slice(0, MAX_ASSETS);
}
'''
new_radar = '''async function latestRadarAssets(): Promise<string[]> {
  const out = new Set<string>(CORE_ASSETS);
  const snapshot = await supabase.from("brian_universe_snapshots")
    .select("observed_at,candidates")
    .order("observed_at", { ascending: false }).limit(1).maybeSingle();
  if (snapshot.error || !snapshot.data) return [...out];
  const observedMs = Date.parse(String(snapshot.data.observed_at));
  if (!Number.isFinite(observedMs) || Date.now() - observedMs > RADAR_MAX_AGE_MS || observedMs > Date.now() + 5_000) return [...out];
  const envelope = snapshot.data.candidates as Record<string, unknown> | null;
  const candidates = envelope && Array.isArray(envelope.candidates) ? envelope.candidates : [];
  for (const raw of candidates.slice(0, MAX_ASSETS)) {
    if (!raw || typeof raw !== "object") continue;
    const symbol = String((raw as Record<string, unknown>).symbol ?? "").trim().toUpperCase();
    if (/^[A-Z0-9]+USDT$/.test(symbol)) out.add(`crypto:${symbol}`);
  }
  return [...out].slice(0, MAX_ASSETS);
}
'''
replace_once(alpha, old_radar, new_radar)

old_evidence = '''async function loadSensorEvidence(assets: string[], nowMs: number): Promise<Map<string, AlphaEvidenceRow[]>> {
  const map = new Map<string, AlphaEvidenceRow[]>();
  const since = new Date(nowMs - 36 * 60 * 60_000).toISOString();
  const resp = await supabase.from("brian_sensor_observations")
    .select("observation_id,asset_id,sensor_family,horizon,independent_group,observed_at,direction,strength,confidence,reliability,available,reason")
    .in("asset_id", assets).gte("observed_at", since).eq("available", true)
    .neq("independent_group", "news_gdelt")
    .order("observed_at", { ascending: false }).limit(6000);
  if (resp.error) throw resp.error;
  for (const r of resp.data ?? []) {
    const asset = String(r.asset_id);
    const rows = map.get(asset) ?? [];
    rows.push({
      observationId: String(r.observation_id), sourceKind: String(r.sensor_family), independentGroup: String(r.independent_group),
      direction: Number(r.direction), strength: finite(r.strength), confidence: finite(r.confidence), reliability: finite(r.reliability),
      observedAt: String(r.observed_at), horizon: String(r.horizon), fresh: isFresh(String(r.observed_at), String(r.horizon), nowMs), reason: String(r.reason ?? "sensor evidence"),
    });
    map.set(asset, rows);
  }
  return map;
}
'''
new_evidence = '''async function loadSensorEvidence(assets: string[], nowMs: number): Promise<Map<string, AlphaEvidenceRow[]>> {
  const map = new Map<string, AlphaEvidenceRow[]>();
  const specs = [
    { horizon: "MICRO_1_5M", windowMs: freshnessMs("MICRO_1_5M") },
    { horizon: "FAST_5_30M", windowMs: freshnessMs("FAST_5_30M") },
    { horizon: "EVENT_DRIVEN", windowMs: freshnessMs("EVENT_DRIVEN") },
    { horizon: "DAILY", windowMs: freshnessMs("DAILY") },
  ] as const;
  const responses = await Promise.all(specs.map(async ({ horizon, windowMs }) => {
    const since = new Date(nowMs - windowMs).toISOString();
    return await supabase.from("brian_sensor_observations")
      .select("observation_id,asset_id,sensor_family,horizon,independent_group,observed_at,direction,strength,confidence,reliability,available,reason")
      .in("asset_id", assets).eq("horizon", horizon).gte("observed_at", since).eq("available", true)
      .neq("independent_group", "news_gdelt")
      .order("observed_at", { ascending: false }).limit(1500);
  }));
  for (const resp of responses) {
    if (resp.error) throw resp.error;
    for (const r of resp.data ?? []) {
      const asset = String(r.asset_id);
      const rows = map.get(asset) ?? [];
      rows.push({
        observationId: String(r.observation_id), sourceKind: String(r.sensor_family), independentGroup: String(r.independent_group),
        direction: Number(r.direction), strength: finite(r.strength), confidence: finite(r.confidence), reliability: finite(r.reliability),
        observedAt: String(r.observed_at), horizon: String(r.horizon), fresh: isFresh(String(r.observed_at), String(r.horizon), nowMs), reason: String(r.reason ?? "sensor evidence"),
      });
      map.set(asset, rows);
    }
  }
  return map;
}
'''
replace_once(alpha, old_evidence, new_evidence)
replace_once(alpha, '      await addDipEvidence(evidence, assets, evidenceNowMs);\n', '      if (ENABLE_DIP_DIRECTIONAL_EVIDENCE) await addDipEvidence(evidence, assets, evidenceNowMs);\n')
replace_once(
    alpha,
    '            emergent_mover_role: "attention_only",\n',
    '            emergent_mover_role: "attention_only",\n            radar_source: "brian_universe_snapshots",\n            dip_directional_evidence_enabled: ENABLE_DIP_DIRECTIONAL_EVIDENCE,\n',
)

replace_once(intrabar, 'const PRIOR_LOOKUP_BATCH = 40;\n', '')
old_intrabar = '''    const priorRows: PriorTick[] = [];
    for (let i = 0; i < allEyeIds.length; i += PRIOR_LOOKUP_BATCH) {
      const chunk = allEyeIds.slice(i, i + PRIOR_LOOKUP_BATCH);
      const priorResp = await supabase.from("brian_micro_book_ticks")
        .select("eye_id,starting_equity,equity_after,peak_equity_after,max_drawdown_pct_after,target_direction,observed_mid_price,observed_at")
        .in("eye_id", chunk)
        .order("observed_at", { ascending: false })
        .limit(1000);
      if (priorResp.error) throw priorResp.error;
      priorRows.push(...((priorResp.data ?? []) as PriorTick[]));
    }
    const latestByEye = new Map<string, PriorTick>();
    priorRows.sort((a, b) => Date.parse(String(b.observed_at ?? 0)) - Date.parse(String(a.observed_at ?? 0)));
    for (const row of priorRows) if (!latestByEye.has(row.eye_id)) latestByEye.set(row.eye_id, row);
'''
new_intrabar = '''    const priorResp = allEyeIds.length
      ? await supabase.rpc("brian_latest_micro_book_ticks", { p_eye_ids: allEyeIds })
      : { data: [], error: null };
    if (priorResp.error) throw priorResp.error;
    const latestByEye = new Map<string, PriorTick>();
    for (const row of (priorResp.data ?? []) as PriorTick[]) latestByEye.set(row.eye_id, row);
'''
replace_once(intrabar, old_intrabar, new_intrabar)
replace_once(intrabar, ', prior_lookup_batch: PRIOR_LOOKUP_BATCH', '')

old_sensor = '''    const eyeIds = observations.map((x) => x.eye_id); const priorResp = eyeIds.length ? await supabase.from("brian_micro_book_ticks").select("eye_id,starting_equity,equity_after,peak_equity_after,max_drawdown_pct_after,target_direction,observed_mid_price,observed_at").in("eye_id", eyeIds).order("observed_at", { ascending: false }).limit(500) : { data: [], error: null };
    if (priorResp.error) throw priorResp.error; const latestByEye = new Map<string, PriorTick>(); for (const row of (priorResp.data ?? []) as PriorTick[]) if (!latestByEye.has(row.eye_id)) latestByEye.set(row.eye_id, row);
'''
new_sensor = '''    const eyeIds = observations.map((x) => x.eye_id);
    const priorResp = eyeIds.length ? await supabase.rpc("brian_latest_micro_book_ticks", { p_eye_ids: eyeIds }) : { data: [], error: null };
    if (priorResp.error) throw priorResp.error;
    const latestByEye = new Map<string, PriorTick>();
    for (const row of (priorResp.data ?? []) as PriorTick[]) latestByEye.set(row.eye_id, row);
'''
replace_once(sensor, old_sensor, new_sensor)

Path("supabase/migrations/202609060001_brian_latest_micro_book_state_rpc.sql").write_text('''-- Brian 2026 production stabilization: exact latest micro state per eye.
-- Uses the existing (eye_id, observed_at DESC) index; intentionally adds no new index.
-- SHADOW-only infrastructure change. No Phase 3.7 or execution semantics change.

CREATE OR REPLACE FUNCTION public.brian_latest_micro_book_ticks(p_eye_ids text[])
RETURNS TABLE (
  eye_id text,
  starting_equity numeric,
  equity_after numeric,
  peak_equity_after numeric,
  max_drawdown_pct_after double precision,
  target_direction smallint,
  observed_mid_price numeric,
  observed_at timestamptz
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT latest.eye_id,
         latest.starting_equity,
         latest.equity_after,
         latest.peak_equity_after,
         latest.max_drawdown_pct_after,
         latest.target_direction,
         latest.observed_mid_price,
         latest.observed_at
  FROM unnest(coalesce(p_eye_ids, ARRAY[]::text[])) AS requested(eye_id)
  CROSS JOIN LATERAL (
    SELECT t.eye_id,
           t.starting_equity,
           t.equity_after,
           t.peak_equity_after,
           t.max_drawdown_pct_after,
           t.target_direction,
           t.observed_mid_price,
           t.observed_at
    FROM public.brian_micro_book_ticks AS t
    WHERE t.eye_id = requested.eye_id
    ORDER BY t.observed_at DESC
    LIMIT 1
  ) AS latest;
$$;

REVOKE ALL ON FUNCTION public.brian_latest_micro_book_ticks(text[]) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.brian_latest_micro_book_ticks(text[]) FROM anon, authenticated;
GRANT EXECUTE ON FUNCTION public.brian_latest_micro_book_ticks(text[]) TO service_role;
''')

Path("tests/test_brian2026_production_stabilization.py").write_text('''from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_alpha_uses_canonical_radar_and_causal_windows():
    source = (ROOT / "supabase/functions/brian-alpha-decision-compiler/index.ts").read_text()
    assert 'from("brian_universe_snapshots")' in source
    assert 'brian_emergent_mover_frames' not in source
    assert '.eq("horizon", horizon)' in source
    assert '.limit(6000)' not in source
    assert 'ENABLE_DIP_DIRECTIONAL_EVIDENCE = false' in source
    assert 'if (ENABLE_DIP_DIRECTIONAL_EVIDENCE) await addDipEvidence' in source


def test_micro_latest_state_consumers_use_rpc():
    intrabar = (ROOT / "supabase/functions/brian-intrabar-eye/index.ts").read_text()
    sensor = (ROOT / "supabase/functions/brian-sensor-mesh/index.ts").read_text()
    for source in (intrabar, sensor):
        assert 'rpc("brian_latest_micro_book_ticks"' in source
    assert '.limit(1000)' not in intrabar
    assert '.limit(500)' not in sensor


def test_latest_state_migration_is_lateral_and_index_neutral():
    sql = (ROOT / "supabase/migrations/202609060001_brian_latest_micro_book_state_rpc.sql").read_text().lower()
    assert "cross join lateral" in sql
    assert "order by t.observed_at desc" in sql
    assert "limit 1" in sql
    assert "create index" not in sql
    assert "grant execute" in sql
''')

Path("tests/test_brian2026_latest_micro_state_postgres.py").write_text('''import os
from pathlib import Path

import pytest

psycopg2 = pytest.importorskip("psycopg2")

DB_URL = os.getenv("BRIAN_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB_URL, reason="BRIAN_TEST_DATABASE_URL not configured")
ROOT = Path(__file__).resolve().parents[1]


def test_latest_micro_state_rpc_returns_exactly_one_latest_row_per_eye():
    with psycopg2.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            for role in ("anon", "authenticated", "service_role"):
                cur.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (role,))
                if cur.fetchone() is None:
                    cur.execute(f'CREATE ROLE "{role}" NOLOGIN')
            cur.execute("DROP FUNCTION IF EXISTS public.brian_latest_micro_book_ticks(text[])")
            cur.execute("DROP TABLE IF EXISTS public.brian_micro_book_ticks CASCADE")
            cur.execute("""
                CREATE TABLE public.brian_micro_book_ticks (
                    tick_id text PRIMARY KEY,
                    eye_id text NOT NULL,
                    starting_equity numeric NOT NULL,
                    equity_after numeric NOT NULL,
                    peak_equity_after numeric NOT NULL,
                    max_drawdown_pct_after double precision NOT NULL,
                    target_direction smallint NOT NULL,
                    observed_mid_price numeric NOT NULL,
                    observed_at timestamptz NOT NULL
                )
            """)
            cur.execute("CREATE INDEX brian_micro_book_tick_eye_time_idx ON public.brian_micro_book_ticks (eye_id, observed_at DESC)")
            cur.execute("""
                INSERT INTO public.brian_micro_book_ticks
                (tick_id,eye_id,starting_equity,equity_after,peak_equity_after,max_drawdown_pct_after,target_direction,observed_mid_price,observed_at)
                VALUES
                ('a-old','eye-a',5,5.1,5.2,1.0,1,100,'2026-09-06T07:00:00Z'),
                ('a-new','eye-a',5,5.3,5.3,1.0,-1,101,'2026-09-06T07:01:00Z'),
                ('b-old','eye-b',3,2.9,3.0,3.3,0,50,'2026-09-06T07:00:30Z'),
                ('b-new','eye-b',3,3.1,3.1,3.3,1,51,'2026-09-06T07:02:00Z')
            """)
            cur.execute((ROOT / "supabase/migrations/202609060001_brian_latest_micro_book_state_rpc.sql").read_text())
            cur.execute("SELECT eye_id,target_direction,observed_mid_price FROM public.brian_latest_micro_book_ticks(ARRAY['eye-a','eye-b']::text[]) ORDER BY eye_id")
            rows = cur.fetchall()
            assert len(rows) == 2
            assert rows[0] == ("eye-a", -1, 101)
            assert rows[1] == ("eye-b", 1, 51)
''')

ci = ".github/workflows/brian-ci.yml"
replace_once(
    ci,
    "          tests/test_brian2026_l2_capture_postgres.py\n          -v",
    "          tests/test_brian2026_l2_capture_postgres.py\n          tests/test_brian2026_latest_micro_state_postgres.py\n          -v",
)

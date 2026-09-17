import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { XMLParser } from "npm:fast-xml-parser@4.5.0";

const URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const ANON = Deno.env.get("SUPABASE_ANON_KEY") ?? "";
const db = createClient(URL, SERVICE, { auth: { persistSession: false, autoRefreshToken: false } });

const V = "brian.catalyst-sentinel.v2";
const E = "PROSPECTIVE_CATALYST_SENTINEL_SHADOW";
const AUTH = "control-v3";
const FAST = ["fed_monetary", "ecb_press", "bls_cpi", "bls_employment", "sec_press_rss"];
const CORE = [
  "crypto:BTCUSDT",
  "crypto:ETHUSDT",
  "crypto:SOLUSDT",
  "crypto:BNBUSDT",
  "crypto:XRPUSDT",
  "crypto:ADAUSDT",
];
const RECHECK = [5, 10, 15, 30];
const ACTIVE = ["WATCHING", "BUILDING", "BREAKOUT_CANDIDATE", "CONFIRMED"];
const WATCH_TTL_MS = 120 * 60_000;
const FEED_RECENCY_MS = 20 * 60_000;
const BASELINE_RECENCY_MS = 2 * 60_000;
const ALERT_BUCKET_MS = 60_000;
const ALERT_COOLDOWN_MS = 60_000;
const DISPATCH_TIMEOUT_MS = 5_000;

type EP = {
  endpoint_id: string;
  source_id: string;
  organization: string;
  canonical_domain: string;
  endpoint_url: string;
  category: string;
  tier: string;
};
type Item = { title: string; link: string; guid: string; publishedAt: string | null };
type Book = { mid: number; spread: number };
type Reaction = { ret: number; spread: number | null; imb?: number; range?: number; cross?: number; vol?: number };

const clip = (n: number) => Math.max(0, Math.min(1, Number.isFinite(n) ? n : 0));
const finite = (x: unknown, d = 0) => Number.isFinite(Number(x)) ? Number(x) : d;
const err = (e: unknown) => e instanceof Error ? `${e.name}: ${e.message}` : String(e);
const arr = <T>(x: T | T[] | null | undefined): T[] => x == null ? [] : Array.isArray(x) ? x : [x];

async function sha(s: string) {
  const d = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s)));
  return [...d].map((x) => x.toString(16).padStart(2, "0")).join("");
}

async function auth(req: Request) {
  const k = (req.headers.get("x-brian-cron-key") ?? "").trim();
  if (!k) throw Error("UNAUTHORIZED_CRON");
  const q = await db.from("brian_dashboard_auth").select("cron_key_sha256").eq("auth_id", AUTH).single();
  if (q.error || !q.data) throw Error(`CRON_AUTH_UNAVAILABLE:${q.error?.message ?? "missing"}`);
  const a = await sha(k);
  const b = String(q.data.cron_key_sha256 ?? "");
  if (a.length !== b.length) throw Error("UNAUTHORIZED_CRON");
  let z = 0;
  for (let i = 0; i < a.length; i++) z |= a.charCodeAt(i) ^ b.charCodeAt(i);
  if (z) throw Error("UNAUTHORIZED_CRON");
}

function txt(x: unknown) {
  if (typeof x === "string") return x.trim();
  if (x && typeof x === "object") {
    const y = x as Record<string, unknown>;
    return String(y["#text"] ?? y["@_href"] ?? y.href ?? "").trim();
  }
  return x == null ? "" : String(x).trim();
}
function iso(x: unknown) {
  const t = Date.parse(txt(x));
  return Number.isFinite(t) ? new Date(t).toISOString() : null;
}
function parse(xml: string): Item[] {
  const d = new XMLParser({ ignoreAttributes: false, trimValues: true }).parse(xml);
  const r = d?.rss?.channel?.item ?? d?.feed?.entry ?? [];
  return arr<any>(r).slice(0, 50).map((x) => {
    let l = "";
    for (const y of arr<any>(x.link)) {
      l = txt(y);
      if (l) break;
    }
    return {
      title: txt(x.title).replace(/\s+/g, " "),
      link: l,
      guid: txt(x.guid ?? x.id) || l || txt(x.title),
      publishedAt: iso(x.pubDate ?? x.published ?? x.updated),
    };
  }).filter((x) => x.title && x.guid);
}

function relevant(e: EP, t: string) {
  if (e.endpoint_id === "fed_monetary") return /fomc|federal reserve|monetary policy|interest rate|federal funds|economic projections|minutes|statement/i.test(t);
  if (e.endpoint_id === "ecb_press") return /monetary policy|interest rate|governing council|policy decision|inflation|rates?/i.test(t);
  if (e.endpoint_id === "bls_cpi" || e.endpoint_id === "bls_employment") return true;
  return /crypto|digital asset|bitcoin|ethereum|ether|stablecoin|blockchain|exchange-traded|\betf\b|token/i.test(t);
}
const macro = (c: string) => /CENTRAL_BANK|MACRO_|FUNDING_LIQUIDITY/i.test(c);

function explicit(asset: unknown, title: string, eligible: Set<string>) {
  const out = new Set<string>();
  const raw = String(asset ?? "").toUpperCase().replace(/^CRYPTO:/, "").replace(/[^A-Z0-9]/g, "");
  if (raw && !/^(GLOBAL|GLOBALWORLD|MACRO|CRYPTO)$/.test(raw)) {
    const s = raw.endsWith("USDT") ? raw : `${raw}USDT`;
    if (eligible.has(s)) out.add(`crypto:${s}`);
  }
  const m: [RegExp, string][] = [
    [/\bbitcoin\b|\bBTC\b/i, "BTCUSDT"],
    [/\bethereum\b|\bether\b|\bETH\b/i, "ETHUSDT"],
    [/\bsolana\b|\bSOL\b/i, "SOLUSDT"],
    [/\bcardano\b|\bADA\b/i, "ADAUSDT"],
    [/\bripple\b|\bXRP\b/i, "XRPUSDT"],
    [/\bbnb\b|binance coin/i, "BNBUSDT"],
  ];
  for (const [r, s] of m) if (r.test(title) && eligible.has(s)) out.add(`crypto:${s}`);
  return [...out];
}

async function universe() {
  const s = new Set(CORE.map((x) => x.slice(7)));
  const q = await db.from("brian_universe_snapshots").select("candidates").order("observed_at", { ascending: false }).limit(1).maybeSingle();
  if (q.error) throw q.error;
  const e = (q.data?.candidates ?? {}) as any;
  for (const x of Array.isArray(e.eligible_symbols) ? e.eligible_symbols : []) {
    if (/^[A-Z0-9]+USDT$/.test(String(x))) s.add(String(x));
  }
  return s;
}

async function books() {
  const r = await fetch("https://api.binance.com/api/v3/ticker/bookTicker", {
    signal: AbortSignal.timeout(3500),
    headers: { "user-agent": "Brian-Catalyst-Sentinel/2.0" },
  });
  if (!r.ok) throw Error(`BINANCE_BOOK_${r.status}`);
  const p = await r.json();
  const m = new Map<string, Book>();
  for (const x of p) {
    const b = finite(x.bidPrice), a = finite(x.askPrice);
    if (b > 0 && a >= b) {
      const mid = (a + b) / 2;
      m.set(`crypto:${x.symbol}`, { mid, spread: 10000 * (a - b) / mid });
    }
  }
  return m;
}

async function depth(asset: string) {
  try {
    const s = asset.slice(7);
    const r = await fetch(`https://api.binance.com/api/v3/depth?symbol=${s}&limit=20`, {
      signal: AbortSignal.timeout(2500),
      headers: { "user-agent": "Brian-Catalyst-Sentinel/2.0" },
    });
    if (!r.ok) return 0;
    const p = await r.json();
    const sum = (x: any) => arr<any>(x).reduce((n, y) => n + finite(y?.[0]) * finite(y?.[1]), 0);
    const b = sum(p.bids), a = sum(p.asks);
    return b + a ? (b - a) / (b + a) : 0;
  } catch {
    return 0;
  }
}

async function endpoints() {
  const q = await db.from("brian_source_registry_status_v2")
    .select("endpoint_id,source_id,organization,canonical_domain,endpoint_url,category,tier")
    .in("endpoint_id", FAST)
    .eq("lifecycle_state", "ACTIVE")
    .eq("eligible_for_research", true);
  if (q.error) throw q.error;
  return (q.data ?? []) as EP[];
}

function recentItems(items: Item[], nowMs: number, maxAgeMs: number) {
  return items.filter((i) => {
    if (!i.publishedAt) return false;
    const t = Date.parse(i.publishedAt);
    return Number.isFinite(t) && t <= nowMs + 60_000 && nowMs - t <= maxAgeMs;
  });
}

async function feed(e: EP, now: string) {
  let priorFailures = 0;
  try {
    const prior = await db.from("brian_catalyst_sentinel_feed_state")
      .select("last_content_hash,consecutive_failures")
      .eq("endpoint_id", e.endpoint_id)
      .maybeSingle();
    if (prior.error) throw prior.error;
    priorFailures = finite(prior.data?.consecutive_failures);
    const r = await fetch(e.endpoint_url, {
      redirect: "follow",
      signal: AbortSignal.timeout(3000),
      headers: {
        accept: "application/rss+xml,application/atom+xml,application/xml,text/xml",
        "user-agent": "Brian-Catalyst-Sentinel/2.0",
        "cache-control": "no-cache",
      },
    });
    if (!r.ok) throw Error(`HTTP_${r.status}`);
    const xml = await r.text();
    const h = await sha(xml);
    const isBaseline = !prior.data?.last_content_hash;
    const changed = !isBaseline && h !== prior.data?.last_content_hash;
    const parsed = parse(xml);
    const nowMs = Date.parse(now);
    const items = isBaseline
      ? recentItems(parsed, nowMs, BASELINE_RECENCY_MS)
      : changed ? recentItems(parsed, nowMs, FEED_RECENCY_MS) : [];
    const up = await db.from("brian_catalyst_sentinel_feed_state").upsert({
      endpoint_id: e.endpoint_id,
      last_checked_at: now,
      last_changed_at: changed || (isBaseline && items.length) ? now : (prior.data ? undefined : null),
      last_content_hash: h,
      consecutive_failures: 0,
      last_error: null,
      updated_at: now,
    }, { onConflict: "endpoint_id" });
    if (up.error) throw up.error;
    return { e, items, baseline: isBaseline, changed };
  } catch (x) {
    const nowIso = new Date().toISOString();
    const up = await db.from("brian_catalyst_sentinel_feed_state").upsert({
      endpoint_id: e.endpoint_id,
      last_checked_at: nowIso,
      consecutive_failures: priorFailures + 1,
      last_error: err(x).slice(0, 600),
      updated_at: nowIso,
    }, { onConflict: "endpoint_id" });
    return { e, items: [] as Item[], error: `${err(x)}${up.error ? `; state:${up.error.message}` : ""}` };
  }
}

async function event(e: EP, i: Item, at: string) {
  const id = await sha(`source-arch-v2|${e.endpoint_id}|${i.guid}`);
  const fp = await sha(`${e.endpoint_id}|${i.title}|${i.link}`);
  const lat = i.publishedAt ? Math.max(0, Date.parse(at) - Date.parse(i.publishedAt)) : null;
  const row: any = {
    event_id: id,
    asset: macro(e.category) ? "MACRO" : "GLOBAL_WORLD",
    event_kind: macro(e.category) ? "OFFICIAL_MACRO_RELEASE" : "OFFICIAL_SOURCE_ITEM",
    source_kind: `CATALYST_FAST_${e.tier}`,
    source_id: e.source_id,
    published_at: i.publishedAt,
    first_observed_at: at,
    captured_at: new Date().toISOString(),
    claim: i.title,
    direction: 0,
    magnitude: 1,
    trust_class: "OFFICIAL_PRIMARY",
    entity_confidence: 1,
    content_fingerprint: fp,
    corroboration_key: await sha(`${e.category}|${i.title.toLowerCase().replace(/[^a-z0-9]+/g, " ")}`),
    provenance_uri: i.link || e.endpoint_url,
    pit_verified: true,
    raw_capture_id: null,
    metadata: {
      runtime: "CATALYST_SENTINEL_FAST_LANE",
      version: V,
      endpoint_id: e.endpoint_id,
      organization: e.organization,
      category: e.category,
      first_http_seen_at: at,
      stub_inserted_at: new Date().toISOString(),
      detection_latency_ms: lat,
      canonical_event_id: id,
      correlation_id: id,
      direction_not_inferred: true,
      asset_mapping_policy: "macro_core_or_explicit_only",
      random_asset_mapping: false,
    },
  };
  const q = await db.from("brian_intel_events").upsert(row, { onConflict: "event_id", ignoreDuplicates: true });
  if (q.error) throw q.error;
  if (macro(e.category)) {
    const oid = await sha(`${V}|macro|${id}`);
    const s = await db.from("brian_sensor_observations").upsert({
      observation_id: oid,
      eye_id: "brian-catalyst-sentinel",
      template_id: "official-macro-fast-lane-v2",
      asset_id: "global:MACRO",
      market_domain: "macro",
      sensor_family: "official_macro_event",
      horizon: "EVENT_DRIVEN",
      independent_group: `official_macro_${e.endpoint_id}`,
      observed_at: at,
      direction: 0,
      strength: 1,
      confidence: 1,
      reliability: 1,
      available: true,
      source_ids: [e.source_id],
      reason: `${e.organization} official release observed; direction delegated to market reaction`,
      evidence_class: E,
      shadow_only: true,
      live_execution: false,
      metadata: {
        event_id: id,
        correlation_id: id,
        organization: e.organization,
        title: i.title,
        published_at: i.publishedAt,
        provenance_uri: i.link || e.endpoint_url,
        detection_latency_ms: lat,
      },
    }, { onConflict: "observation_id", ignoreDuplicates: true });
    if (s.error) throw s.error;
  }
  return row;
}

async function watch(ev: any, asset: string, b?: Book) {
  const id = await sha(`${V}|watch|${ev.event_id}|${asset}`);
  const st = String(ev.first_observed_at);
  const row: any = {
    watch_id: id,
    event_id: ev.event_id,
    asset_id: asset,
    source_id: ev.source_id,
    event_kind: ev.event_kind,
    event_title: ev.claim,
    event_published_at: ev.published_at,
    started_at: st,
    expires_at: new Date(Date.parse(st) + WATCH_TTL_MS).toISOString(),
    status: "WATCHING",
    direction: 0,
    reference_price: b?.mid ?? null,
    last_price: b?.mid ?? null,
    last_return: 0,
    reaction_score: 0,
    mfe: 0,
    mae: 0,
    last_evaluated_at: new Date().toISOString(),
    last_state_change_at: new Date().toISOString(),
    recheck_count: 0,
    next_recheck_at: new Date(Date.parse(st) + RECHECK[0] * 60_000).toISOString(),
    metadata: { version: V, correlation_id: ev.event_id, asset_binding: "macro_core_or_explicit_only", random_asset_mapping: false },
    evidence_class: E,
    shadow_only: true,
    live_execution: false,
    updated_at: new Date().toISOString(),
  };
  const existing = await db.from("brian_catalyst_sentinel_watches")
    .select("watch_id").eq("event_id", ev.event_id).eq("asset_id", asset).maybeSingle();
  if (existing.error) throw existing.error;
  if (existing.data?.watch_id) return null;
  const q = await db.from("brian_catalyst_sentinel_watches").insert(row).select("watch_id").maybeSingle();
  if (q.error) {
    if ((q.error as any).code === "23505") return null;
    throw q.error;
  }
  return q.data?.watch_id ? row : null;
}

async function updateAlertDispatch(alertId: string, patch: Record<string, unknown>, metaPatch: Record<string, unknown>) {
  const q = await db.from("brian_catalyst_sentinel_alerts").select("metadata").eq("alert_id", alertId).maybeSingle();
  const metadata = { ...((q.data?.metadata ?? {}) as Record<string, unknown>), ...metaPatch };
  const u = await db.from("brian_catalyst_sentinel_alerts").update({ ...patch, metadata }).eq("alert_id", alertId);
  if (u.error) throw u.error;
}

async function dispatch(eventId: string, assetId: string, alertId: string, key: string) {
  const token = ANON || SERVICE;
  const attemptedAt = new Date().toISOString();
  try {
    const r = await fetch(`${URL}/functions/v1/brian-alpha-event-recheck`, {
      method: "POST",
      headers: { "content-type": "application/json", Authorization: `Bearer ${token}`, apikey: token, "x-brian-cron-key": key },
      body: JSON.stringify({ event_id: eventId, asset_id: assetId, alert_id: alertId }),
      signal: AbortSignal.timeout(DISPATCH_TIMEOUT_MS),
    });
    const raw = await r.text();
    let j: any = {};
    try { j = raw ? JSON.parse(raw) : {}; } catch { j = {}; }
    await updateAlertDispatch(alertId, {
      alpha_dispatched: r.ok,
      alpha_dispatch_request_id: String(j?.decision_id ?? `HTTP_${r.status}`),
      alpha_dispatched_at: attemptedAt,
    }, {
      last_dispatch_attempt_at: attemptedAt,
      alpha_dispatch_http_status: r.status,
      alpha_dispatch_error: r.ok ? null : String(j?.error ?? raw ?? `HTTP_${r.status}`).slice(0, 600),
    });
    return r.ok;
  } catch (x) {
    await updateAlertDispatch(alertId, { alpha_dispatched: false, alpha_dispatch_request_id: "NETWORK_ERROR", alpha_dispatched_at: attemptedAt }, {
      last_dispatch_attempt_at: attemptedAt,
      alpha_dispatch_error: err(x).slice(0, 600),
    });
    return false;
  }
}

async function alert(w: any, type: string, dir: number, score: number, m: Reaction, key: string) {
  const id = await sha(`${V}|alert|${w.watch_id}|${type}|${Math.floor(Date.now() / ALERT_BUCKET_MS)}`);
  const row = {
    alert_id: id, watch_id: w.watch_id, event_id: w.event_id, asset_id: w.asset_id,
    observed_at: new Date().toISOString(), alert_type: type, direction: dir, reaction_score: score,
    price_return: m?.ret ?? null, spread_bps: m?.spread ?? null, orderbook_imbalance: m?.imb ?? null,
    realized_range_bps: m?.range ?? null, cross_market_confirmation: m?.cross ?? null,
    metadata: { version: V, volume_confirmation: m?.vol ?? null, hard_risk_gates_bypassed: false, scheduling_priority_bypass_only: true },
    evidence_class: E, shadow_only: true, live_execution: false,
  };
  const q = await db.from("brian_catalyst_sentinel_alerts").insert(row).select("alert_id").maybeSingle();
  if (q.error) {
    if ((q.error as any).code === "23505") return { id, inserted: false, dispatched: false };
    throw q.error;
  }
  if (!q.data?.alert_id) return { id, inserted: false, dispatched: false };
  const dispatched = await dispatch(w.event_id, w.asset_id, id, key);
  return { id, inserted: true, dispatched };
}

async function sensor(w: any, dir: number, score: number, m: Reaction) {
  if (!dir || score < .15) return;
  const id = await sha(`${V}|sensor|${w.event_id}|${w.asset_id}|${Math.floor(Date.now() / 30_000)}`);
  const q = await db.from("brian_sensor_observations").upsert({
    observation_id: id, eye_id: "brian-catalyst-sentinel", template_id: "catalyst-market-reaction-v2",
    asset_id: w.asset_id, market_domain: "CRYPTO", sensor_family: "catalyst_sentinel_reaction", horizon: "EVENT_DRIVEN",
    independent_group: "catalyst_sentinel_reaction", observed_at: new Date().toISOString(), direction: dir,
    strength: clip(score), confidence: clip(.6 + .3 * score), reliability: .65, available: true,
    source_ids: [w.source_id], reason: `Catalyst Sentinel reaction for ${w.event_id}`, evidence_class: E,
    shadow_only: true, live_execution: false,
    metadata: { event_id: w.event_id, correlation_id: w.event_id, watch_id: w.watch_id, reaction_score: score,
      price_return: m.ret, spread_bps: m.spread, orderbook_imbalance: m.imb, realized_range_bps: m.range,
      cross_market_confirmation: m.cross, volume_confirmation: m.vol, news_direction_inferred: false, market_reaction_direction: true },
  }, { onConflict: "observation_id", ignoreDuplicates: true });
  if (q.error) throw q.error;
}

async function ranges(assets: string[]) {
  const m = new Map<string, number>();
  const q = await db.from("brian_micro_book_ticks").select("asset_id,observed_mid_price").in("asset_id", assets)
    .gte("observed_at", new Date(Date.now() - 5 * 60_000).toISOString()).limit(5000);
  if (q.error) throw q.error;
  const v = new Map<string, number[]>();
  for (const r of q.data ?? []) {
    const a = String(r.asset_id), p = finite(r.observed_mid_price);
    if (p > 0) { const z = v.get(a) ?? []; z.push(p); v.set(a, z); }
  }
  for (const [a, z] of v) if (z.length > 1) m.set(a, 10000 * (Math.max(...z) - Math.min(...z)) / z[0]);
  return m;
}

async function volumes(assets: string[]) {
  const m = new Map<string, { d: number; s: number }>();
  const q = await db.from("brian_sensor_observations").select("asset_id,direction,strength,confidence,reliability,observed_at")
    .in("asset_id", assets).eq("independent_group", "micro_volume").eq("available", true)
    .gte("observed_at", new Date(Date.now() - 5 * 60_000).toISOString()).order("observed_at", { ascending: false }).limit(500);
  if (q.error) throw q.error;
  for (const r of q.data ?? []) {
    const a = String(r.asset_id);
    if (!m.has(a)) m.set(a, { d: Number(r.direction), s: clip(finite(r.strength) * finite(r.confidence) * finite(r.reliability)) });
  }
  return m;
}

async function monitor(ws: any[], bm: Map<string, Book>, key: string) {
  if (!ws.length) return { alerts: 0, dispatches: 0, errors: [] as string[] };
  const assets = [...new Set(ws.map((w) => String(w.asset_id)))];
  const [rg, vo, imPairs] = await Promise.all([ranges(assets), volumes(assets), Promise.all(assets.map(async (a) => [a, await depth(a)] as const))]);
  const im = new Map(imPairs);
  const cur = new Map<string, { ret: number; dir: number }>();
  for (const w of ws) {
    const b = bm.get(w.asset_id), r = finite(w.reference_price);
    if (b && r > 0) { const x = b.mid / r - 1; cur.set(w.asset_id, { ret: x, dir: x > .00005 ? 1 : x < -.00005 ? -1 : 0 }); }
  }
  const results = await Promise.all(ws.map(async (w) => {
    try {
      const b = bm.get(w.asset_id), ref = finite(w.reference_price);
      if (!b || ref <= 0) return { alerts: 0, dispatches: 0 };
      const ret = b.mid / ref - 1, dir = ret > .00005 ? 1 : ret < -.00005 ? -1 : 0, ab = Math.abs(ret) * 10000;
      const imb = im.get(w.asset_id) ?? 0, ran = rg.get(w.asset_id) ?? 0, vs = vo.get(w.asset_id), vol = dir && vs?.d === dir ? vs.s : 0;
      const peers = CORE.filter((a) => a !== w.asset_id && cur.has(a)).slice(0, 2);
      let cross = .5;
      if (peers.length) { let ok = 0, use = 0; for (const p of peers) { const x = cur.get(p)!; if (Math.abs(x.ret) >= .0005) { use++; if (x.dir === dir) ok++; } } if (use) cross = ok / use; }
      const score = clip(.35 * clip(ab / 80) + .18 * clip((dir * imb + .05) / .35) + .12 * clip(ran / 120) + .15 * cross + .20 * vol - .12 * clip(b.spread / 30));
      const st = ab >= 60 && score >= .68 && b.spread <= 25 ? "CONFIRMED" : ab >= 35 && score >= .55 && b.spread <= 30 ? "BREAKOUT_CANDIDATE" : ab >= 15 || score >= .34 ? "BUILDING" : "WATCHING";
      const now = new Date().toISOString(), changed = st !== w.status, flip = w.direction && dir && w.direction !== dir && ab >= 20;
      const mins = (Date.now() - Date.parse(w.started_at)) / 60_000;
      let rc = Number(w.recheck_count || 0), timed: string | null = null;
      if (rc < RECHECK.length && mins >= RECHECK[rc]) { timed = `TIMED_RECHECK_${RECHECK[rc]}M`; rc++; }
      const met: Reaction = { ret, spread: b.spread, imb, range: ran, cross, vol };
      const up: any = { status: st, direction: dir, last_price: b.mid, last_return: ret, reaction_score: score,
        mfe: Math.max(finite(w.mfe), ret), mae: Math.min(finite(w.mae), ret), spread_bps: b.spread,
        orderbook_imbalance: imb, realized_range_bps: ran, cross_market_confirmation: cross, last_evaluated_at: now,
        last_state_change_at: changed ? now : w.last_state_change_at, recheck_count: rc,
        next_recheck_at: rc < RECHECK.length ? new Date(Date.parse(w.started_at) + RECHECK[rc] * 60000).toISOString() : null, updated_at: now };
      const uq = await db.from("brian_catalyst_sentinel_watches").update(up).eq("watch_id", w.watch_id).select("watch_id").maybeSingle();
      if (uq.error || !uq.data) throw Error(`WATCH_UPDATE_FAILED:${uq.error?.message ?? "missing"}`);
      const live = { ...w, ...up };
      await sensor(live, dir, score, met);
      const cooldown = !w.last_alpha_trigger_at || Date.now() - Date.parse(w.last_alpha_trigger_at) >= ALERT_COOLDOWN_MS;
      const type = flip ? "DIRECTION_FLIP" : changed && st !== "WATCHING" ? st : timed;
      if (!type || !cooldown) return { alerts: 0, dispatches: 0 };
      const a = await alert(live, type, dir, score, met, key);
      if (a.inserted && a.dispatched) {
        const lu = await db.from("brian_catalyst_sentinel_watches").update({ last_alpha_trigger_at: now, updated_at: now }).eq("watch_id", w.watch_id);
        if (lu.error) throw lu.error;
      }
      return { alerts: a.inserted ? 1 : 0, dispatches: a.dispatched ? 1 : 0 };
    } catch (x) {
      return { alerts: 0, dispatches: 0, error: `${w.watch_id}:${err(x)}` };
    }
  }));
  return { alerts: results.reduce((n, x) => n + x.alerts, 0), dispatches: results.reduce((n, x) => n + x.dispatches, 0), errors: results.map((x) => x.error).filter(Boolean) as string[] };
}

async function discover(eligible: Set<string>, bm: Map<string, Book>) {
  const q = await db.from("brian_intel_events").select("event_id,asset,event_kind,source_id,published_at,first_observed_at,claim,trust_class,metadata")
    .gte("first_observed_at", new Date(Date.now() - 20 * 60_000).toISOString())
    .in("trust_class", ["OFFICIAL_PRIMARY", "INSTITUTIONAL", "INDEPENDENT_PROFESSIONAL"])
    .order("first_observed_at", { ascending: false }).limit(300);
  if (q.error) throw q.error;
  const out: any[] = [];
  for (const e of q.data ?? []) {
    const md = (e.metadata ?? {}) as Record<string, unknown>, cat = String(md.category ?? "");
    const filing = /sec_edgar_current/i.test(String(e.source_id)) || /CORPORATE_FILINGS/i.test(cat);
    if (filing && !/crypto|digital asset|bitcoin|ethereum|stablecoin|blockchain|\betf\b|token/i.test(String(e.claim))) continue;
    let a = explicit(e.asset, e.claim, eligible);
    if (macro(cat) || String(e.asset).toUpperCase() === "MACRO" || /OFFICIAL_MACRO/i.test(String(e.event_kind))) a = CORE.filter((x) => eligible.has(x.slice(7)));
    for (const x of a) { const w = await watch(e, x, bm.get(x)); if (w) out.push(w); }
  }
  return out;
}

async function expireOld(now: string) {
  const q = await db.from("brian_catalyst_sentinel_watches").update({ status: "EXPIRED", updated_at: now, last_state_change_at: now })
    .in("status", ACTIVE).lte("expires_at", now).select("watch_id");
  if (q.error) throw q.error;
  return (q.data ?? []).length;
}

async function retryPending(key: string) {
  const q = await db.from("brian_catalyst_sentinel_alerts").select("alert_id,event_id,asset_id,metadata,observed_at")
    .eq("alpha_dispatched", false).gte("observed_at", new Date(Date.now() - 10 * 60_000).toISOString())
    .order("observed_at", { ascending: true }).limit(3);
  if (q.error) throw q.error;
  let n = 0;
  for (const a of q.data ?? []) {
    const last = Date.parse(String((a.metadata as any)?.last_dispatch_attempt_at ?? ""));
    if (Number.isFinite(last) && Date.now() - last < 30_000) continue;
    if (await dispatch(String(a.event_id), String(a.asset_id), String(a.alert_id), key)) n++;
  }
  return n;
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return new Response("POST", { status: 405 });
  try { await auth(req); }
  catch (e) { return new Response(JSON.stringify({ status: "UNAUTHORIZED", error: err(e) }), { status: 401, headers: { "content-type": "application/json" } }); }
  const key = (req.headers.get("x-brian-cron-key") ?? "").trim();
  try {
    const at = new Date().toISOString();
    const expired = await expireOld(at);
    const [el, bm, eps] = await Promise.all([universe(), books(), endpoints()]);
    const fr = await Promise.all(eps.map((e) => feed(e, at)));
    const degraded = fr.filter((x: any) => x.error).map((x: any) => `${x.e.endpoint_id}:${x.error}`);
    const evs: any[] = [];
    for (const x of fr) for (const i of x.items) if (relevant(x.e, i.title)) evs.push(await event(x.e, i, at));
    const created: any[] = [];
    for (const e of evs) {
      let aa = explicit(e.asset, e.claim, el);
      if (macro(String(e.metadata.category))) aa = CORE.filter((x) => el.has(x.slice(7)));
      for (const a of aa) { const w = await watch(e, a, bm.get(a)); if (w) created.push(w); }
    }
    created.push(...await discover(el, bm));
    const detected = await Promise.all(created.map(async (w) => {
      const a = await alert(w, "EVENT_DETECTED", 0, 0, { ret: 0, spread: bm.get(w.asset_id)?.spread ?? null }, key);
      if (a.inserted && a.dispatched) {
        const now = new Date().toISOString();
        const u = await db.from("brian_catalyst_sentinel_watches").update({ last_alpha_trigger_at: now, updated_at: now }).eq("watch_id", w.watch_id);
        if (u.error) throw u.error;
      }
      return a;
    }));
    const q = await db.from("brian_catalyst_sentinel_watches").select("*").in("status", ACTIVE)
      .gt("expires_at", new Date().toISOString()).order("started_at", { ascending: false }).limit(60);
    if (q.error) throw q.error;
    const monitored = await monitor(q.data ?? [], bm, key);
    const retried = await retryPending(key);
    const feedBaselines = fr.filter((x: any) => x.baseline).length, feedChanges = fr.filter((x: any) => x.changed).length;
    const detectedAlerts = detected.filter((x) => x.inserted).length, detectedDispatches = detected.filter((x) => x.dispatched).length;
    return new Response(JSON.stringify({
      status: degraded.length || monitored.errors.length ? "DEGRADED" : "SUCCESS", version: V,
      feeds_checked: eps.length, feed_baselines: feedBaselines, feed_changes: feedChanges, events_detected: evs.length,
      watches_created: created.length, watches_expired: expired, watches_evaluated: (q.data ?? []).length,
      alerts_created: detectedAlerts + monitored.alerts, alpha_dispatches: detectedDispatches + monitored.dispatches + retried,
      pending_dispatches_retried: retried, monitor_errors: monitored.errors.slice(0, 10), degraded,
      fast_lane_seconds: 10, persistent_rechecks_minutes: RECHECK, watch_ttl_minutes: WATCH_TTL_MS / 60000,
      asset_binding: "macro_core_or_explicit_only", random_asset_mapping: false, hard_risk_gates_bypassed: false,
      scheduling_priority_bypass_only: true, shadow_only: true, live_execution: false,
    }), { headers: { "content-type": "application/json", "cache-control": "no-store" } });
  } catch (e) {
    return new Response(JSON.stringify({ status: "FAILED_CLOSED", error: err(e), version: V, shadow_only: true, live_execution: false }), {
      status: 500, headers: { "content-type": "application/json", "cache-control": "no-store" },
    });
  }
});
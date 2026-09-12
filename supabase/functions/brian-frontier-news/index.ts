import { createClient } from "npm:@supabase/supabase-js@2.116.0";

const URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(URL, SERVICE, { auth: { persistSession: false, autoRefreshToken: false } });
const AUTH_ID = "control-v3";
const ALLOWED_ORIGIN = /^https:\/\/monster-coins(?:-pro)?-[a-z0-9-]*oemer-yildirim\.vercel\.app$/i;
const ALLOWED_EXACT = new Set([
  "https://monster-coins-pro-seven.vercel.app",
  "https://monster-coins-pro-oemer-yildirim.vercel.app",
  "https://monster-coins-pro-git-brian-2026-oemer-yildirim.vercel.app",
  "http://localhost:3000",
  "http://127.0.0.1:3000",
]);

function cors(origin?: string | null): Record<string, string> {
  const allowed = origin && (ALLOWED_EXACT.has(origin) || ALLOWED_ORIGIN.test(origin)) ? origin : "https://monster-coins-pro-oemer-yildirim.vercel.app";
  return {
    "access-control-allow-origin": allowed,
    "access-control-allow-headers": "content-type,x-brian-dashboard-key",
    "access-control-allow-methods": "POST,OPTIONS",
    vary: "Origin",
  };
}
function out(body: unknown, status = 200, origin?: string | null) {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store", ...cors(origin) } });
}
async function sha256Hex(value: string) {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value)));
  return [...digest].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}
function constantTimeEqual(left: string, right: string) {
  if (left.length !== right.length) return false;
  let diff = 0;
  for (let index = 0; index < left.length; index++) diff |= left.charCodeAt(index) ^ right.charCodeAt(index);
  return diff === 0;
}
async function auth(req: Request) {
  const supplied = (req.headers.get("x-brian-dashboard-key") ?? "").trim();
  if (!supplied) throw new Error("UNAUTHORIZED_DASHBOARD");
  const q = await db.from("brian_dashboard_auth").select("dashboard_key_sha256").eq("auth_id", AUTH_ID).single();
  if (q.error || !q.data) throw new Error("DASHBOARD_AUTH_UNAVAILABLE");
  if (!constantTimeEqual(await sha256Hex(supplied), String(q.data.dashboard_key_sha256 ?? ""))) throw new Error("UNAUTHORIZED_DASHBOARD");
}

const LABEL_TR: Record<string, string> = {
  "narrative:GEOPOLITICS": "Jeopolitik risk",
  "narrative:MONETARY_POLICY": "Para politikası",
  "narrative:INFLATION": "Enflasyon",
  "narrative:ENERGY_SUPPLY": "Enerji arzı",
  "narrative:CRYPTO_REGULATION": "Kripto düzenlemeleri",
  "narrative:ETF_FLOWS": "ETF para akışları",
  "narrative:AI_COMPUTE": "Yapay zekâ / hesaplama talebi",
  "narrative:SEMICONDUCTOR_SUPPLY": "Yarı iletken tedarik zinciri",
  "narrative:CYBERSECURITY": "Siber güvenlik riski",
  "narrative:PRODUCT_LAUNCH": "Ürün / teknoloji lansmanı",
  "narrative:EARNINGS": "Şirket finansal sonuçları",
  "narrative:TOKEN_EVENT": "Token / listeleme olayı",
};
const ENTITY_TR: Record<string, string> = {
  "person:TRUMP": "Donald Trump",
  "person:MUSK": "Elon Musk",
  "centralbank:FED": "ABD Merkez Bankası (Fed)",
  "centralbank:ECB": "Avrupa Merkez Bankası (ECB)",
  "regulator:SEC": "ABD SEC",
  "commodity:OIL": "petrol",
  "commodity:GOLD": "altın",
  "asset:BTC": "Bitcoin",
  "asset:ETH": "Ethereum",
  "company:NVIDIA": "NVIDIA",
  "company:APPLE": "Apple",
  "company:TSMC": "TSMC",
  "company:OPENAI": "OpenAI",
  "technology:AI": "yapay zekâ",
};

type Frame = {
  frame_id: string;
  event_id: string;
  observed_at: string;
  published_at: string | null;
  event_kind: string;
  source_id: string;
  claim: string;
  primary_asset: string | null;
  entity_ids: string[] | null;
  narrative_ids: string[] | null;
  source_trust_class: string;
  provenance_uri: string | null;
};

function importantScore(row: Frame): number {
  const narratives = row.narrative_ids ?? [];
  const entities = row.entity_ids ?? [];
  let score = 0.25;
  if (narratives.some((id) => ["narrative:GEOPOLITICS", "narrative:MONETARY_POLICY", "narrative:INFLATION", "narrative:ENERGY_SUPPLY", "narrative:EARNINGS", "narrative:PRODUCT_LAUNCH", "narrative:CYBERSECURITY", "narrative:CRYPTO_REGULATION"].includes(id))) score += 0.28;
  if (entities.some((id) => id === "person:TRUMP" || id === "person:MUSK" || id.startsWith("centralbank:") || id === "regulator:SEC")) score += 0.22;
  if (row.primary_asset) score += 0.12;
  if (/PRIMARY|OFFICIAL|REGULATOR|EXCHANGE|FILING/i.test(row.source_trust_class)) score += 0.10;
  if (/war|conflict|missile|sanction|rate|inflation|cpi|earnings|launch|hack|exploit|tariff|election|regulat/i.test(row.claim)) score += 0.12;
  return Math.min(1, score);
}
function turkishSummary(row: Frame): { title: string; summary: string } {
  const narratives = row.narrative_ids ?? [];
  const entities = row.entity_ids ?? [];
  const person = entities.find((id) => id === "person:TRUMP" || id === "person:MUSK");
  const central = entities.find((id) => id.startsWith("centralbank:"));
  const entity = person ?? central ?? entities.find((id) => ENTITY_TR[id]);
  const narrative = narratives.find((id) => LABEL_TR[id]);
  const subject = entity ? ENTITY_TR[entity] : narrative ? LABEL_TR[narrative] : row.primary_asset || "Küresel piyasa";
  let title = `${subject}: Brian için önemli gelişme`;
  let summary = "Brian bu gelişmeyi fiyatlama, risk ve olası ikinci-order etkiler açısından takip ediyor.";
  if (person === "person:TRUMP") { title = "Donald Trump kaynaklı piyasa etkisi taşıyan gelişme"; summary = "Trump ile ilişkili yeni açıklama/gelişme; Brian olası politika, risk iştahı ve varlık fiyatlama etkisini izliyor."; }
  else if (person === "person:MUSK") { title = "Elon Musk / teknoloji kaynaklı gelişme"; summary = "Musk ile ilişkili yeni gelişme; Brian teknoloji anlatısı, şirket beklentileri ve crowd tepkisini izliyor."; }
  else if (narrative === "narrative:GEOPOLITICS") { title = "Jeopolitik riskte önemli gelişme"; summary = "Savaş, yaptırım, çatışma veya güvenlik riskiyle ilişkili gelişme; enerji, güvenli limanlar ve risk iştahı etkileri izleniyor."; }
  else if (narrative === "narrative:MONETARY_POLICY") { title = `${central ? ENTITY_TR[central] : "Merkez bankası"}: politika beklentisi değişiyor`; summary = "Faiz/likidite beklentisini etkileyebilecek gelişme; tahvil, dolar, altın ve risk varlıklarına geçiş etkisi izleniyor."; }
  else if (narrative === "narrative:EARNINGS") { title = "Şirket sonuçları / beklenti sürprizi"; summary = "Gelir, kâr veya yönlendirme kaynaklı beklenti farkı; fiyatlanmış kısım ile yeni bilgi ayrıştırılıyor."; }
  else if (narrative === "narrative:PRODUCT_LAUNCH") { title = "Yeni teknoloji / ürün katalizörü"; summary = "Ürün veya teknoloji gelişmesinin beklenti, tedarik zinciri ve ilgili hisseler üzerindeki etkisi izleniyor."; }
  else if (narrative === "narrative:ENERGY_SUPPLY") { title = "Enerji arzı / petrol dengesi değişimi"; summary = "Fiziksel arz ve lojistik etkileri; petrol, enflasyon, faiz ve risk varlıklarına ikinci-order geçiş izleniyor."; }
  return { title, summary };
}

Deno.serve(async (req: Request) => {
  const origin = req.headers.get("origin");
  if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: cors(origin) });
  if (req.method !== "POST") return out({ error: "POST required" }, 405, origin);
  try { await auth(req); } catch (error) { return out({ error: String(error) }, 401, origin); }
  const q = await db.from("brian_world_event_frames")
    .select("frame_id,event_id,observed_at,published_at,event_kind,source_id,claim,primary_asset,entity_ids,narrative_ids,source_trust_class,provenance_uri")
    .order("observed_at", { ascending: false }).limit(140);
  if (q.error) return out({ status: "DEGRADED", error: q.error.message, items: [], shadow_only: true, live_execution: false }, 500, origin);
  const items = ((q.data ?? []) as Frame[])
    .map((row) => ({ row, score: importantScore(row), tr: turkishSummary(row) }))
    .filter((item) => item.score >= 0.5)
    .sort((a, b) => b.score - a.score || Date.parse(b.row.observed_at) - Date.parse(a.row.observed_at))
    .slice(0, 24)
    .map(({ row, score, tr }) => ({
      id: row.frame_id,
      observed_at: row.observed_at,
      published_at: row.published_at,
      urgency: score >= 0.8 ? "CRITICAL" : score >= 0.65 ? "HIGH" : "MEDIUM",
      importance: score,
      title_tr: tr.title,
      summary_tr: tr.summary,
      original_claim: row.claim,
      event_kind: row.event_kind,
      source_id: row.source_id,
      source_trust_class: row.source_trust_class,
      primary_asset: row.primary_asset,
      entity_ids: row.entity_ids ?? [],
      narrative_ids: row.narrative_ids ?? [],
      provenance_uri: row.provenance_uri,
    }));
  return out({ status: "ONLINE", observed_at: new Date().toISOString(), items, semantics: { filtered_for_brian_relevance: true, turkish_summary_is_structured_paraphrase_not_literal_translation: true, original_claim_preserved: true }, shadow_only: true, live_execution: false }, 200, origin);
});
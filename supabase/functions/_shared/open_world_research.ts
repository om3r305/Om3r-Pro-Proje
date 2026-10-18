// External material is evidence to investigate, never instructions or execution permission.
export const OPEN_WORLD_VERSION = "brian.open-world-research.v1";
export type ResearchHypothesis = { hypothesisId: string; hypothesisKind: string; problemStatement: string;
  proposedMechanism: string; targetCapabilities: string[]; evidenceRefs: string[]; priority: number };
export type ResearchMission = { key: string; hypothesisId: string | null; question: string;
  searchQuery: string; priority: number; evidenceRefs: string[]; successCriteria: string[] };
export type ResearchDocument = { provider: string; uri: string; title: string; excerpt: string;
  publishedAt: string | null; license: string | null; metadata: Record<string, unknown> };
export type LearningObservation = { kind: string; observation: string; metrics: Record<string, unknown>;
  evidenceRefs: string[]; nextExperiment: string };

function text(value: unknown, limit = 900): string {
  return String(value ?? "").replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, " ")
    .replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim().slice(0, limit);
}
export function publicDocumentUri(value: unknown): string | null {
  try {
    const uri = new URL(String(value));
    if (uri.protocol !== "https:" || uri.username || uri.password || (uri.port && uri.port !== "443")) return null;
    const host = uri.hostname.toLowerCase();
    if (!/^[a-z0-9.-]+$/.test(host) || !host.includes(".") || /^\d/.test(host) ||
        /(^|\.)(localhost|local|internal|test|invalid)$/.test(host)) return null;
    uri.hash = "";
    return uri.toString();
  } catch { return null; }
}
export function learningObservations(summary: Record<string, unknown>): LearningObservation[] {
  const n = (key: string) => typeof summary[key] === "number" && Number.isFinite(summary[key]) ? Number(summary[key]) : null;
  const mature = n("mature_5m"), resolved = n("resolved_5m");
  const refs = [`learning_window:${String(summary.window_start)}:${String(summary.window_end)}:latest_300`];
  const rows: LearningObservation[] = [];
  if (mature !== null && mature >= 20 && resolved !== null && resolved < mature * 0.95) rows.push({
    kind: "MEASUREMENT_COVERAGE", observation: "Outcome coverage is incomplete; resolved-only performance cannot justify promotion.",
    metrics: { mature, resolved, coverage: resolved / mature, sample_policy: "latest_300_decisions_not_random" }, evidenceRefs: refs,
    nextExperiment: "Separate clock-driven price observation from action decisions; retain strict horizon semantics and measure missingness by asset.",
  });
  const samples = n("action_samples_5m"), net = n("mean_action_net_bps_5m");
  if (samples !== null && samples >= 20 && net !== null && net < 0) rows.push({
    kind: "NEGATIVE_NET_EDGE", observation: "Resolved directional decisions have negative mean return after contemporaneous modeled cost.",
    metrics: { samples, net_bps: net, overlapping_samples: true, causal_explanation_proven: false }, evidenceRefs: refs,
    nextExperiment: "Compare a preregistered abstention/expected-edge challenger with the unchanged baseline on later data, including missed opportunities.",
  });
  return rows;
}
function topic(capabilities: string[]): string {
  const value = capabilities.join(" ").toLowerCase();
  if (/cost|edge|calibration/.test(value)) return "transaction cost expected returns backtest";
  if (/psych|crowd|behavior|sentiment/.test(value)) return "behavioral finance investor sentiment crowding";
  if (/world\.fx|foreign|currency/.test(value)) return "foreign exchange macroeconomic event study";
  if (/source|news|world/.test(value)) return "financial news event study causal impact";
  if (/market|price|telemetry|audit/.test(value)) return "market data event time sampling missing data";
  // Capability identifiers only: no logs, credentials, raw external text or private objectives in public search.
  return capabilities.filter(x => /^[a-z][a-z0-9._-]{1,70}$/i.test(x) && !/secret|auth|credential|token|password/i.test(x))
    .join(" ").replace(/[._-]/g, " ").split(/\s+/).filter(Boolean).slice(0, 6).join(" ") || "financial research reproducible evaluation";
}
export function researchMissions(hypotheses: ResearchHypothesis[], observations: LearningObservation[]): ResearchMission[] {
  const missions: ResearchMission[] = observations.map(row => ({ key: `lesson:${row.kind}`, hypothesisId: null,
    question: `${row.observation} ${row.nextExperiment}`, searchQuery: row.kind === "MEASUREMENT_COVERAGE"
      ? "market data event time sampling missing data" : "transaction cost expected returns backtest",
    priority: 1, evidenceRefs: row.evidenceRefs, successCriteria: [row.nextExperiment, "No future information in decision features", "Record failures and negative results"] }));
  for (const h of hypotheses.slice(0, 20)) missions.push({ key: `hypothesis:${h.hypothesisId}`, hypothesisId: h.hypothesisId,
    question: text(`${h.problemStatement} Investigate: ${h.proposedMechanism}`, 1600), searchQuery: topic(h.targetCapabilities),
    priority: Number.isFinite(h.priority) ? Math.max(0, Math.min(0.99, h.priority)) : 0.5,
    evidenceRefs: h.evidenceRefs.slice(0, 30), successCriteria: ["Independent reproduction before accepting claims", "Point-in-time evaluation after costs", "A later prospective comparison against the frozen baseline"] });
  for (const [key, query] of [["causal-world", "financial news event study causal impact"],
    ["human-behavior", "behavioral finance investor sentiment crowding"],
    ["self-evaluation", "continual learning concept drift evaluation"]]) missions.push({ key, hypothesisId: null,
      question: `Investigate ${query}; distinguish evidence, assumptions and counterexamples.`, searchQuery: query,
      priority: 0.5, evidenceRefs: [], successCriteria: ["Source provenance", "Reproduction and counter-evidence", "No untested production change"] });
  return [...new Map(missions.map(m => [m.key, m])).values()].sort((a, b) => b.priority - a.priority).slice(0, 25);
}
export function parseRepositoryLeads(payload: unknown): ResearchDocument[] {
  const value = payload as { items?: Array<Record<string, any>>; incomplete_results?: boolean };
  if (!Array.isArray(value?.items)) throw new Error("GITHUB_SEARCH_SCHEMA_INVALID");
  return value.items.slice(0, 3).flatMap(row => {
    const uri = publicDocumentUri(row.html_url);
    if (!uri || !/^https:\/\/github\.com\/[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(uri) || row.private === true) return [];
    return [{ provider: "github_public_search", uri, title: text(row.full_name, 200), excerpt: text(row.description),
      publishedAt: null, license: typeof row.license?.spdx_id === "string" ? row.license.spdx_id : null,
      metadata: { repository: row.full_name, stars: row.stargazers_count ?? null, archived: row.archived === true,
        pushed_at: row.pushed_at ?? null, search_incomplete: value.incomplete_results === true,
        stars_are_not_validation: true, code_executed: false } }];
  });
}
export function parseLiteratureLeads(payload: unknown): ResearchDocument[] {
  const value = payload as { message?: { items?: Array<Record<string, any>> } };
  if (!Array.isArray(value?.message?.items)) throw new Error("CROSSREF_SEARCH_SCHEMA_INVALID");
  return value.message.items.slice(0, 3).flatMap(row => {
    const uri = publicDocumentUri(row.URL), title = text(Array.isArray(row.title) ? row.title[0] : row.title, 400);
    if (!uri || !title) return [];
    const parts = row.published?.["date-parts"]?.[0];
    // Bibliographic dates are not evidence of when Brian knew this information.
    const date = Array.isArray(parts) && Number.isInteger(parts[0]) && parts[0] >= 1500 && parts[0] <= 2200
      ? `${parts[0]}-${String(parts[1] ?? 1).padStart(2, "0")}-${String(parts[2] ?? 1).padStart(2, "0")}T00:00:00Z` : null;
    return [{ provider: "crossref_public_metadata", uri, title, excerpt: text(row.abstract),
      publishedAt: date && Number.isFinite(Date.parse(date)) ? new Date(date).toISOString() : null,
      license: publicDocumentUri(row.license?.[0]?.URL), metadata: { doi: text(row.DOI, 200), type: row.type ?? null,
        publisher: text(row.publisher, 200), bibliographic_metadata_only: true, full_text_read: false, peer_review_verified: false } }];
  });
}
// This transport has no arbitrary redirect/document fetch surface. Research topics and repository
// names are open; credentials, private network addresses and downloaded code are never forwarded.
export async function boundedPublicRead(url: string, fetcher: typeof fetch = fetch): Promise<string> {
  const uri = new URL(url);
  if (uri.protocol !== "https:" || uri.username || uri.password || uri.port ||
      !["api.github.com", "api.crossref.org"].includes(uri.hostname)) throw new Error("RESEARCH_TRANSPORT_DESTINATION_REJECTED");
  const response = await fetcher(uri.toString(), { redirect: "error", headers: { accept: "application/json",
    "user-agent": "BrianResearch/1.0 (+https://github.com/om3r305/Om3r-Pro-Proje)" }, signal: AbortSignal.timeout(6500) });
  if (!response.ok) throw new Error(`RESEARCH_PROVIDER_HTTP_${response.status}`);
  const declared = Number(response.headers.get("content-length"));
  if (declared > 512_000) { await response.body?.cancel(); throw new Error("RESEARCH_RESPONSE_TOO_LARGE"); }
  const reader = response.body?.getReader();
  if (!reader) throw new Error("RESEARCH_RESPONSE_EMPTY");
  const chunks: Uint8Array[] = []; let bytes = 0;
  try {
    while (true) { const next = await reader.read(); if (next.done) break;
      bytes += next.value.byteLength; if (bytes > 512_000) throw new Error("RESEARCH_RESPONSE_TOO_LARGE"); chunks.push(next.value); }
  } finally { await reader.cancel().catch(() => {}); }
  const buffer = new Uint8Array(bytes); let offset = 0;
  for (const chunk of chunks) { buffer.set(chunk, offset); offset += chunk.length; }
  return new TextDecoder().decode(buffer);
}

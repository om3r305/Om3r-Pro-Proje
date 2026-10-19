// Read-only evidence dossiers. A heuristic score is never a calibrated probability.
export type MechanismInput = {
  mechanism_id: string; narrative_id?: string; observed_at?: string; confidence?: number;
  cause?: string; transmission?: unknown; affected_assets?: unknown; evidence_refs?: string[];
};
export type SourceInput = {
  event_id: string; source_id?: string; claim?: string; provenance_uri?: string;
  trust_class?: string; published_at?: string | null; first_observed_at?: string; created_at?: string;
};
export type ScenarioInput = { mechanism_id?: string; branch?: string; invalidators?: string[] };
export const DOSSIER_VERSION = "brian.world-dossier.v1";
export const MAX_DOSSIER_EVENTS = 400;
const PER_MECHANISM = 80;
const eventId = (ref: string) => /^intel_event:([a-f0-9]{64})$/.exec(ref)?.[1] ?? null;
const uniqueRefs = (m: MechanismInput) => [...new Set(Array.isArray(m.evidence_refs) ? m.evidence_refs : [])];
const stamp = (value: unknown) => typeof value === "string" && Number.isFinite(Date.parse(value)) ? Date.parse(value) : null;
const clean = (value: unknown, limit = 420) => typeof value === "string" ? value.replace(/\p{Cc}/gu, " ").slice(0, limit) : "";
export function safeSourceUri(value: unknown): string | null {
  try {
    const u = new URL(String(value));
    if (u.protocol !== "https:" || u.username || u.password || u.port) return null;
    if (!/^[a-z][a-z0-9.-]+\.[a-z]{2,}$/i.test(u.hostname) ||
      /(^|\.)(localhost|local|internal|test|invalid)$/i.test(u.hostname)) return null;
    u.hash = ""; u.search = "";
    return u.toString();
  } catch { return null; }
}
export function selectDossierEventIds(mechanisms: MechanismInput[]): string[] {
  // Round-robin prevents the first long narrative from consuming the full lookup budget.
  const groups = mechanisms.slice(0, 40).map(m => uniqueRefs(m).map(eventId).filter((id): id is string => !!id).slice(0, PER_MECHANISM));
  const ids = new Set<string>();
  for (let i = 0; i < PER_MECHANISM && ids.size < MAX_DOSSIER_EVENTS; i++) {
    for (const group of groups) {
      if (group[i]) ids.add(group[i]);
      if (ids.size === MAX_DOSSIER_EVENTS) break;
    }
  }
  return [...ids];
}
export function buildWorldDossiers(mechanisms: MechanismInput[], sources: SourceInput[],
  scenarios: ScenarioInput[], asOf: string, lookupState = "AVAILABLE") {
  const now = stamp(asOf);
  if (now === null) throw new Error("DOSSIER_AS_OF_INVALID");
  const byId = new Map(sources.map(source => [source.event_id, source]));
  const requested = new Set(selectDossierEventIds(mechanisms));
  return mechanisms.slice(0, 40).map(m => {
    const observed = stamp(m.observed_at);
    const validTime = observed !== null && observed <= now;
    const cutoff = validTime ? observed : null;
    const refs = uniqueRefs(m), ids = refs.map(eventId).filter((id): id is string => !!id);
    const available = ids.flatMap(id => {
      const row = byId.get(id);
      if (lookupState !== "AVAILABLE" || !requested.has(id) || !row || cutoff === null) return [];
      const captured = stamp(row.first_observed_at), persisted = stamp(row.created_at), published = stamp(row.published_at);
      if (captured === null || persisted === null || captured > cutoff || persisted > cutoff) return [];
      if (published !== null && published > cutoff) return [];
      return [row];
    });
    const counts = { official_primary_label: 0, professional_label: 0, discovery_label: 0, other_label: 0 };
    const hosts = new Set<string>();
    for (const row of available) {
      const uri = safeSourceUri(row.provenance_uri);
      if (uri) hosts.add(new URL(uri).hostname.replace(/^www\./, "").toLowerCase());
      if (row.trust_class === "OFFICIAL_PRIMARY") counts.official_primary_label++;
      else if (row.trust_class === "INDEPENDENT_PROFESSIONAL") counts.professional_label++;
      else if (row.trust_class === "UNVERIFIED_DISCOVERY") counts.discovery_label++;
      else counts.other_label++;
    }
    const uncovered = ids.length - available.length;
    const needs = [
      ...(!validTime ? ["MECHANISM_TIME_INVALID"] : []),
      ...(lookupState !== "AVAILABLE" ? ["SOURCE_LOOKUP_FAILED"] : []),
      ...(uncovered || refs.length !== ids.length ? ["SOURCE_REFERENCE_COVERAGE_INCOMPLETE"] : []),
      ...(!counts.official_primary_label ? ["PRIMARY_SOURCE_UNAVAILABLE"] : []),
      "SOURCE_CLAIMS_NOT_INDEPENDENTLY_VERIFIED", "COUNTER_EVIDENCE_NOT_ASSESSED",
      "PRICED_IN_NOT_ASSESSED", "ACTIONABLE_HORIZON_UNRESOLVED", "PREDICTIVE_CALIBRATION_UNVERIFIED",
    ];
    return {
      dossier_id: "dossier:" + m.mechanism_id, mechanism_id: m.mechanism_id,
      narrative_id: clean(m.narrative_id, 160), observed_at: m.observed_at ?? null, assessed_at: asOf,
      stage: "RESEARCH_INCOMPLETE",
      mechanism: { cause: clean(m.cause, 800), transmission: Array.isArray(m.transmission) ? m.transmission.map(x => clean(x, 300)).slice(0, 12) : [],
        affected_assets: Array.isArray(m.affected_assets) ? m.affected_assets.slice(0, 20) : [] },
      score: { value: typeof m.confidence === "number" && Number.isFinite(m.confidence) && m.confidence >= 0 && m.confidence <= 1 ? m.confidence : null,
        kind: "HEURISTIC_UNCALIBRATED", calibrated_probability: null },
      source_summary: { ...counts, publisher_host_groups: hosts.size,
        independently_verified_source_groups: null, source_labels_are_verification: false },
      coverage: { references: refs.length, valid_event_references: ids.length,
        requested: ids.filter(id => requested.has(id)).length, available: available.length,
        unresolved: uncovered + refs.length - ids.length,
        complete: validTime && lookupState === "AVAILABLE" && refs.length > 0 && refs.length === available.length,
        lookup_state: lookupState, maximum_query_events: MAX_DOSSIER_EVENTS },
      source_claims: available.slice(0, 8).map(row => ({ event_id: row.event_id,
        source_id: clean(row.source_id, 180), claim: clean(row.claim), uri: safeSourceUri(row.provenance_uri),
        published_at: row.published_at ?? null, captured_at: row.first_observed_at ?? null,
        persisted_at: row.created_at ?? null, source_trust_class: clean(row.trust_class, 80),
        role: "ASSOCIATED_NARRATIVE_CLAIM", claim_verification: "NOT_INDEPENDENTLY_VERIFIED",
        full_text_read: false })),
      counter_evidence: { status: "NOT_ASSESSED", receipts: [],
        proposed_invalidators: scenarios.filter(s => s.mechanism_id === m.mechanism_id && s.branch === "MECHANISM_HOLDS")
          .flatMap(s => Array.isArray(s.invalidators) ? s.invalidators : []).slice(0, 8).map(x => clean(x, 300)),
        proposed_invalidators_are_evidence: false },
      priced_in: { status: "NOT_ASSESSED" }, actionable_horizon: { status: "UNRESOLVED", seconds: null },
      missing_requirements: needs, trade_ready: false, direct_alpha_influence: false,
      shadow_only: true, live_execution: false, version: DOSSIER_VERSION,
    };
  });
}


import type { SupabaseClient } from "npm:@supabase/supabase-js@2.116.0";
import { OPEN_WORLD_VERSION, boundedPublicRead, learningObservations, parseLiteratureLeads,
  parseRepositoryLeads, publicDocumentUri, researchMissions, type ResearchDocument, type ResearchHypothesis } from "./open_world_research.ts";

async function hash(value: string) {
  return [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value)))].map(x => x.toString(16).padStart(2, "0")).join("");
}
function checked(result: { error: { message: string } | null }, step: string) {
  if (result.error) throw new Error(`${step}:${result.error.message}`);
}
export async function runOpenWorldCycle(db: SupabaseClient, hypotheses: ResearchHypothesis[], observedAt: string) {
  const controls = await db.from("brian_evolution_runtime_config").select("config_key,config_value")
    .in("config_key", ["brian_system_enabled", "brian_open_world_research_enabled"]);
  checked(controls, "research_controls");
  const flags = new Map((controls.data ?? []).map(x => [x.config_key, x.config_value]));
  if (flags.get("brian_system_enabled") !== "true" || flags.get("brian_open_world_research_enabled") !== "true") {
    return { status: "DISABLED", documents: 0, observations: 0, contexts: [] as any[], learningHypotheses: [] as any[] };
  }
  const summary = await db.rpc("brian_learning_window_v1", { p_now: observedAt });
  checked(summary, "learning_window");
  const observations = learningObservations(summary.data ?? {});
  const learningHypotheses = observations.map(row => ({
    hypothesisId: `hypothesis|open-world-v1|${row.kind.toLowerCase()}`, observedAt,
    problemStatement: row.observation, proposedMechanism: row.nextExperiment,
    targetCapabilities: row.kind === "MEASUREMENT_COVERAGE" ? ["research.measurement", "market.reference-path"] : ["alpha.expected-edge", "research.calibration"],
    evidenceRefs: row.evidenceRefs, counterEvidenceRefs: [], measurableSuccessCriteria: [
      "Register baseline and challenger before evaluation", "Compare later data without changing the frozen success criteria",
      "Report missing coverage and modeled transaction costs", "No live execution or canonical mutation"],
    stage: "RESEARCHING" as const, uncertainty: 0.5, priority: 0.99,
    hypothesisKind: row.kind === "MEASUREMENT_COVERAGE" ? "CAPABILITY_GAP" as const : "EXPECTED_EDGE" as const,
    metadata: { learning_version: OPEN_WORLD_VERSION, lesson_kind: row.kind, causal_explanation_proven: false,
      observation_metrics: row.metrics, evidence_origin: "INTERNAL_OUTCOME_MEASUREMENT" },
  }));
  for (const row of observations) {
    const episodeId = await hash(`${OPEN_WORLD_VERSION}|${row.kind}|${observedAt}`);
    checked(await db.from("brian_learning_episodes").upsert({ episode_id: episodeId, observed_at: observedAt,
      window_start: summary.data.window_start, window_end: observedAt, kind: row.kind, observation: row.observation,
      metrics: row.metrics, evidence_refs: row.evidenceRefs, next_experiment: row.nextExperiment,
      evidence_class: "PROSPECTIVE_EVOLUTION_SHADOW", shadow_only: true, live_execution: false,
    }, { onConflict: "episode_id", ignoreDuplicates: true }), "persist_learning");
  }
  const missions = researchMissions([...learningHypotheses, ...hypotheses], []);
  const missionRows = await Promise.all(missions.map(async m => ({ mission_id: await hash(`${OPEN_WORLD_VERSION}|${m.key}|${m.searchQuery}`),
    hypothesis_id: m.hypothesisId, question: m.question, search_query: m.searchQuery, priority: m.priority,
    evidence_refs: m.evidenceRefs, success_criteria: m.successCriteria, created_at: observedAt,
    next_research_at: observedAt, state: "QUEUED", version: OPEN_WORLD_VERSION })));
  if (missionRows.length) checked(await db.from("brian_research_missions").upsert(missionRows,
    { onConflict: "mission_id", ignoreDuplicates: true }), "persist_missions");
  const due = await db.from("brian_research_missions").select("*").lte("next_research_at", observedAt)
    .order("next_research_at", { ascending: true }).order("priority", { ascending: false }).limit(1).maybeSingle();
  checked(due, "research_due");
  let stored = 0; const failures: string[] = []; let missionId: string | null = null;
  if (due.data) {
    const mission = due.data; missionId = mission.mission_id;
    const query = String(mission.search_query).slice(0, 180);
    const githubQuery = query.includes("transaction cost") ? "transaction cost backtest" : query.split(" ").slice(0, 4).join(" ");
    const requests = [
      { name: "github", url: `https://api.github.com/search/repositories?${new URLSearchParams({ q: githubQuery, per_page: "3" })}`, parse: parseRepositoryLeads },
      { name: "crossref", url: `https://api.crossref.org/works?${new URLSearchParams({ query, rows: "3", select: "DOI,title,URL,abstract,published,publisher,type,license" })}`, parse: parseLiteratureLeads },
    ];
    const collected = await Promise.allSettled(requests.map(async source => {
      const body = await boundedPublicRead(source.url); const responseHash = await hash(body);
      return source.parse(JSON.parse(body)).map(doc => ({ ...doc,
        metadata: { ...doc.metadata, provider_query_uri: source.url, provider_response_sha256: responseHash } }));
    }));
    const documents: ResearchDocument[] = [];
    for (let i = 0; i < collected.length; i++) {
      const result = collected[i];
      if (result.status === "fulfilled") documents.push(...result.value);
      else failures.push(`${requests[i].name}:${String(result.reason).slice(0, 200)}`);
    }
    // Read repository documentation as inert text; no install, import, subprocess or code execution.
    await Promise.all(documents.filter(d => d.provider === "github_public_search").slice(0, 2).map(async doc => {
      const repository = new URL(doc.uri).pathname.slice(1);
      try {
        const body = await boundedPublicRead(`https://api.github.com/repos/${repository}/readme`);
        const readme = JSON.parse(body);
        if (readme.encoding !== "base64" || typeof readme.content !== "string") throw new Error("README_SCHEMA_INVALID");
        const decoded = new TextDecoder().decode(Uint8Array.from(atob(readme.content.replace(/\s/g, "")), x => x.charCodeAt(0)));
        doc.metadata = { ...doc.metadata, readme_blob_sha: readme.sha ?? null,
          readme_uri: publicDocumentUri(readme.html_url), readme_content_sha256: await hash(decoded),
          documentation_received: true, documentation_excerpt_chars: Math.min(decoded.length, 1000), code_executed: false };
        doc.excerpt = decoded.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim().slice(0, 1000);
      } catch (error) { doc.metadata.documentation_error = String(error).slice(0, 200); }
    }));
    for (const doc of documents) {
      const contentHash = await hash(JSON.stringify({ title: doc.title, excerpt: doc.excerpt, metadata: doc.metadata }));
      const documentId = await hash(`${missionId}|${doc.provider}|${doc.uri}|${contentHash}`);
      checked(await db.from("brian_research_documents").upsert({ document_id: documentId, mission_id: missionId,
        provider: doc.provider, canonical_uri: doc.uri, title: doc.title, excerpt: doc.excerpt,
        published_at: doc.publishedAt, captured_at: new Date().toISOString(), content_sha256: contentHash,
        license: doc.license, metadata: doc.metadata, trust_class: "UNVERIFIED_EXTERNAL",
        evidence_role: "RESEARCH_LEAD_NOT_TRADE_SIGNAL", code_executed: false,
        shadow_only: true, live_execution: false,
      }, { onConflict: "document_id", ignoreDuplicates: true }), "persist_document");
      stored++;
    }
    checked(await db.from("brian_research_missions").update({ last_researched_at: new Date().toISOString(),
      next_research_at: new Date(Date.parse(observedAt) + (stored ? 6 : 1) * 3600_000).toISOString(),
      state: stored ? failures.length ? "PARTIAL" : "EVIDENCE_COLLECTED" : "NO_EVIDENCE",
      last_error: failures.length ? failures.join(" | ") : stored ? null : "No matching public records; no conclusion drawn",
      attempts: Number(mission.attempts ?? 0) + 1,
    }).eq("mission_id", missionId), "complete_mission");
  }
  // Context may be captured after observedAt; it is for the later engineering run, never backdated into a decision.
  const contextRows = await db.from("brian_research_documents").select("document_id,mission_id,canonical_uri,title,excerpt,captured_at,trust_class,evidence_role")
    .gte("captured_at", new Date(Date.parse(observedAt) - 48 * 3600_000).toISOString())
    .order("captured_at", { ascending: false }).limit(60);
  checked(contextRows, "research_context");
  const allowedMissions = new Map(missionRows.map(m => [m.mission_id, m.hypothesis_id]));
  const contexts = (contextRows.data ?? []).filter(doc => allowedMissions.get(doc.mission_id)).map(doc => ({
    ...doc, hypothesis_id: allowedMissions.get(doc.mission_id), untrusted_data: true, learned_claim_verified: false,
  }));
  const status = failures.length ? stored ? "DEGRADED" : "FAILED" : due.data && !stored ? "NO_EVIDENCE" : "SUCCESS";
  checked(await db.from("brian_collector_runs").insert({ run_id: await hash(`${OPEN_WORLD_VERSION}|${observedAt}`),
    collector_id: "brian-open-world-research-v1", started_at: observedAt, finished_at: new Date().toISOString(),
    status: status === "NO_EVIDENCE" ? "DEGRADED" : status, observed_records: missionRows.length,
    stored_records: stored + observations.length, degraded_sources: failures,
    evidence_class: "PROSPECTIVE_EVOLUTION_SHADOW", shadow_only: true, live_execution: false,
    metadata: { version: OPEN_WORLD_VERSION, mission_id: missionId, research_status: status,
      observations: observations.length, retrieved_documents: stored, external_claims_verified: 0,
      observation_policy: "PUBLIC_READ_ONLY_UNTRUSTED_DATA", model_training_performed: false,
      code_executed: false, canonical_mutation: false },
  }), "research_receipt");
  return { status, missionId, documents: stored, observations: observations.length, failures, contexts, learningHypotheses };
}

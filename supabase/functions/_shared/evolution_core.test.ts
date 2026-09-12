import {
  assessWorldSource,
  buildEvolutionDashboardModel,
  deriveCapabilitySnapshots,
  detectCapabilityGaps,
  discoverWorldSources,
  inferAuthorityClass,
  type CollectorRunLike,
} from "./evolution_core.ts";

Deno.test("capability graph marks fresh successful collectors healthy", () => {
  const at = "2026-09-11T12:00:00.000Z";
  const runs: CollectorRunLike[] = [
    { collector_id: "brian-universe-collector", finished_at: "2026-09-11T11:59:30.000Z", status: "SUCCESS" },
    { collector_id: "brian-sensor-mesh", finished_at: "2026-09-11T11:59:30.000Z", status: "SUCCESS" },
    { collector_id: "brian-intrabar-eye", finished_at: "2026-09-11T11:59:30.000Z", status: "SUCCESS" },
    { collector_id: "phase39-binance-usdm-derivatives", finished_at: "2026-09-11T11:59:00.000Z", status: "SUCCESS" },
    { collector_id: "phase39-gdelt-news", finished_at: "2026-09-11T11:55:00.000Z", status: "SUCCESS" },
    { collector_id: "brian-official-macro-eye", finished_at: "2026-09-11T11:30:00.000Z", status: "SUCCESS" },
    { collector_id: "phase39-ecb-fx", finished_at: "2026-09-11T08:00:00.000Z", status: "SUCCESS" },
    { collector_id: "brian-alpha-decision-compiler-v2", finished_at: "2026-09-11T11:59:40.000Z", status: "SUCCESS" },
    { collector_id: "brian-missed-opportunity-auditor-v2", finished_at: "2026-09-11T11:58:00.000Z", status: "SUCCESS" },
    { collector_id: "brian-alpha-calibration-challenger-v1", finished_at: "2026-09-11T11:00:00.000Z", status: "SUCCESS" },
    { collector_id: "brian-evolution-orchestrator-v1", finished_at: "2026-09-11T11:59:00.000Z", status: "SUCCESS" },
  ];
  const snapshots = deriveCapabilitySnapshots(runs, at);
  const alpha = snapshots.find((row) => row.capabilityId === "alpha.compiler");
  const evolution = snapshots.find((row) => row.capabilityId === "evolution.capability-graph");
  if (alpha?.health !== "HEALTHY") throw new Error(`expected healthy alpha, got ${alpha?.health}`);
  if (alpha.stage !== "ACTIVE") throw new Error(`expected ACTIVE alpha, got ${alpha.stage}`);
  if (evolution?.stage !== "EXPERIMENTAL") throw new Error(`expected experimental evolution, got ${evolution?.stage}`);
  if (!alpha.shadowOnly || alpha.liveExecution) throw new Error("capability snapshot execution boundary broken");
});

Deno.test("capability graph detects stale and missing runtime eyes", () => {
  const at = "2026-09-11T12:00:00.000Z";
  const snapshots = deriveCapabilitySnapshots([
    { collector_id: "brian-alpha-decision-compiler-v2", finished_at: "2026-09-11T10:00:00.000Z", status: "SUCCESS" },
  ], at);
  const alpha = snapshots.find((row) => row.capabilityId === "alpha.compiler");
  const news = snapshots.find((row) => row.capabilityId === "world.news-discovery");
  if (alpha?.health !== "STALE") throw new Error(`expected stale alpha, got ${alpha?.health}`);
  if (news?.health !== "MISSING") throw new Error(`expected missing news, got ${news?.health}`);
  const gaps = detectCapabilityGaps(snapshots);
  if (!gaps.some((row) => row.capabilityId === "world.news-discovery" && row.severity === "CRITICAL")) {
    throw new Error("missing runtime capability did not create critical gap");
  }
  if (!gaps.some((row) => row.capabilityId === "portfolio.treasury")) {
    throw new Error("future treasury capability gap missing");
  }
});

Deno.test("world explorer de-duplicates discovered domains prospectively", () => {
  const rows = discoverWorldSources([
    { event_id: "a", provenance_uri: "https://www.sec.gov/news/test", source_id: "sec", asset: "NVDA" },
    { event_id: "b", provenance_uri: "https://sec.gov/other", source_id: "sec", asset: "BTC" },
    { event_id: "c", provenance_uri: "https://reddit.com/r/test", source_id: "reddit", asset: "BTC" },
  ], "2026-09-11T12:00:00.000Z");
  if (rows.length !== 2) throw new Error(`expected 2 domain candidates, got ${rows.length}`);
  const sec = rows.find((row) => row.sourceId === "world:sec.gov");
  const reddit = rows.find((row) => row.sourceId === "world:reddit.com");
  if (sec?.authorityClass !== "OFFICIAL_PRIMARY") throw new Error("SEC not classified official");
  if (reddit?.authorityClass !== "COMMUNITY") throw new Error("Reddit not classified community");
  if (reddit.manipulationRisk <= sec.manipulationRisk) throw new Error("community manipulation risk should be higher");
});

Deno.test("source trust fails closed for directional use before ACTIVE", () => {
  const [source] = discoverWorldSources([
    { event_id: "a", provenance_uri: "https://www.reuters.com/world/test", source_id: "Reuters", asset: "GOLD" },
  ], "2026-09-11T12:00:00.000Z");
  const assessment = assessWorldSource(source, "2026-09-11T12:00:10.000Z");
  if (!assessment.eligibleForResearch) throw new Error("professional source should be research eligible");
  if (assessment.eligibleForDecisionEvidence) throw new Error("DISCOVERED source must not influence ALPHA");
});

Deno.test("authority classifier prefers primary institutions", () => {
  if (inferAuthorityClass("ECB", "https://www.ecb.europa.eu/press") !== "OFFICIAL_PRIMARY") {
    throw new Error("ECB should be official primary");
  }
  if (inferAuthorityClass("community", "https://x.com/example") !== "COMMUNITY") {
    throw new Error("X should be community");
  }
});

Deno.test("dashboard model remains shadow-only and surfaces gaps", () => {
  const at = "2026-09-11T12:00:00.000Z";
  const snapshots = deriveCapabilitySnapshots([], at);
  const gaps = detectCapabilityGaps(snapshots);
  const sources = discoverWorldSources([
    { event_id: "a", provenance_uri: "https://www.sec.gov/news/test", source_id: "SEC", asset: "NVDA" },
  ], at);
  const model = buildEvolutionDashboardModel(snapshots, gaps, sources, at, at);
  if (!model.shadowOnly || model.liveExecution) throw new Error("dashboard execution boundary broken");
  if (!model.cloudIndependent) throw new Error("cloud independence flag lost");
  if (!model.criticalGaps.length) throw new Error("expected critical gaps for empty runtime evidence");
  if (model.sourceCounts.total !== 1) throw new Error("source count mismatch");
});

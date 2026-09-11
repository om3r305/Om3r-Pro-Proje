import { buildWorldBrainBatch } from "./world_brain.ts";

Deno.test("World Brain maps cross-asset entities and narratives without direct ALPHA influence", () => {
  const at = "2026-09-11T12:00:00.000Z";
  const batch = buildWorldBrainBatch([
    {
      event_id: "evt-ai-1",
      asset: "NVDA",
      event_kind: "NEWS_HEADLINE",
      source_kind: "OFFICIAL_RELEASE",
      source_id: "nvidia.com",
      first_observed_at: "2026-09-11T11:58:00.000Z",
      claim: "NVIDIA announces new AI GPU platform for data center inference with TSMC advanced packaging",
      direction: 1,
      trust_class: "PRIMARY",
      entity_confidence: 0.92,
      provenance_uri: "https://www.nvidia.com/example",
    },
  ], at);

  const ids = new Set(batch.entityObservations.map((row) => row.entityId));
  if (!ids.has("company:NVIDIA")) throw new Error("NVIDIA entity missing");
  if (!ids.has("company:TSMC")) throw new Error("TSMC entity missing");
  if (!ids.has("technology:AI")) throw new Error("AI entity missing");
  if (!batch.narrativeSnapshots.some((row) => row.narrativeId === "narrative:AI_COMPUTE")) {
    throw new Error("AI compute narrative missing");
  }
  if (!batch.causalMechanisms.some((row) => row.narrativeId === "narrative:AI_COMPUTE")) {
    throw new Error("AI causal mechanism missing");
  }
  if (!batch.assetImpacts.some((row) => row.assetId === "company:TSMC")) {
    throw new Error("second-order TSMC impact candidate missing");
  }
  if (batch.assetImpacts.some((row) => row.directAlphaInfluence)) throw new Error("Layer 2 must not directly influence ALPHA");
  if (!batch.shadowOnly || batch.liveExecution) throw new Error("World Brain execution boundary broken");
});

Deno.test("World Brain represents monetary policy as a conditional mechanism, not certain forecast", () => {
  const at = "2026-09-11T12:00:00.000Z";
  const batch = buildWorldBrainBatch([
    {
      event_id: "evt-fed-1",
      asset: "MACRO",
      event_kind: "OFFICIAL_MACRO",
      source_id: "federalreserve.gov",
      first_observed_at: "2026-09-11T11:50:00.000Z",
      claim: "Federal Reserve policy statement discusses interest rate outlook and inflation",
      direction: 0,
      trust_class: "OFFICIAL_PRIMARY",
      provenance_uri: "https://www.federalreserve.gov/example",
    },
  ], at);
  const mechanism = batch.causalMechanisms.find((row) => row.narrativeId === "narrative:MONETARY_POLICY");
  if (!mechanism) throw new Error("monetary-policy mechanism missing");
  if (!mechanism.counterEvidenceRequired) throw new Error("counter-evidence requirement lost");
  const gold = mechanism.affectedAssets.find((row) => row.assetId === "commodity:GOLD");
  if (!gold) throw new Error("gold transmission missing");
  const breakScenario = batch.scenarios.find((row) => row.mechanismId === mechanism.mechanismId && row.branch === "MECHANISM_BREAKS");
  if (!breakScenario) throw new Error("mechanism-break scenario missing");
  if (breakScenario.assetImpacts.some((row) => row.conditionalDirection !== 0)) throw new Error("broken-mechanism branch retained directional assertion");
});

Deno.test("World Brain only creates future calendar items from explicit future timestamps", () => {
  const at = "2026-09-11T12:00:00.000Z";
  const batch = buildWorldBrainBatch([
    {
      event_id: "evt-apple-future",
      asset: "AAPL",
      event_kind: "PRODUCT_EVENT",
      source_id: "apple.com",
      first_observed_at: at,
      claim: "Apple product event announced",
      provenance_uri: "https://www.apple.com/example",
      metadata: { scheduled_at: "2026-10-01T17:00:00Z" },
    },
    {
      event_id: "evt-no-date",
      asset: "BTCUSDT",
      event_kind: "NEWS_HEADLINE",
      source_id: "example.com",
      first_observed_at: at,
      claim: "Bitcoin market update with no scheduled event",
    },
  ], at);
  if (batch.futureEvents.length !== 1) throw new Error(`expected 1 explicit future event, got ${batch.futureEvents.length}`);
  if (batch.futureEvents[0].scheduledAt !== "2026-10-01T17:00:00.000Z") throw new Error("future event timestamp changed");
  if (batch.futureEvents[0].stage !== "VERIFYING") throw new Error("future event should require verification");
});

Deno.test("World Brain relation parser keeps mere co-mentions low confidence", () => {
  const at = "2026-09-11T12:00:00.000Z";
  const batch = buildWorldBrainBatch([
    {
      event_id: "evt-comention",
      asset: "NVDA",
      first_observed_at: at,
      claim: "NVIDIA and Apple were discussed at the same technology conference",
      source_id: "example.com",
    },
  ], at);
  const relation = batch.relationAssertions.find((row) => row.srcEntityId !== row.dstEntityId);
  if (!relation) throw new Error("expected relation assertion");
  if (relation.relation !== "CO_MENTIONED") throw new Error(`unexpected relation ${relation.relation}`);
  if (relation.confidence >= 0.5) throw new Error("co-mention confidence too high");
  if (!relation.mechanism.includes("no causal relationship")) throw new Error("co-mention caution missing");
});

Deno.test("World Brain energy/geopolitics can create multiple conditional scenario paths", () => {
  const at = "2026-09-11T12:00:00.000Z";
  const batch = buildWorldBrainBatch([
    {
      event_id: "evt-oil-risk",
      asset: "OIL",
      first_observed_at: at,
      claim: "Geopolitical conflict raises oil supply and pipeline disruption concerns",
      direction: 1,
      source_id: "reuters.com",
      provenance_uri: "https://www.reuters.com/example",
    },
  ], at);
  const narratives = new Set(batch.narrativeSnapshots.map((row) => row.narrativeId));
  if (!narratives.has("narrative:GEOPOLITICS")) throw new Error("geopolitics narrative missing");
  if (!narratives.has("narrative:ENERGY_SUPPLY")) throw new Error("energy narrative missing");
  if (batch.scenarios.length < 4) throw new Error("expected holds/breaks branches for multiple mechanisms");
  if (!batch.assetImpacts.some((row) => row.assetId === "commodity:OIL")) throw new Error("oil impact candidate missing");
});

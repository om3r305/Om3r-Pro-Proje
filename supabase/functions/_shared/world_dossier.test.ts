import { buildWorldDossiers, selectDossierEventIds } from "./world_dossier.ts";

Deno.test("world dossier score remains uncalibrated even with an official reference", () => {
  const id = "a".repeat(64), now = "2026-10-19T09:00:00Z";
  const [d] = buildWorldDossiers([{ mechanism_id: "m", observed_at: now, confidence: 0.97, evidence_refs: [`intel_event:${id}`] }],
    [{ event_id: id, trust_class: "OFFICIAL_PRIMARY", provenance_uri: "https://example.org/news", first_observed_at: now, created_at: now }], [], now);
  if (d.score.calibrated_probability !== null || d.trade_ready || d.live_execution || d.coverage.available !== 1) throw Error("dossier semantics changed");
});

Deno.test("world dossier lookup failure and future evidence cannot turn coverage complete", () => {
  const id = "b".repeat(64), now = "2026-10-19T09:00:00Z";
  const mechanisms = [{ mechanism_id: "m", observed_at: now, evidence_refs: [`intel_event:${id}`] }];
  const sources = [{ event_id: id, first_observed_at: now, created_at: "2026-10-19T10:00:00Z" }];
  for (const state of ["AVAILABLE", "QUERY_FAILED"]) {
    const [d] = buildWorldDossiers(mechanisms, sources, [], now, state);
    if (d.coverage.complete || d.coverage.available !== 0) throw Error("future/failed lookup accepted");
  }
});

Deno.test("world dossier event lookup enforces a fixed upper bound", () => {
  const mechanisms = Array.from({ length: 40 }, (_, i) => ({ mechanism_id: String(i),
    evidence_refs: Array.from({ length: 100 }, (_, n) => `intel_event:${(1 + i * 100 + n).toString(16).padStart(64, "0")}`) }));
  if (selectDossierEventIds(mechanisms).length !== 400) throw Error("lookup bound changed");
});

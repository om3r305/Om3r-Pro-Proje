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
  const allowed = origin && (ALLOWED_EXACT.has(origin) || ALLOWED_ORIGIN.test(origin))
    ? origin
    : "https://monster-coins-pro-oemer-yildirim.vercel.app";
  return {
    "access-control-allow-origin": allowed,
    "access-control-allow-headers": "content-type,x-brian-dashboard-key",
    "access-control-allow-methods": "POST,OPTIONS",
    vary: "Origin",
  };
}

function out(body: unknown, status = 200, origin?: string | null) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
      ...cors(origin),
    },
  });
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
  if (!constantTimeEqual(await sha256Hex(supplied), String(q.data.dashboard_key_sha256 ?? ""))) {
    throw new Error("UNAUTHORIZED_DASHBOARD");
  }
}

const clamp = (value: number, lo = 0, hi = 100) => Math.max(lo, Math.min(hi, value));
const round1 = (value: number) => Math.round(value * 10) / 10;
const ratioPct = (numerator: number, denominator: number) => denominator > 0 ? (100 * numerator / denominator) : 0;
const evidenceCurve = (samples: number, target: number) => target > 0
  ? clamp(100 * Math.log1p(Math.max(0, samples)) / Math.log1p(target))
  : 0;

async function exactCount(table: string, configure?: (q: any) => any): Promise<number> {
  let q: any = db.from(table).select("*", { count: "exact", head: true });
  if (configure) q = configure(q);
  const result = await q;
  if (result.error) throw new Error(`${table}: ${result.error.message}`);
  return Number(result.count ?? 0);
}

async function safeRows(label: string, promise: Promise<any>, fallback: any[] = []): Promise<any[]> {
  try {
    const result = await promise;
    if (result.error) throw new Error(result.error.message);
    return Array.isArray(result.data) ? result.data : fallback;
  } catch (error) {
    console.error(label, error);
    return fallback;
  }
}

function weightedRunHealth(rows: any[]) {
  if (!rows.length) return { score: 0, success: 0, degraded: 0, failed: 0, total: 0 };
  let success = 0, degraded = 0, failed = 0;
  for (const row of rows) {
    const status = String(row.status ?? "").toUpperCase();
    if (status === "SUCCESS") success += 1;
    else if (status === "DEGRADED" || status.startsWith("SKIPPED")) degraded += 1;
    else failed += 1;
  }
  const score = 100 * (success + degraded * 0.5) / rows.length;
  return { score: round1(score), success, degraded, failed, total: rows.length };
}

function maturityTier(score: number) {
  if (score < 10) return "ÇEKİRDEK";
  if (score < 25) return "ÇIRAK";
  if (score < 45) return "GELİŞEN";
  if (score < 65) return "YETKİN";
  if (score < 80) return "UZMAN";
  return "OLGUN";
}

function component(
  id: string,
  anatomy: string,
  name: string,
  quality: number | null,
  evidence: number,
  samples: number,
  rationale: string,
  metrics: Record<string, unknown>,
) {
  const evidencePct = clamp(evidence);
  const maturity = quality === null ? 0 : clamp(quality) * evidencePct / 100;
  return {
    id,
    anatomy,
    name,
    quality_pct: quality === null ? null : round1(clamp(quality)),
    evidence_pct: round1(evidencePct),
    maturity_pct: round1(maturity),
    samples,
    evidence_state: evidencePct < 10 ? "UNPROVEN" : evidencePct < 35 ? "THIN" : evidencePct < 70 ? "GROWING" : "MATURE_SAMPLE",
    rationale,
    metrics,
  };
}

Deno.serve(async (req: Request) => {
  const origin = req.headers.get("origin");
  if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: cors(origin) });
  if (req.method !== "POST") return out({ error: "POST required" }, 405, origin);
  try { await auth(req); } catch (error) { return out({ error: String(error) }, 401, origin); }

  try {
    const now = Date.now();
    const dayAgo = new Date(now - 24 * 60 * 60 * 1000).toISOString();
    const weekAgo = new Date(now - 7 * 24 * 60 * 60 * 1000).toISOString();

    const [
      alphaRows,
      collectorRows,
      worldAssessments,
      worldRuns,
      experimentResults,
      promotions,
      artifacts,
      gaps,
      treasuryRows,
      treasuryActions,
      sensorTotal,
      sensorHits,
      sourceCount,
      jobCount,
      codegenCount,
    ] = await Promise.all([
      safeRows("alpha-outcomes", db.from("brian_alpha_decision_outcomes")
        .select("direction_adjusted_return,horizon_seconds,resolved_at,brian_alpha_decisions!inner(action,estimated_round_trip_cost_bps)")
        .gte("resolved_at", weekAgo)
        .in("brian_alpha_decisions.action", ["OPEN_LONG", "OPEN_SHORT"])
        .order("resolved_at", { ascending: false })
        .limit(1000)),
      safeRows("collector-runs", db.from("brian_collector_runs")
        .select("collector_id,status,started_at,finished_at,observed_records,stored_records,degraded_sources,error_class")
        .gte("started_at", dayAgo)
        .not("collector_id", "ilike", "%dip%")
        .order("started_at", { ascending: false })
        .limit(1000)),
      safeRows("world-assessments", db.from("brian_world_source_assessments")
        .select("source_id,assessed_at,trust_score,eligible_for_research,eligible_for_decision_evidence")
        .order("assessed_at", { ascending: false })
        .limit(500)),
      safeRows("world-runs", db.from("brian_world_brain_runs")
        .select("status,started_at,finished_at,input_events,event_frames,entity_observations,narrative_snapshots,asset_impacts")
        .gte("started_at", weekAgo)
        .order("started_at", { ascending: false })
        .limit(500)),
      safeRows("experiment-results", db.from("brian_evolution_experiment_results")
        .select("result_id,experiment_id,measured_at,role,samples,regimes,net_edge_bps,favorable_after_cost_rate,leakage_detected,data_quality_ok,stability_score,complexity_delta")
        .order("measured_at", { ascending: false })
        .limit(200)),
      safeRows("promotions", db.from("brian_evolution_promotion_decisions")
        .select("decision_id,experiment_id,decided_at,decision,score,reasons,required_next_stage")
        .order("decided_at", { ascending: false })
        .limit(30)),
      safeRows("artifacts", db.from("brian_evolution_code_artifact_receipts")
        .select("receipt_id,candidate_id,hypothesis_id,evidence_kind,observed_at,passed,protected_scope_clear,leakage_detected,changed_paths,generated_by")
        .order("observed_at", { ascending: false })
        .limit(100)),
      safeRows("gaps", db.from("brian_evolution_gap_snapshots")
        .select("gap_id,observed_at,capability_id,domain,severity,reason,suggested_action,evidence_refs")
        .order("observed_at", { ascending: false })
        .limit(40)),
      safeRows("treasury", db.from("brian_treasury_shadow_snapshots")
        .select("snapshot_id,observed_at,starting_equity_usd,cash_usd,equity_usd,realized_pnl_usd,cumulative_costs_usd,deployment_usd,deployment_pct,cash_reserve_pct,positions,promotion_gate_open,blocked_reasons")
        .order("observed_at", { ascending: false }).limit(1)),
      safeRows("treasury-actions", db.from("brian_treasury_shadow_actions")
        .select("action_id,observed_at,kind,asset_id,direction,capital_usd,cost_usd,expected_net_edge_bps,reason")
        .order("observed_at", { ascending: false }).limit(200)),
      exactCount("brian_sensor_reliability_prospective_calibration", (q) => q.not("realized_hit", "is", null)),
      exactCount("brian_sensor_reliability_prospective_calibration", (q) => q.eq("realized_hit", true)),
      exactCount("brian_world_source_candidates"),
      exactCount("brian_system_job_registry"),
      exactCount("brian_evolution_codegen_requests"),
    ]);

    // 1) Brain / ALPHA: quality is actual directional success after decision-time estimated cost.
    let alphaGrossWins = 0;
    let alphaAfterCostWins = 0;
    let alphaCostSamples = 0;
    const horizon = new Map<number, { n: number; gross: number; after: number; costN: number }>();
    for (const row of alphaRows) {
      const adjustedBps = Number(row.direction_adjusted_return ?? 0) * 10000;
      const embedded = Array.isArray(row.brian_alpha_decisions)
        ? row.brian_alpha_decisions[0]
        : row.brian_alpha_decisions;
      const cost = Number(embedded?.estimated_round_trip_cost_bps);
      if (adjustedBps > 0) alphaGrossWins += 1;
      const h = Number(row.horizon_seconds ?? 0);
      const bucket = horizon.get(h) ?? { n: 0, gross: 0, after: 0, costN: 0 };
      bucket.n += 1;
      if (adjustedBps > 0) bucket.gross += 1;
      if (Number.isFinite(cost) && cost >= 0) {
        alphaCostSamples += 1;
        bucket.costN += 1;
        if (adjustedBps > cost) {
          alphaAfterCostWins += 1;
          bucket.after += 1;
        }
      }
      horizon.set(h, bucket);
    }
    const alphaGrossRate = ratioPct(alphaGrossWins, alphaRows.length);
    const alphaAfterCostRate = ratioPct(alphaAfterCostWins, alphaCostSamples);
    const alphaQuality = alphaCostSamples >= 50 ? alphaAfterCostRate : alphaGrossRate * 0.7;
    const alphaEvidence = evidenceCurve(alphaRows.length, 3000);
    const alpha = component(
      "alpha",
      "Beyin",
      "ALPHA Karar Kalitesi",
      alphaQuality,
      alphaEvidence,
      alphaRows.length,
      "Gerçekleşmiş yön sonucu, karar anındaki tahmini round-trip maliyeti geçebildiği ölçüde başarılı sayılır.",
      {
        gross_direction_hit_pct: round1(alphaGrossRate),
        after_cost_favorable_pct: round1(alphaAfterCostRate),
        cost_matched_samples: alphaCostSamples,
        horizons: [...horizon.entries()].map(([seconds, v]) => ({
          seconds,
          samples: v.n,
          gross_hit_pct: round1(ratioPct(v.gross, v.n)),
          after_cost_favorable_pct: round1(ratioPct(v.after, v.costN)),
        })).sort((a, b) => a.seconds - b.seconds),
      },
    );

    // 2) Eyes / ears / nerves: prospective sensor calibration plus actual collector transfer health.
    const sensorHitRate = ratioPct(sensorHits, sensorTotal);
    const flow = weightedRunHealth(collectorRows);
    const sensorQuality = sensorTotal > 0 ? sensorHitRate * 0.65 + flow.score * 0.35 : flow.score;
    const sensorEvidence = 0.7 * evidenceCurve(sensorTotal, 10000) + 0.3 * evidenceCurve(collectorRows.length, 500);
    const sensors = component(
      "sensors",
      "Gözler / Kulaklar / Sinir Sistemi",
      "Veri Algısı ve Akış",
      sensorQuality,
      sensorEvidence,
      sensorTotal,
      "Sensör isabeti ile son 24 saatteki ana veri toplayıcılarının SUCCESS/DEGRADED/FAILED akışı birlikte ölçülür.",
      {
        prospective_hit_pct: round1(sensorHitRate),
        calibration_samples: sensorTotal,
        data_exchange_health_pct: flow.score,
        collector_runs_24h: flow,
      },
    );

    // Collector arm report. Main Brian only; separate systems are excluded at query level.
    const armMap = new Map<string, any[]>();
    for (const row of collectorRows) {
      const id = String(row.collector_id ?? "unknown");
      const list = armMap.get(id) ?? [];
      list.push(row);
      armMap.set(id, list);
    }
    const arms = [...armMap.entries()].map(([id, rows]) => {
      const health = weightedRunHealth(rows);
      const stored = rows.reduce((sum, r) => sum + Number(r.stored_records ?? 0), 0);
      const observed = rows.reduce((sum, r) => sum + Number(r.observed_records ?? 0), 0);
      return {
        id,
        success_pct: health.score,
        runs: health.total,
        success: health.success,
        degraded: health.degraded,
        failed: health.failed,
        observed_records: observed,
        stored_records: stored,
        transfer_capture_pct: observed > 0 ? round1(ratioPct(stored, observed)) : null,
      };
    }).sort((a, b) => b.runs - a.runs || b.success_pct - a.success_pct);

    // 3) World intelligence: core reasoning, source trust/eligibility and discovery transport.
    const worldCore = weightedRunHealth(worldRuns);
    const avgTrust = worldAssessments.length
      ? 100 * worldAssessments.reduce((sum, r) => sum + Number(r.trust_score ?? 0), 0) / worldAssessments.length
      : 0;
    const eligibleResearch = worldAssessments.filter((r) => Boolean(r.eligible_for_research)).length;
    const eligibleDecision = worldAssessments.filter((r) => Boolean(r.eligible_for_decision_evidence)).length;
    const researchPct = ratioPct(eligibleResearch, worldAssessments.length);
    const decisionPct = ratioPct(eligibleDecision, worldAssessments.length);
    const discoveryRows = collectorRows.filter((r) => String(r.collector_id) === "brian-world-discovery-eye-v1");
    const discovery = weightedRunHealth(discoveryRows);
    const worldQuality = 0.40 * worldCore.score + 0.25 * avgTrust + 0.20 * researchPct + 0.15 * discovery.score;
    const worldEvidence = 0.55 * clamp(100 * sourceCount / 1000) + 0.45 * evidenceCurve(worldRuns.length, 200);
    const world = component(
      "world",
      "Gövde / Dünya Hafızası",
      "Dünya Ekonomi ve Kaynak Zekâsı",
      worldQuality,
      worldEvidence,
      sourceCount,
      "Kaynak sayısı tek başına puan değildir; çekirdek World Brain çalışma sağlığı, kaynak güveni ve araştırmaya uygunluk ile birlikte değerlendirilir.",
      {
        discovered_sources: sourceCount,
        assessed_sample: worldAssessments.length,
        average_trust_pct: round1(avgTrust),
        eligible_for_research_pct: round1(researchPct),
        eligible_for_decision_evidence_pct: round1(decisionPct),
        world_core_health_pct: worldCore.score,
        discovery_eye_health_pct: discovery.score,
        discovery_eye_runs_24h: discovery.total,
      },
    );

    // 4) Arms / lab: clean prospective experiments, code artifacts and governance decisions.
    const cleanResults = experimentResults.filter((r) => Boolean(r.data_quality_ok) && !Boolean(r.leakage_detected));
    const cleanRate = ratioPct(cleanResults.length, experimentResults.length);
    const artifactPassed = artifacts.filter((r) => r.passed === true && r.protected_scope_clear === true && !r.leakage_detected).length;
    const artifactPassRate = ratioPct(artifactPassed, artifacts.length);
    const decided = promotions.length;
    const promoteCount = promotions.filter((p) => p.decision === "PROMOTE_CANDIDATE").length;
    const keepCount = promotions.filter((p) => p.decision === "KEEP_EXPERIMENTAL").length;
    const rejectCount = promotions.filter((p) => p.decision === "REJECT").length;
    const governanceCoverage = decided > 0 ? 100 : 0;
    const labQuality = 0.45 * cleanRate + 0.35 * artifactPassRate + 0.20 * governanceCoverage;
    const labEvidence = clamp(
      50 * experimentResults.length / 100 +
      30 * artifacts.length / 50 +
      20 * promotions.length / 50,
    );
    const lab = component(
      "lab",
      "Kollar / Laboratuvar",
      "Araştırma, Eğitim ve Öz-Gelişim",
      labQuality,
      labEvidence,
      experimentResults.length,
      "Bir fikrin çok üretilmesi başarı sayılmaz; temiz deney, sızıntısız test, korumalı kod artefaktı ve Promotion Council kararı kanıt sayılır.",
      {
        experiment_results: experimentResults.length,
        clean_experiment_pct: round1(cleanRate),
        artifacts: artifacts.length,
        safe_artifact_pass_pct: round1(artifactPassRate),
        codegen_requests: codegenCount,
        promotion_decisions: decided,
        promoted: promoteCount,
        kept_experimental: keepCount,
        rejected: rejectCount,
      },
    );

    // 5) Legs / treasury: deliberately UNPROVEN until enough actual OPEN/EXIT lifecycle evidence exists.
    const exits = treasuryActions.filter((a) => a.kind === "EXIT");
    const opens = treasuryActions.filter((a) => a.kind === "OPEN");
    const treasury = treasuryRows[0] ?? null;
    const startEquity = Number(treasury?.starting_equity_usd ?? 0);
    const equity = Number(treasury?.equity_usd ?? 0);
    const returnPct = startEquity > 0 ? 100 * (equity - startEquity) / startEquity : 0;
    const treasuryEvidence = clamp(100 * exits.length / 100);
    let treasuryQuality: number | null = null;
    if (exits.length >= 20) {
      // Once lifecycle evidence exists, quality starts neutral and is moved only by observed net treasury return.
      treasuryQuality = clamp(50 + clamp(returnPct * 5, -40, 40));
    }
    const treasuryComponent = component(
      "treasury",
      "Bacaklar / Uygulama",
      "Hazine ve Sermaye Kullanımı",
      treasuryQuality,
      treasuryEvidence,
      treasuryActions.length,
      exits.length < 20
        ? "Henüz yeterli tamamlanmış OPEN→EXIT döngüsü yok. Sağlıklı servis çalışması yatırım başarısı sayılmadığı için puan uydurulmaz."
        : "Tamamlanmış hazine döngüleri ve net equity değişimi üzerinden ölçülür.",
      {
        opens: opens.length,
        exits: exits.length,
        starting_equity_usd: startEquity,
        equity_usd: equity,
        treasury_return_pct: round1(returnPct),
        deployment_pct: round1(100 * Number(treasury?.deployment_pct ?? 0)),
        realized_pnl_usd: Number(treasury?.realized_pnl_usd ?? 0),
        cumulative_costs_usd: Number(treasury?.cumulative_costs_usd ?? 0),
      },
    );

    const components = [alpha, sensors, world, lab, treasuryComponent];
    const weights: Record<string, number> = { alpha: 0.30, sensors: 0.20, world: 0.15, lab: 0.20, treasury: 0.15 };

    // Geometric maturity is intentional: one strong organ cannot hide an unproven critical organ.
    let logSum = 0;
    for (const c of components) logSum += (weights[c.id] ?? 0) * Math.log(Math.max(2, Number(c.maturity_pct)));
    const overallMaturity = clamp(Math.exp(logSum));
    const measuredQualityWeight = components.reduce((sum, c) => sum + (c.quality_pct === null ? 0 : (weights[c.id] ?? 0)), 0);
    const overallQuality = measuredQualityWeight > 0
      ? components.reduce((sum, c) => sum + (c.quality_pct === null ? 0 : Number(c.quality_pct) * (weights[c.id] ?? 0)), 0) / measuredQualityWeight
      : 0;
    const evidenceConfidence = components.reduce((sum, c) => sum + Number(c.evidence_pct) * (weights[c.id] ?? 0), 0);

    const provenComponents = components.filter((c) => Number(c.evidence_pct) >= 10);
    const strengths = [...provenComponents]
      .sort((a, b) => Number(b.maturity_pct) - Number(a.maturity_pct))
      .slice(0, 3)
      .map((c) => ({ id: c.id, name: c.name, maturity_pct: c.maturity_pct, reason: c.rationale }));
    const weaknesses = [...provenComponents]
      .sort((a, b) => Number(a.maturity_pct) - Number(b.maturity_pct))
      .slice(0, 3)
      .map((c) => ({ id: c.id, name: c.name, maturity_pct: c.maturity_pct, reason: c.rationale }));
    const unproven = components.filter((c) => Number(c.evidence_pct) < 10)
      .map((c) => ({ id: c.id, name: c.name, evidence_pct: c.evidence_pct, reason: c.rationale }));

    const uniqueGaps = new Map<string, any>();
    for (const gap of gaps) if (!uniqueGaps.has(String(gap.gap_id))) uniqueGaps.set(String(gap.gap_id), gap);
    const gapList = [...uniqueGaps.values()]
      .filter((g) => ["CRITICAL", "HIGH", "MEDIUM"].includes(String(g.severity)))
      .slice(0, 10);

    const featureSignals = promotions.slice(0, 12).map((p) => ({
      experiment_id: p.experiment_id,
      observed_at: p.decided_at,
      signal: p.decision === "PROMOTE_CANDIDATE" ? "STRENGTH_CANDIDATE" : p.decision === "REJECT" ? "WEAK_REJECTED" : "NEEDS_MORE_EVIDENCE",
      score: Number(p.score ?? 0),
      decision: p.decision,
      reasons: p.reasons ?? [],
    }));

    return out({
      status: "ONLINE",
      observed_at: new Date().toISOString(),
      model: "brian.development-anatomy.v1",
      score_contract: {
        quality: "Observed behavioral quality only. Service uptime alone is not success.",
        evidence: "Sample depth / coverage. Low evidence prevents false confidence.",
        component_maturity: "quality × evidence",
        overall_maturity: "weighted geometric maturity across critical organs",
        note: "An unproven critical organ constrains the whole Brian score; strong infrastructure cannot hide missing real-world evidence.",
      },
      overall: {
        brain_development_pct: round1(overallMaturity),
        measured_quality_pct: round1(overallQuality),
        evidence_confidence_pct: round1(evidenceConfidence),
        tier: maturityTier(overallMaturity),
        jobs_registered: jobCount,
        data_exchange_health_pct: flow.score,
      },
      components,
      arms,
      strengths,
      weaknesses,
      unproven,
      capability_gaps: gapList,
      feature_signals: featureSignals,
      latest_promotions: promotions.slice(0, 12),
      shadow_only: true,
      live_execution: false,
    }, 200, origin);
  } catch (error) {
    return out({
      status: "DEGRADED",
      error: String(error),
      shadow_only: true,
      live_execution: false,
    }, 500, origin);
  }
});

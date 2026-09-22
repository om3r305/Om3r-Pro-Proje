// Pure, side-effect-free scoring logic for brian-derivatives-eye, split out of index.ts so it
// can be exercised directly by logic.test.ts without a live network/Supabase dependency.

export const EVIDENCE = "PROSPECTIVE_DEVELOPMENT_SHADOW";

export function finite(v: unknown, fallback = 0): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

export function clip(v: number): number {
  return Math.max(0, Math.min(1, v));
}

export function sign(v: number): number {
  return v > 0 ? 1 : v < 0 ? -1 : 0;
}

export function bytes(v: string): Uint8Array {
  return new TextEncoder().encode(v);
}

export async function sha(v: string | Uint8Array): Promise<string> {
  const b = typeof v === "string" ? bytes(v) : v;
  const d = new Uint8Array(await crypto.subtle.digest("SHA-256", b));
  return [...d].map((x) => x.toString(16).padStart(2, "0")).join("");
}

export interface DirectionalSignal {
  direction: number;
  strength: number;
}

/** Contrarian signal: an extreme funding rate is faded, not followed. Null below the noise threshold. */
export function fundingCrowdingSignal(funding: number): DirectionalSignal | null {
  if (Math.abs(funding) < 0.0002) return null;
  return { direction: -sign(funding), strength: clip(Math.abs(funding) / 0.001) };
}

/** Confirmation signal: open interest and price expanding together, in the direction of price. */
export function oiPriceConfirmationSignal(oiChange: number, priceReturn: number): DirectionalSignal | null {
  if (Math.abs(oiChange) < 0.003 || Math.abs(priceReturn) < 0.001) return null;
  return {
    direction: sign(priceReturn),
    strength: clip((Math.abs(oiChange) / 0.02 + Math.abs(priceReturn) / 0.01) / 2),
  };
}

/** Public taker buy/sell ratio imbalance. Null inside the neutral band around 1.0. */
export function takerImbalanceSignal(ratio: number): DirectionalSignal | null {
  if (ratio < 1.08 && ratio > 0.925) return null;
  return { direction: ratio > 1 ? 1 : -1, strength: clip(Math.abs(Math.log(Math.max(ratio, 1e-12))) / 0.35) };
}

export async function makeObs(
  eyeId: string,
  templateId: string,
  assetId: string,
  family: string,
  group: string,
  observedAt: string,
  direction: number,
  strength: number,
  confidence: number,
  captureId: string,
  reason: string,
  metadata: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  const observationId = await sha(`${eyeId}|${observedAt}|${direction}|${strength.toFixed(12)}`);
  return {
    observation_id: observationId, eye_id: eyeId, template_id: templateId, asset_id: assetId,
    market_domain: "crypto", sensor_family: family, horizon: "FAST_5_30M", independent_group: group,
    observed_at: observedAt, direction, strength: clip(strength), confidence: clip(confidence), reliability: 0.5,
    available: true, source_ids: [captureId], reason, evidence_class: EVIDENCE, shadow_only: true,
    live_execution: false, metadata,
  };
}

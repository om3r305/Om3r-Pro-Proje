// Brian DIP V8.4.1 Authority contracts. SHADOW ONLY.
// V8.3 remains frozen. V8.4 Package-1 history remains append-only baseline evidence.

export {
  canonicalize,
  canonicalJson,
  clip,
  evaluatePath,
  executionCalibration,
  hash,
  initialRuntime,
  manifestHash,
  mean,
  n,
  same,
  unavailableCalibration,
} from "./dip_v84_contract.ts";
export type {
  Bar,
  Direction,
  ExecutionCalibration,
  FundingSettlement,
  J,
  Pivot,
  Position,
  Resolution,
  Rules,
  Runtime,
  Segment,
  Setup,
  Struct,
} from "./dip_v84_contract.ts";

import { n, type J } from "./dip_v84_contract.ts";

export const SYMBOL = "ETHUSDT" as const;
export const RELEASE_ID = "dip-v841-brian-authority-20260911.1" as const;
export const RELEASE_STAGE = "PRODUCTION_SHADOW_AUTHORITY" as const;
export const ENGINE_VERSION = "brian-dip-v841-authority" as const;
export const POLICY_VERSION = "dip-v841-brian-authority-20260911.1" as const;
export const DECISION_REVISION = "dip-v841-authority-occurrence-20260911.1" as const;
export const METRIC_VERSION = "target-before-invalidation-v841.1" as const;
export const RESOLVER_VERSION = "dip-v841-path-20260911.1" as const;
export const ENTRY_GUARD_VERSION = "brian-setup-specific-v1" as const;
export const TARGET_PLANNER_VERSION = "brian-structural-ladder-v1" as const;
export const EXECUTION_MODEL_VERSION = "fill-forward-frozen-exit-v841.1" as const;
export const COST_MODEL_VERSION = "FILL_FORWARD_V841_1" as const;
export const MARKET_DATA_CONTRACT_VERSION = "binance-usdm-sealed-v84.1" as const;
export const DB_CONTRACT_VERSION = "brian-dip-v84-db-2-authority" as const;
export const CALIBRATION_FAMILY_ID = "dip-v841-brian-authority-family-1" as const;
export const STRATEGY_MANIFEST_HASH = "1b9bc41c4edd2f0b9896be316843bde4cd27b67793d4d3912fb3450fe47ee7f1" as const;

// Reviewed behavior identity. The following seal-only commit changes this metadata, not strategy behavior.
export const LOGIC_HASH = "git:b16ada5b40e5da2b9766946ddd83647e6c299763" as const;

export const MAX_HOLD_MS = 90 * 60_000;
export const MAX_SHADOW_LEVERAGE:1 = 1;
export const MIN_EXECUTION_CAL_SAMPLES = 40;

export const STRATEGY_MANIFEST = Object.freeze({
  browser_execution:false,
  calibration:{entry_blocking:false,execution_source:"ACTUAL_SHADOW_FILLS_ONLY",forecast_authority:false},
  cost:{entry_price:"ask/bid plus opening slippage",exit_model:"frozen-at-open executable-side conversion once",expected_funding_horizon_minutes:90,version:COST_MODEL_VERSION},
  decision_authority:"BRIAN_SINGLE_STRATEGY_AUTHORITY",
  decision_cadence_seconds:15,
  live_execution:false,
  max_hold_ms:MAX_HOLD_MS,
  risk:{capital_allocation:"BRIAN_0_TO_1_AVAILABLE_CASH",fixed_notional_fraction_cap:false,max_shadow_leverage:MAX_SHADOW_LEVERAGE},
  server_authoritative:true,
  shadow_only:true,
  soft_evidence:["TARGET_BELOW_COST","ECONOMIC_RR_TOO_LOW","WAIT_RETEST","ENTRY_TOO_LATE","DIRECTION_REFEREE_REJECT","COUNTER_STRUCTURE","OPPOSING_MULTI_FLOW","RAW_CONVICTION_LOW"],
  structure:{atr_period:14,bos_buffer_atr:.15,pivot_left:3,pivot_right:3},
  symbol:SYMBOL,
  target:{l1_mandatory:false,policy:"BRIAN_STRUCTURAL_LADDER",strategy_cost_gate:false},
});

export function validateSession(config:J):void {
  if(config.release_id!==RELEASE_ID||config.logic_hash!==LOGIC_HASH||config.strategy_manifest_hash!==STRATEGY_MANIFEST_HASH||config.calibration_family_id!==CALIBRATION_FAMILY_ID||config.db_contract_version!==DB_CONTRACT_VERSION)throw Error("V841_RELEASE_CONTRACT_MISMATCH");
  if(config.engine_version!==ENGINE_VERSION||config.policy_version!==POLICY_VERSION||config.decision_revision!==DECISION_REVISION||config.metric_version!==METRIC_VERSION)throw Error("V841_VERSION_CONTRACT_MISMATCH");
  if(JSON.stringify(config.symbols)!==JSON.stringify([SYMBOL]))throw Error("V841_SYMBOL_CONTRACT_MISMATCH");
  if(config.shadow_only!==true||config.live_execution!==false||config.browser_execution!==false||config.server_authoritative!==true)throw Error("V841_INVALID_SHADOW_CONTRACT");
  if(config.allow_shadow_short!==true||n(config.max_shadow_leverage,0)!==MAX_SHADOW_LEVERAGE)throw Error("V841_INVALID_RISK_CONTRACT");
  if(!["OBSERVE","SHADOW_PAPER"].includes(String(config.execution_mode)))throw Error("V841_INVALID_EXECUTION_MODE");
  if(config.decision_authority!=="BRIAN")throw Error("V841_AUTHORITY_CONTRACT_MISMATCH");
}

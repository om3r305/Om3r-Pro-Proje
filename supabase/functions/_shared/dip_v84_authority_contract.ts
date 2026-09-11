// Brian DIP V8.4.3 Long-only Stateful Profit Protect contracts. SHADOW ONLY.
// V8.3 and prior V8.4/V8.4.1/V8.4.2 evidence remain historical baselines.

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
export const RELEASE_ID = "dip-v843-profit-protect-20260911.1" as const;
export const RELEASE_STAGE = "PRODUCTION_SHADOW_LONG_ONLY" as const;
export const ENGINE_VERSION = "brian-dip-v843-profit-protect" as const;
export const POLICY_VERSION = "dip-v843-profit-protect-20260911.1" as const;
export const DECISION_REVISION = "dip-v843-long-occurrence-20260911.1" as const;
export const METRIC_VERSION = "target-before-invalidation-v843.1" as const;
export const RESOLVER_VERSION = "dip-v843-path-20260911.1" as const;
export const ENTRY_GUARD_VERSION = "brian-live-entry-quality-v2" as const;
export const TARGET_PLANNER_VERSION = "brian-dynamic-advisory-target-v3" as const;
export const EXECUTION_MODEL_VERSION = "stateful-profit-protect-v843.1" as const;
export const COST_MODEL_VERSION = "FILL_FORWARD_V843_1" as const;
export const MARKET_DATA_CONTRACT_VERSION = "binance-usdm-live-book-closed-structure-v842.1" as const;
export const DB_CONTRACT_VERSION = "brian-dip-v84-db-2-authority" as const;
export const CALIBRATION_FAMILY_ID = "dip-v843-profit-protect-family-1" as const;
export const STRATEGY_MANIFEST_HASH = "6c2ca6cba10903220240ebb0c078d14b31b68fe8edd45a623b5e0bb5bc78a194" as const;

// Sealed only after the behavior commit is fixed. Activation metadata does not alter trade logic.
export const LOGIC_HASH = "UNSEALED_GITHUB_ONLY" as const;

export const MAX_HOLD_MS = 90 * 60_000;
export const MAX_SHADOW_LEVERAGE:1 = 1;
export const MIN_EXECUTION_CAL_SAMPLES = 40;
export const DECISION_CADENCE_SECONDS = 10;
export const MIN_ENTRY_QUALITY = 0.34;
export const PROFIT_PROTECT_ARM_BPS = 25;
export const PROFIT_PROTECT_GIVEBACK_BPS = 7;
export const PROFIT_PROTECT_MOMENTUM_ATR_CEILING = 0.8;
export const SELL_SIGNAL_LATCH_MS = 30_000;

export const STRATEGY_MANIFEST = Object.freeze({
  browser_execution:false,
  calibration:{entry_blocking:false,execution_source:"ACTUAL_SHADOW_FILLS_ONLY",forecast_authority:false},
  cost:{entry_price:"ask plus opening slippage",exit_model:"stateful long-only executable-side conversion once",expected_funding_horizon_minutes:90,version:COST_MODEL_VERSION},
  decision_authority:"BRIAN_LONG_ONLY_STATEFUL_PROFIT_PROTECT",
  decision_cadence_seconds:DECISION_CADENCE_SECONDS,
  entry:{chase_protection:"BRIAN_INTERNAL",direction_and_quality_separate:true,minimum_entry_quality:MIN_ENTRY_QUALITY},
  exit:{
    hard_stop:"STRUCTURAL_INVALIDATION",
    profit_protect:{arm_profit_bps:PROFIT_PROTECT_ARM_BPS,giveback_bps:PROFIT_PROTECT_GIVEBACK_BPS,momentum_atr_ceiling:PROFIT_PROTECT_MOMENTUM_ATR_CEILING,structural_resistance_arms:true},
    sell_signal_latch_ms:SELL_SIGNAL_LATCH_MS,
    soft_sell_vote_decay:true,
    strong_sell_immediate:true,
  },
  live_execution:false,
  max_hold_ms:MAX_HOLD_MS,
  risk:{capital_allocation:"BRIAN_CONFIDENCE_X_ENTRY_QUALITY",fixed_notional_fraction_cap:false,max_shadow_leverage:MAX_SHADOW_LEVERAGE},
  server_authoritative:true,
  shadow_only:true,
  short_entries:false,
  structure:{atr_period:14,bos_buffer_atr:.15,pivot_left:3,pivot_right:3,live_tactical_price:"BINANCE_USDM_BOOK_MID"},
  symbol:SYMBOL,
  target:{hard_take_profit:false,policy:"BRIAN_DYNAMIC_ADVISORY_TARGET",trailing_thesis:true},
});

export function validateSession(config:J):void {
  if(config.release_id!==RELEASE_ID||config.logic_hash!==LOGIC_HASH||config.strategy_manifest_hash!==STRATEGY_MANIFEST_HASH||config.calibration_family_id!==CALIBRATION_FAMILY_ID||config.db_contract_version!==DB_CONTRACT_VERSION)throw Error("V843_RELEASE_CONTRACT_MISMATCH");
  if(config.engine_version!==ENGINE_VERSION||config.policy_version!==POLICY_VERSION||config.decision_revision!==DECISION_REVISION||config.metric_version!==METRIC_VERSION)throw Error("V843_VERSION_CONTRACT_MISMATCH");
  if(JSON.stringify(config.symbols)!==JSON.stringify([SYMBOL]))throw Error("V843_SYMBOL_CONTRACT_MISMATCH");
  if(config.shadow_only!==true||config.live_execution!==false||config.browser_execution!==false||config.server_authoritative!==true)throw Error("V843_INVALID_SHADOW_CONTRACT");
  if(config.allow_shadow_short!==false||n(config.max_shadow_leverage,0)!==MAX_SHADOW_LEVERAGE)throw Error("V843_INVALID_LONG_ONLY_CONTRACT");
  if(!["OBSERVE","SHADOW_PAPER"].includes(String(config.execution_mode)))throw Error("V843_INVALID_EXECUTION_MODE");
  if(config.decision_authority!=="BRIAN")throw Error("V843_AUTHORITY_CONTRACT_MISMATCH");
}
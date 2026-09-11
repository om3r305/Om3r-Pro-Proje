// Brian DIP V8.4.4 Cycle Forecast + Harvest contracts. SHADOW ONLY.
// V8.4.3 remains the immutable production baseline for before/after comparison.

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
export const RELEASE_ID = "dip-v844-cycle-forecast-20260911.1" as const;
export const RELEASE_STAGE = "PRODUCTION_SHADOW_LONG_ONLY" as const;
export const ENGINE_VERSION = "brian-dip-v844-cycle-forecast" as const;
export const POLICY_VERSION = "dip-v844-cycle-forecast-20260911.1" as const;
export const DECISION_REVISION = "dip-v844-cycle-occurrence-20260911.1" as const;
export const METRIC_VERSION = "forecast-before-entry-v844.1" as const;
export const RESOLVER_VERSION = "dip-v844-path-20260911.1" as const;
export const ENTRY_GUARD_VERSION = "brian-cycle-entry-timing-v4" as const;
export const TARGET_PLANNER_VERSION = "brian-forecast-destination-v4" as const;
export const EXECUTION_MODEL_VERSION = "stateful-cycle-harvest-v844.1" as const;
export const COST_MODEL_VERSION = "FILL_FORWARD_V844_1" as const;
export const MARKET_DATA_CONTRACT_VERSION = "binance-usdm-live-book-closed-structure-v842.1" as const;
export const DB_CONTRACT_VERSION = "brian-dip-v84-db-2-authority" as const;
export const CALIBRATION_FAMILY_ID = "dip-v844-cycle-forecast-family-1" as const;
export const STRATEGY_MANIFEST_HASH = "b41bcf3d3ff6f5359937894f03d7165bdef3b425bc9bc6ff7e8abb6669b89449" as const;

// Immutable behavior identity: final behavior commit before this metadata-only seal.
export const LOGIC_HASH = "git:e210589642dbac13e65570a6d729360c7d656b54" as const;

export const MAX_HOLD_MS = 90 * 60_000;
export const MAX_SHADOW_LEVERAGE:1 = 1;
export const MIN_EXECUTION_CAL_SAMPLES = 40;
export const DECISION_CADENCE_SECONDS = 10;
export const MIN_ENTRY_QUALITY = 0.46;
export const MIN_FORECAST_NET_EDGE_BPS = 5;
export const RANGE_PRIMARY_RANK_CAP = 4;
export const STRONG_PRIMARY_RANK_CAP = 5;
export const CHASE_RANGE_POSITION = 0.58;
export const CHASE_MOMENTUM_ATR = 0.85;
export const MIN_THESIS_HOLD_MS = 60_000;
export const HARVEST_BREAK_EVEN_BUFFER_BPS = 5;
export const HARVEST_GIVEBACK_BPS = 6;
export const REBASE_PULLBACK_BPS = 8;
export const REBASE_PULLBACK_ATR = 0.65;
export const REBASE_RECLAIM_ATR = 0.18;
// Compatibility exports used by the legacy feature generator. V8.4.4 overrides
// the final arm/exit policy in cycle.ts with actual break-even-aware harvest logic.
export const PROFIT_PROTECT_ARM_BPS = 25;
export const PROFIT_PROTECT_GIVEBACK_BPS = HARVEST_GIVEBACK_BPS;
export const PROFIT_PROTECT_MOMENTUM_ATR_CEILING = 0.8;

export const STRATEGY_MANIFEST = Object.freeze({
  browser_execution:false,
  calibration:{entry_blocking:false,execution_source:"ACTUAL_SHADOW_FILLS_ONLY",forecast_authority:false},
  cost:{entry_price:"ask plus opening slippage",exit_model:"stateful cycle harvest executable-side conversion once",expected_funding_horizon_minutes:90,version:COST_MODEL_VERSION},
  cycle:{phases:["SEEK_DIP","ENTER_RECOVERY","RIDE","HARVEST","REBASE"],same_leg_reentry:false},
  decision_authority:"BRIAN_LONG_ONLY_CYCLE_FORECAST_HARVEST",
  decision_cadence_seconds:DECISION_CADENCE_SECONDS,
  entry:{chase_protection:"RANGE_PLUS_MOMENTUM",direction_and_quality_separate:true,minimum_entry_quality:MIN_ENTRY_QUALITY,post_exit_rebase_required:true},
  exit:{
    hard_stop:"STRUCTURAL_INVALIDATION",
    harvest:{break_even_buffer_bps:HARVEST_BREAK_EVEN_BUFFER_BPS,minimum_net_edge_bps:MIN_FORECAST_NET_EDGE_BPS,requires_reachable_primary:true},
    minimum_thesis_hold_ms:MIN_THESIS_HOLD_MS,
    profit_protect:{arm_mode:"ACTUAL_BREAK_EVEN_PLUS_BUFFER",giveback_bps:HARVEST_GIVEBACK_BPS,structural_resistance_arms:true},
    raw_bearish_immediate:false,
    severe_bearish_immediate:true,
    soft_sell_confirmation:"2_CONSECUTIVE_VOTES",
  },
  forecast:{economics_after_forecast:true,far_pivots:"STRETCH_ONLY",no_trade_if_primary_below_cost:true,policy:"NEAR_TERM_REACHABLE_DESTINATION_FIRST",range_primary_rank_cap:RANGE_PRIMARY_RANK_CAP,strong_primary_rank_cap:STRONG_PRIMARY_RANK_CAP},
  live_execution:false,
  max_hold_ms:MAX_HOLD_MS,
  risk:{capital_allocation:"BRIAN_CONFIDENCE_X_ENTRY_QUALITY",fixed_notional_fraction_cap:false,max_shadow_leverage:MAX_SHADOW_LEVERAGE},
  server_authoritative:true,
  shadow_only:true,
  short_entries:false,
  structure:{atr_period:14,bos_buffer_atr:.15,pivot_left:3,pivot_right:3,live_tactical_price:"BINANCE_USDM_BOOK_MID"},
  symbol:SYMBOL,
  target:{hard_take_profit:false,policy:"BRIAN_FORECAST_PRIMARY_WITH_STRETCH_TELEMETRY",trailing_thesis:true},
});

export function validateSession(config:J):void {
  if(config.release_id!==RELEASE_ID||config.logic_hash!==LOGIC_HASH||config.strategy_manifest_hash!==STRATEGY_MANIFEST_HASH||config.calibration_family_id!==CALIBRATION_FAMILY_ID||config.db_contract_version!==DB_CONTRACT_VERSION)throw Error("V844_RELEASE_CONTRACT_MISMATCH");
  if(config.engine_version!==ENGINE_VERSION||config.policy_version!==POLICY_VERSION||config.decision_revision!==DECISION_REVISION||config.metric_version!==METRIC_VERSION)throw Error("V844_VERSION_CONTRACT_MISMATCH");
  if(JSON.stringify(config.symbols)!==JSON.stringify([SYMBOL]))throw Error("V844_SYMBOL_CONTRACT_MISMATCH");
  if(config.shadow_only!==true||config.live_execution!==false||config.browser_execution!==false||config.server_authoritative!==true)throw Error("V844_INVALID_SHADOW_CONTRACT");
  if(config.allow_shadow_short!==false||n(config.max_shadow_leverage,0)!==MAX_SHADOW_LEVERAGE)throw Error("V844_INVALID_LONG_ONLY_CONTRACT");
  if(!["OBSERVE","SHADOW_PAPER"].includes(String(config.execution_mode)))throw Error("V844_INVALID_EXECUTION_MODE");
  if(config.decision_authority!=="BRIAN"||config.worker_lease_key!=="brian-dip-v844-cycle-worker")throw Error("V844_AUTHORITY_CONTRACT_MISMATCH");
}
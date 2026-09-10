# Brian DIP V8.4 — FINAL SPEC v3 (DESIGN LOCKED)

Status: **GitHub-only preparation. Not deployed.**

## Non-negotiable safety

- `shadow_only=true`
- `live_execution=false`
- `browser_execution=false`
- `server_authoritative=true`
- No Binance order endpoints.
- Confirmed structure and promotion evidence use sealed candles only.
- V8.3 deployed artifact, tables, RPCs, cron, foresight and shared source remain untouched.

## V8.3 baseline isolation

V8.4 owns a separate control/data plane:

- `brian_dip_v84_session_events`
- `brian_dip_v84_runtime`
- `brian_dip_v84_worker_leases`
- `brian_dip_v84_decisions`
- `brian_dip_v84_ledger`
- `brian_dip_v84_commit(...)`
- `brian_dip_v84_forecast_calibration(...)`
- `brian_dip_v84_execution_calibration(...)`
- `brian-dip-v84-worker`
- `brian-dip-v84-foresight`
- `_shared/dip_v84_*`

V8.4 never alters `brian_dip_v82_*`, `brian_dip_v8_*`, `brian-dip-dual-shadow-worker`, `brian-dip-foresight`, or `_shared/dip_v8_dual.ts`.

## Release contract

Every authoritative V8.4 decision is bound to:

- `release_id`
- `strategy_manifest_hash`
- `logic_hash`
- `calibration_family_id`
- `db_contract_version`
- engine / policy / decision / metric / resolver / entry-guard / target-planner / execution-model / cost-model versions

Unexpected mismatch is **FAIL_CLOSED**. A new calibration family is created only by an explicit new release; no automatic fallback to old evidence.

## Calibration firewall

Three states are mandatory:

- `UNAVAILABLE`: RPC/contract/release/family failure. Entry prohibited.
- `COLD_NEW_FAMILY`: valid family but insufficient execution fills. 1x cold shadow policy only. `CALIBRATION_NO_EDGE` is not evaluated.
- `WARM`: sufficient valid execution evidence. Statistical gates may be evaluated only by a later explicitly promoted release.

Forecast calibration never controls cold/warm state, sizing, max-notional, `CALIBRATION_NO_EDGE`, or leverage.

Package 1 evidence release is intentionally capped at **1x / 8% gross-notional ceiling / 0.5% account risk**. Warm-risk promotion, 20% exposure and 2x are disabled until a separate OOS-validated release.

Execution calibration source: **actual V8.4 shadow ledger fills/outcomes only**. An execution `AMBIGUOUS` result is a conservative loss, not a null sample. Forecast ambiguous outcomes remain forecast uncertainty and use a separate aggregator.

## Cost contract

Three distinct concepts:

1. `REFERENCE_ROUNDTRIP_COST` — research/reporting only.
2. `FILL_FORWARD_COST` — remaining economic friction from actual simulated fill.
3. `REALIZED_EXECUTION_COST` — realized shadow-ledger accounting.

Entry is actual simulated fill: LONG = ask + opening slippage; SHORT = bid - opening slippage. Opening spread/slippage already embedded in fill are not charged again.

Exit rule: executable-side conversion exactly once, exit slippage exactly once, opening fee once, closing fee once, funding once. Expected funding is an entry hurdle; realized settlements are ledger P&L.

## Immutable occurrence identity

A 15-second evaluation is not automatically a new occurrence. A new sealed candle is not automatically a new occurrence.

A new occurrence is allowed only when:

- the prior occurrence became terminal, or
- immutable thesis identity materially changed (`setup`, `direction`, trigger identity/pivot time, L1 identity, stop identity/source, material structure fingerprint), or
- a declared structural state transition creates a child occurrence (future Package 2).

OFI, book pressure or direction-score movement alone cannot rewrite or mint a thesis occurrence.

## Deterministic L1

Live Package 1 policy name: `NEAREST_STRUCTURAL_THEN_ECONOMIC_GATE`.

L1 is the deterministic first forward structural level known at signal time:

- correct directional level kind
- `confirmed_at <= signal_at`
- price strictly beyond actual simulated fill
- canonical level identity / deterministic clustering
- sort by distance from fill
- tie-break: higher timeframe, earlier confirmation, stable level id

Freshness (`UNTOUCHED`, `TOUCHED`, `SWEPT`, `CLOSE_CROSSED`) is telemetry initially, not a learned quality weight.

If L1 passes fill-forward economics and all existing gates: `L1_EXECUTABLE`.

If L1 exists but is non-economic: `L1_BLOCKING`, no fill. Farther economic levels are research only. Price moving away later never converts the old blocking occurrence into executable.

Research-only skip policy is explicitly named `NEAREST_ECONOMIC_SKIP_OBSTACLE` and is never live in Package 1.

## L1 blocking lifecycle

First terminal event wins:

- frozen executable stop reached
- confirmed opposing structural invalidation
- original structural identity materially superseded
- forecast expiry (`signal_at + 90m`)
- future Package-2 L1 acceptance event

If ordering between acceptance and invalidation cannot be proven: ambiguous, no promotion.

## Position horizon

Forecast horizon: `signal_at + 90m`.

Position horizon: `opened_at + 90m`.

Counterfactual 120/180/240 minute paths are research-only and never rewrite the original outcome.

## Stop-quality rule for Package 1

No arbitrary ATR floor is introduced into the initial live-shadow policy.

Every fill records stop distance in bps, ATR(1m) and setup-TF ATR. Stop-policy and ATR-floor challengers are replay/OOS research.

**Warm/risk promotion is disabled until stop geometry, setup alpha and execution economics pass a separately frozen OOS protocol.**

## Alpha sanity controls

Before risk promotion, compare the real setup engine against:

- same-timestamp randomized direction
- same-timestamp systematically inverted direction
- matched random timing
- simple HTF / 5m BOS baseline
- same-geometry dumb L1 control

Candidate-set telemetry records every candidate and component score so flow/ranking challengers can be replayed honestly.

## Evaluation protocol

Each forward evaluation has immutable `evaluation_protocol_id` with:

- fixed calendar start/end
- primary and secondary metrics
- minimum fills / distinct episodes as sufficiency checks
- predeclared exclusions and failure conditions

If the fixed window ends without sufficient data: `INCONCLUSIVE`. Extending the window requires a new protocol id.

## Package order

### Package 1A — integrity/accounting

Isolation, release contract, calibration firewall, cost contract, lease fencing, append-only telemetry, alpha sanity plumbing.

### Package 1B — replay/research

Stop geometry, ATR-floor, target-policy and flow challengers plus null controls. No live policy mutation.

### Package 1C — isolated shadow evidence release

`L1_EXECUTABLE` / `L1_BLOCKING`, cold 1x only, no L2, no skip-obstacle, no 20%, no 2x.

### Package 2 — future release only

Sealed `L1_ACCEPTED` may create a same-episode child occurrence; current market is re-read and entry/stop/target/cost/RR/guard are recomputed. Old hypothetical fill is never reused.

## Definition of success

V8.4 succeeds by preserving release integrity, calibration isolation, coherent modeled economics, immutable lineage, OOS robustness and strict SHADOW operation. More trades, higher theoretical RR or more leverage are not success criteria.

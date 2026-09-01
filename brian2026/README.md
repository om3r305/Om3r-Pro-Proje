# Brian 2026 — Foundation

Brian 2026 is a **shadow-first adaptive trading research core**.  It is not a
profit guarantee and it does not place live orders in this phase.

## What changed conceptually

The legacy project already had the right DNA: multiple signals, regime logic,
risk controls, learning, evolution, self-heal and a shadow-arena idea.  The new
foundation turns the strongest modern ideas into one controlled loop:

1. **Specialist committee** — trend, momentum, order-book, breakout and mean-reversion brains vote independently.
2. **Meta Trader** — combines the committee and can explicitly choose `WAIT` when edge or agreement is weak.
3. **Long-term memory** — decisions, outcomes and specialist reliability are stored and reused.
4. **Independent Risk Governor** — learning cannot override hard loss/drawdown/spread/position limits.
5. **Deterministic Replay Lab** — counterfactual TP/SL/delay experiments use candle paths and real fees/slippage, never random scores.
6. **Researcher** — clusters recurring losses and proposes bounded, measurable experiments.
7. **Promotion Gate** — candidates must beat objective metrics and walk-forward stability before becoming a champion candidate.
8. **Shadow-first deployment** — the old bot remains untouched while Brian watches the same market and builds an evidence trail.

## Why this structure

It borrows architectural ideas (not source code) from modern quantitative agent
systems: autonomous research/evaluation loops, reinforcement-learning style
environments, multi-agent specialist debate, continual learning, and explicit
memory.  The important addition is governance: a model that learns is never
allowed to promote itself directly into live-money execution.

## First integration target

Use `LegacyShadowBridge` next to the existing signal path.  Feed it real market
features and log its decision.  When a legacy trade closes, call `BrianEngine.learn`
with the linked outcome.  After enough observations, run the replay lab and the
promotion gate.

Runtime state is written under `runtime/brian2026/` and should not be committed.

## Phase 1.1 safety boundary

When the legacy loop enables `brian2026.shadow_enabled`, code-writing background
services (autocoder, auto-repair/file watchers, evolutionary promotion, intent
synthesis, and random TP/SL tuning) are not started or ticked. Brian consumes
only typed snapshots built from completed candles, point-in-time order-book
data, and the observed legacy signal. Legacy random/mock research modules remain
quarantined historical code and are not imported by `brian2026`.

## Phase 2 foundation

- `dataset.py` defines immutable, hashed point-in-time market datasets.
- `features.py` records schema/Brian versions, source times, availability and dataset identity.
- `equity.py` mirrors legacy positions for shadow equity, exposure and drawdown accounting.
- `replay.py` provides deterministic LONG/SHORT/WAIT simulation with explicit costs and fill interfaces.
- `counterfactual.py` compares bounded alternatives on one identical future path.
- `splits.py` creates purged and embargoed walk-forward train/validation/test boundaries.
- `experiments.py` writes immutable, content-addressed experiment manifests.

## Phase 2.1 supervised baseline

Point-in-time samples, completed-candle multi-timeframe joins, train-only preprocessing, validation-only calibration/thresholds, locked walk-forward tests, deterministic replay evaluation, and research-only champion candidates. Logistic regression and gradient boosting are reproducible baselines; no model can execute or promote itself to live trading.

## Phase 2.2 public research data

`python -m brian2026.data` is a public, unauthenticated research-data CLI. It never starts the trading loop and contains no exchange execution methods.

Stages:

- `fetch`: download a bounded Binance public spot kline range and persist immutable raw content.
- `inspect`: display raw identity, instrument, timeframe, and requested range.
- `validate`: normalize in memory and emit deterministic quality diagnostics.
- `build-dataset`: create an immutable Brian MarketDataset only after quality checks.

Example bounded fetch:

`python -m brian2026.data fetch --symbol BTCUSDT --timeframe 1m --start 2024-01-01T00:00:00Z --end 2024-02-01T00:00:00Z --output research_data`

Offline candles become available at their exchange close timestamp. Download time is raw-import provenance, not a claim that the candle was known earlier. Spot OHLCV does not contain historical bid/ask, depth, or funding; those remain unavailable. Any replay spread is explicitly tagged `simulation_assumption`.
# Brian Phase 3.2 — Universe Radar + Smart-Money Consensus

Status: SHADOW_RESEARCH_ONLY

## Purpose

This increment gives Brian a broad market-discovery layer. It does not choose trades and it does not send exchange orders.

The flow is:

`public/prospective observation -> immutable raw capture -> universe ranking -> event/social/on-chain deep research -> opportunity intelligence -> downstream portfolio/risk research`

## Universe Radar

The initial public radar scans Binance Spot USDT instruments and filters on:

- active spot status
- quote asset
- minimum quote volume
- minimum 24h trade count
- minimum positive price
- configurable excluded base assets

Eligible assets receive relative research-priority components for:

- liquidity
- trade activity
- 24h range / volatility
- absolute 24h movement
- top-of-book spread quality when available

A radar score is **not an alpha score** and has no BUY/SELL field. It decides which assets deserve deeper intelligence work.

Missing spread data is neutral/unavailable, never treated as zero spread.

The first universe snapshot is a baseline, not a listing alert. A symbol becomes `newly_observed` only by a chronological diff against a prior snapshot.

## Prospective public collector

`BinancePublicUniverseCollector` uses only unauthenticated public endpoints:

- exchange information
- 24h ticker state
- optional top-of-book ticker

The collector accepts no API key/secret and exposes no order method.

Required exchange/ticker responses fail closed. Top-of-book is optional: failure records a degraded source and leaves spread unavailable instead of fabricating it.

When an `IntelligenceStore` is supplied, raw responses are stored as immutable content-addressed point-in-time captures before their derived universe snapshot is used.

This one-shot collector is a data-plane primitive, not the final always-on cloud runtime. CI/GitHub Actions must not be mistaken for a low-latency trading/event bus. A later cloud collector deployment needs durable storage and an appropriate persistent or scheduled runtime.

## Smart-Money Consensus

Brian must not follow one wallet blindly. Consensus evaluates only whale observations whose flow semantics and entity attribution are sufficiently resolved.

It measures:

- qualified observation count
- unique entity breadth
- bullish vs bearish entity breadth
- unresolved observation fraction
- gross and directional USD flow
- largest-entity concentration
- direction
- confidence
- accumulation/distribution context scores

Strong confidence requires independent entity breadth. A single large wallet is explicitly concentration-penalized and vetoed for insufficient breadth.

Internal transfers and unresolved flows do not create direction. User-generated/low-trust labels do not qualify as smart-money consensus.

## Next layers

- durable prospective cloud capture
- official announcement/event adapters
- provider adapters for credentialed on-chain/social intelligence when legitimate access exists
- smart-money recurrence and wallet-quality history
- source latency/reputation integration
- narrative graph and cross-asset propagation
- derivatives intelligence
- adversarial information defense
- event-driven opportunity tournament

No layer may bypass portfolio/risk research or create authenticated exchange execution.

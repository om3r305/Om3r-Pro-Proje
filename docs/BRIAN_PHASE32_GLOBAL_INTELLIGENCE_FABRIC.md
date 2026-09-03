# Brian Phase 3.2 — Global Intelligence Fabric

Status: FOUNDATION / SHADOW_RESEARCH_ONLY

## Objective

Brian is not a BTC-only trading strategy. Phase 3.2 establishes a provider-agnostic market-intelligence fabric that can discover, verify, rank and remember market-moving events across a broad asset universe before any downstream portfolio decision is considered.

The system must separate **observation**, **truth assessment**, **entity/whale interpretation**, **social authenticity**, **market confirmation**, **opportunity ranking**, **portfolio allocation**, **risk**, and **execution**. No event source is allowed to place an order directly.

## Competitive capability targets

The architecture intentionally absorbs strong ideas seen in modern market-intelligence and quant systems without copying their implementations:

- Smart-money accumulation, holder breadth and wallet-performance context.
- Entity labels, fund-flow tracing, verified-vs-predicted/custom attribution and transaction alerts.
- Cross-platform social volume/sentiment/trend acceleration.
- Out-of-distribution uncertainty and abstention.
- Universe -> signal -> portfolio -> risk -> execution separation.
- Streaming/event-time semantics instead of future-aware batch reasoning.

## Phase 3.2 foundation contracts

### 1. Immutable intelligence event

Every event records:

- asset
- event kind
- source kind and source id
- publication time
- first-observed time
- claim
- directional hypothesis and magnitude
- trust class
- entity confidence
- content fingerprint
- corroboration key
- provenance URI when available
- whether historical timestamp knowledge is point-in-time verified

The event ID is content-addressed and deterministic.

### 2. Event Truth Engine

Brian never treats popularity as truth. Truth assessment considers:

- official/exchange confirmation
- independent source count
- source-kind diversity
- provider/entity confidence
- duplicate-content ratio
- low-trust attribution fraction
- social-only evidence penalty
- manipulation risk

Copied posts cannot become independent corroboration merely by appearing many times.

### 3. Social Forensics

Social bursts track:

- total mentions
- unique authors
- unique-text ratio
- median account age when available
- official/verified-source fraction
- bot likelihood
- cross-platform breadth

The output is an authenticity score, not a BUY/SELL instruction.

Future adapters may include X, Reddit, YouTube/TikTok aggregate providers and other terms-compliant sources. Credentialed providers are disabled by default and secrets must never be committed.

### 4. Whale / Smart-Money Semantics

A large transfer is not automatically bullish or bearish.

Brian distinguishes:

- exchange deposit
- exchange withdrawal
- DEX buy
- DEX sell
- internal transfer
- unresolved flow

Entity attribution has an independent trust/confidence score. User-generated or unknown labels cannot be treated as verified whale identity. Internal transfers have zero assumed economic direction.

Future smart-money logic should measure breadth, recurrence, wallet historical quality, concentration, counterparty type and coordinated-vs-organic behavior rather than following a single wallet blindly.

### 5. Opportunity Radar

Asset intelligence fuses event truth, event strength, social authenticity, whale context and market confirmation into a **priority score**. This score ranks research attention only. It does not create an exchange order.

Hard vetoes include very low truth, very high manipulation risk, unresolved whale-only claims, and strongly inorganic social-only hype.

### 6. Provider Capability Registry

The initial capability registry includes:

- Binance public market/announcement context
- official project feeds
- Arkham
- Nansen
- LunarCrush
- X API
- Reddit API
- Glassnode
- official macro/regulatory sources

The registry records whether credentials are required and whether historical data can safely be replayed point-in-time.

### 7. Brian's proprietary PIT memory

`IntelligenceStore` writes one immutable content-addressed capture per observation with provider, record type, observed time, capture time, provenance and payload hash.

This prospective store is a strategic data asset. Brian must never manufacture a historical social/on-chain dataset by applying today's labels or knowledge to old timestamps.

## Historical and prospective research rule

Historical intelligence replay is allowed only when the publication/observation timestamp and the information available at that time are independently verifiable. Provider data whose historical labeling semantics are uncertain is `provider_dependent`; social APIs default to `prospective_only`.

If a record cannot meet this condition, it is collected prospectively in shadow mode from the date Brian begins observing it. Hindsight backfill is forbidden.

## Planned expansion after foundation validation

1. Multi-asset Universe Radar across liquid spot/perpetual markets, with new-token/listing and liquidity filters.
2. Source Reputation Memory: empirical precision/latency/manipulation history by source and event type.
3. Entity Graph: wallets, exchanges, funds, deployers, bridges, counterparties and ownership-confidence edges.
4. Smart-Money Consensus: breadth, repeated accumulation, PnL quality, crowding and exit-flow detection.
5. Narrative Graph: map projects, sectors, tokens, people, protocols and macro events into causal narratives.
6. Cross-Asset Propagation: estimate which assets normally react first/second to specific event classes.
7. Derivatives Intelligence: funding, OI, basis, liquidations, options/IV where legitimate point-in-time data exists.
8. New-Token Safety: liquidity, holder concentration, contract/deployer risk, mint/freeze/admin privileges, honeypot/rug indicators where chain-specific verification is available.
9. Adversarial Information Defense: coordinated social campaigns, spoofed announcements, recycled headlines, fake screenshots, mislabeled wallets and wash-like activity.
10. Opportunity Tournament: event-driven specialists compete against chart/ML/RL specialists under validation-only weighting.
11. Capital allocator remains downstream and cannot manufacture edge. No "100% confidence" state exists.
12. Only after long prospective shadow evidence may a separate paper/live-readiness phase be considered.

## Safety / scientific invariants

- SHADOW_RESEARCH_ONLY.
- No authenticated exchange execution.
- No credential material in repository content.
- No automatic model or strategy promotion.
- No historical hindsight backfill.
- No social popularity -> truth shortcut.
- No whale transfer -> buy/sell shortcut.
- No claim of profitability from development/backtest evidence.
- Existing contaminated 2026 final-holdout status remains unchanged for the historical development programme.

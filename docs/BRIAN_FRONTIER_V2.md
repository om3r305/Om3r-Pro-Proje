# Brian Frontier V2 — Market Belief, Behavioral World & Reflexivity

## Mission

Brian Frontier V2 is not a bigger news reader and not an indicator bundle. Its purpose is to turn world events into **point-in-time, price-aware intelligence**:

`WORLD REALITY → HUMAN RESPONSE → MARKET EXPECTATION → POSITIONING → PRICE ABSORPTION → RESIDUAL OPPORTUNITY`

The core objects are deliberately separated:

- **P — Brian belief / nowcast:** what Brian believes about the world using only information available at the decision timestamp.
- **Q — market-implied belief:** what the market already prices through spot/futures/options/prediction markets/positioning and related observable state.
- **X\* — realized state:** what is eventually resolved as reality.
- **H — behavioral state:** fear, uncertainty, FOMO, capitulation, disbelief, euphoria and other crowd/participant responses inferred from behavior rather than prose alone.
- **Z — positioning state:** who is exposed, crowded, forced, hedged or under-positioned.
- **L — liquidity state:** spread, depth, impact, fragility and exit capacity.

The tradable object is **not news polarity**. It is the remaining, after-cost residual between Brian's belief and the market-implied state, conditioned on behavioral response, positioning, liquidity, regime and horizon.

## Non-negotiable boundaries

1. `/dip` is a separate system and is outside Frontier V2 control, data mutation, governance and recommendation paths.
2. Frontier V2 remains **SHADOW ONLY** until a separate, explicit real-money program exists.
3. No exchange order, withdrawal or authenticated live-money surface is created by this program.
4. Canonical ALPHA is not mutated by research/challenger code without evidence gates and human review.
5. Browser/mobile remains controller/monitor only. Runtime continuity belongs to cloud workers.
6. PIT clocks, accounting, CAS, cost fail-closed behavior, promotion control, auth/secrets, CI and migrations are not autonomous self-coding targets.
7. Human psychology may inform an opportunity, but **psychology/sentiment alone may never authorize capital**.
8. “Big event” does not imply “trade.” `DO_NOT_TRADE` is a first-class successful decision.

## F0 — Frontier contracts (this branch starts here)

Before adding new feeds or agents, define a stable language for the system:

- belief snapshots: P / Q / X\*
- behavioral evidence and crowd state
- participant classes and reaction channels
- earliest-knowable timestamps
- horizon distributions and incorporation clocks
- claim provenance and source independence
- point-in-time legality
- explicit `DO_NOT_TRADE` reasons
- SHADOW/canonical/DIP safety envelope

F0 contains no database migration, cron activation, live deployment or canonical ALPHA mutation.

## F1 — Expectation, Surprise & Incorporation Engine

Build the missing market-belief organ.

For every `(event family, entity, instrument, horizon, regime)` maintain append-only PIT snapshots of:

- stated consensus distribution
- market-implied distribution from eligible instruments
- positioned consensus / pain map
- Brian's own nowcast
- event realization when resolved

Derived objects:

- `belief_residual = P - Q`
- realized surprise `X* - Q`
- incorporation probability by asset/venue/horizon
- “who has not reacted?” graph
- remaining after-cost opportunity

A good story with `P ≈ Q` produces no trade.

## F2 — Behavioral World & Reflexivity Engine

Brian should model how humans and institutions react to events, not merely whether text is positive or negative.

### Participant classes

At minimum:

- retail
- discretionary macro / hedge funds
- CTA / trend followers
- volatility-control / risk-parity style mechanical allocators
- dealers / market makers
- ETF/fund flows
- long-only institutions
- corporate treasury / issuers
- governments / regulators / central banks

### Behavioral state

Candidate latent dimensions include:

- fear / panic
- uncertainty
- FOMO / chase
- capitulation
- disbelief
- euphoria
- crowding
- attention concentration
- reflexive amplification
- forced-flow pressure

These states must be inferred from measurable behavior such as flows, options, funding, liquidations, order-book fragility, prediction-market changes, cross-venue reaction and attention. Generic sentiment text alone is insufficient.

### Event families

The engine must be able to reason about broad event classes without hard-coding directional outcomes:

- war / geopolitical escalation / sanctions
- elections and political crises
- NATO / US / China / Russia / EU policy moves
- central-bank and fiscal decisions
- sovereign stress and economic crises
- technology breakthroughs and failures
- major product/launch outcomes
- regulation and antitrust
- energy/supply-chain shocks
- corporate actions and earnings
- natural disasters and infrastructure outages

For each event Brian asks:

1. What is actually known, and from which primary claim?
2. What does the market already price?
3. Which participant classes are reacting and how?
4. Is the reaction rational, mechanical, reflexive or panic-driven?
5. Which linked assets/venues have not absorbed the information yet?
6. What would falsify the thesis?
7. Is there any after-cost opportunity left?

## F3 — Primary Claim, Source Provenance & Information Diffusion

A source is a conditional instrument, not a single trust score.

Track by source × event family × asset × regime:

- accuracy / proper score
- lead time
- originality
- parent/copy probability
- manipulation/retraction risk
- pre-price vs post-price behavior
- specialization
- independent information contribution
- market-impact distribution

World Explorer should prefer **primary claims** over derivative headlines and should create research tickets for coverage holes rather than browsing randomly.

## F4 — Skeptic, Forensics & Autonomous Research Agenda

Every non-trivial SHADOW opportunity passes an adversarial thesis attack.

The Skeptic is measured on:

- bad trades prevented
- valid winners falsely vetoed
- calibration by veto reason

Forensics runs on winners, losers and misses:

- earliest point Brian could legally have known
- source/sensor gap
- expectation/priced-in error
- horizon error
- regime/causal error
- cost/impact error
- gate/veto reason
- exit/replacement error

Unjustified misses and repeated failure causes become ranked research tickets.

## F5 — Regime-Causal World & Portfolio Brain V2

Build regime-conditioned causal operators rather than static causal cartoons. Relationships may invert or vanish as liquidity, inflation, growth, positioning, leverage, volatility and event sensitivity change.

Portfolio Brain V2 reasons in portfolio units:

- marginal correlation/factor exposure
- liquidity and impact
- tail/scenario risk
- concentration
- convexity
- drawdown state
- opportunity cost
- dynamic hedging where it isolates rather than destroys the thesis

Concentration remains possible in SHADOW, but it must be an informed result rather than an absence of portfolio-risk accounting.

## Earliest-Knowable Ledger

Every material event should eventually support an append-only forensic timeline:

- first primary-world timestamp
- first source publication timestamp
- first Brian observation timestamp
- first market-implied Q movement
- first legal actionable timestamp
- first actual Brian decision

This separates “Brian was wrong” from “Brian learned too late” and prevents post-hoc hindsight from training the system to chase unknowable moves.

## Calibration OS

Every probabilistic output should eventually be scored by event family × asset × horizon × regime using proper scoring rules and reliability diagnostics. Stated confidence must be taxed by empirical calibration. A 70% claim that realizes near 55% should be sized like ~55%, not like the prose says 70%.

## Do-Not-Trade Ledger

`DO_NOT_TRADE` is not a residual WAIT state. It is an explicit decision with a reason, e.g.:

- already priced
- uncalibrated in current regime
- insufficient independent evidence
- stale/unknown cost or mark
- incorporation window exhausted
- duplicated portfolio factor
- liquidity/impact consumes edge
- Skeptic veto
- research has positive value-of-information versus immediate betting

Forensics later scores whether the pass was justified.

## Value of Information

Research resources become a second portfolio. Future versions rank research/API/compute tasks by expected decision improvement net of cost. Frontier V2 must not buy “institutional-looking” data without a named chain:

`data → world variable → expectation gap → instrument → horizon → measurable residual`

## What we intentionally do not build first

- generic social sentiment alpha
- multi-agent majority voting
- HFT/queue-position alpha
- generic satellite/data-supermarket ingestion
- end-to-end LLM chart trading
- autonomous live execution
- autonomous mutation of accounting/PIT/promotion/security
- fixed human percentages masquerading as learned risk

## Target architecture

```text
PRIMARY WORLD / CLAIMS / MARKET SENSORS
              ↓
WORLD EXPLORER + SOURCE PROVENANCE
              ↓
P (Brian belief) ↔ Q (market-implied belief) ↔ X* (realization)
              ↓
EXPECTATION + SURPRISE + INCORPORATION
              ↓
BEHAVIORAL WORLD / PARTICIPANTS / REFLEXIVITY
              ↓
REGIME-CONDITIONAL CAUSAL WORLD
              ↓
SPECIALIST CLAIM LEDGER (no majority vote)
              ↓
SKEPTIC / ADVERSARIAL GATE
              ↓
HORIZON ROUTER
              ↓
ALPHA CHALLENGER
              ↓
PORTFOLIO BRAIN / DO-NOT-TRADE
              ↓
FORENSICS → RESEARCH AGENDA → VALUE OF INFORMATION
              ↓
EVOLUTION / REGISTERED EXPERIMENTS / HUMAN PROMOTION
```

## Rollout principle

Frontier V2 grows as a separate long-lived draft program on top of Evolution OS. It does not expand PR #92. Each runtime capability begins as contract + tests, then append-only shadow data, then prospective challenger evidence, then human-reviewed promotion. Activation is always a separate explicit action.
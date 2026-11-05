# BRIAN multi-agent (Claude + Grok) — Phase 0 audit and Phase 1 delivery, 2026-11-05

Scope: operator request "BRIAN 2.0 — Claude + Grok multi-agent intelligence", phases 0 and 1.
Shadow only. No paid API call was made, no key was used, nothing was deployed, no data was changed.

## Phase 0 — what exists today

### Repository and parallel work
- `origin/brian-2026` HEAD `fc74c5b` (daily risk guard, #218). No newer commit since 2026-10-26.
- Open PRs: #185, #177 (research-lab docs), #135, #78, #77, #38, #28, #1 — none touches an LLM
  provider layer or multi-agent orchestration, so this work does not compete with another writer.
- Untracked work of another developer in another worktree (DIP, `brian_ai/`, local Qwen/Ollama) is out
  of scope and was not used or committed.

### Decision chain (live, CORE `qbcjuxhvhwagvqbjyemo`)
- Market scanning: edge functions `brian-multiasset-opportunity-engine` (stocks, ETFs, indices,
  commodities with a futures-roll feed guard, FX) and the canonical crypto ALPHA compiler, both
  writing point-in-time decisions.
- Evidence: hourly `brian_treasury_signal_outcomes` (60-min realised outcome per signal, futures-roll
  rows excluded) → `brian_treasury_segment_edge_cache` (lane × asset class × direction × news trust).
- Execution: none. The shadow treasury (`brian-evolution-treasury`, v19) opens a position only when a
  segment's net edge (gross shrunk by n/(n+100), minus one standard error, minus measured round-trip
  cost) is ≥ 3 bps with n ≥ 100; daily risk guard −2 % / 12 opens. There is no order path to any
  exchange; shadow and real trading are not mixed because real trading does not exist.
- Costs: per asset class from the engines' cost model (equity 18, ETF 12, commodity 14, crypto ~21 bps
  round trip), carried on every outcome row.

### Measured state (2026-11-05 22:06 UTC)
| Item | Value |
|---|---|
| Treasury | 9,997.45 USD, cash, 0 positions; **0 actions in the last 10 days** |
| Segments clearing cost | 0 of 28; best `MULTIASSET_EVENT_REACTION/commodity/short/OFFICIAL` n=159, net −7.1 bps |
| 60-min outcomes, 7 days | multiasset 1,167 signals: gross +1.7, cost 13.0, **net −11.3 bps**; crypto ALPHA 4,157: gross −0.1, cost 21.1, **net −21.1 bps** |
| ORB study (prospective, 180 trades each) | 1R −13.1, close −7.0, in-play −9.6 bps net (evaluation still open) |
| CORE size | 9,061 MB (crossed the 8 GB mark the 2026-10-21 audit projected) |

NO_TRADE classification: all 2,756 treasury snapshots of the last 10 days carry the single blocked reason
"no signal family clears cost + 3 bps on measured 60-min outcomes". No cycle was blocked by the risk
guard, data freshness or failures (0 FAILED treasury runs in 24 h).

The 2026-10-21 audit finding still holds in the current data: the signals have no after-cost edge.
A language model does not change a measured gross move; it can only help **select** among signals.
That is what the comparison in Phase 3 has to show, on later data, before anything else.

### Third-party projects
| Project | Licence | Used how |
|---|---|---|
| anthropic-sdk-python | MIT | Optional dependency for the Claude adapter (lazy import) |
| xai-sdk-python | Apache-2.0 | Optional dependency for the Grok adapter (lazy import; `x_search(to_date=…)`) |
| LangGraph | MIT | **Not added.** The flow is a fixed, bounded pipeline (2 rounds, ≤6 calls); BRIAN already orchestrates with pg_cron + edge functions + collector leases. A graph runtime adds a dependency without a capability we need yet. Revisit if human-approval pauses or resumable multi-step runs are needed. |
| TradingAgents v0.7.1 (Oct 2026) | Apache-2.0 | **No code copied.** Ideas adopted: analysts produce independent reports before any debate; bounded debate rounds; point-in-time discipline on every dated input (their v0.5–v0.7 fixes for filings/statements published after the run date); lessons settle only after the holding period. Not adopted: model-produced trade sizing (BRIAN sizes from measured evidence), memory files as decision input before evaluation. |

X Search pricing: $5 per 1,000 posts fetched plus tokens; its date filter is day-granular, so the data
verifier additionally drops any cited post timestamped after the decision.

## Phase 1 — what was built (`brian2026/multi_agent/`)

| Module | Status |
|---|---|
| `config.py` | Env config; defaults off / shadow only / budget 0. `BRIAN_REAL_TRADING_ENABLED=true` or `BRIAN_MULTI_AGENT_SHADOW_ONLY=false` raise. Rounds ≤2, calls ≤6 clamped. `BRIAN_MULTI_AGENT_STOP` = operator stop. |
| `providers/base.py` | Common guard: worst-case cost reserved before every paid call, refused if it could exceed the run budget; unknown model price → refused; per-candidate call cap; cache; rate limit; errors reduced to a class name (no keys/prompts in records). |
| `providers/claude_provider.py` | `anthropic` SDK, `output_config.format` JSON schema, re-validated; refusal / truncation / bad JSON fail closed. **Tested only against a fake SDK.** |
| `providers/grok_provider.py` | `xai-sdk`, optional X Search with `to_date`; reported `cost_usd` used. **Tested only against a fake SDK.** |
| `schemas.py` | Decision values (BUY/SELL_CANDIDATE, WAIT, REJECT, INSUFFICIENT_DATA, NO_TRADE); record with separate evidence / counter-evidence / sources; `execution_allowed` can only be `False`. |
| `agents.py` | Scanner, verifier, skeptic, risk manager: deterministic. Strategist (Claude) and news intelligence (Grok): model calls. |
| `coordinator.py` | Pipeline + pure `decide`. Risk is checked first: when the measured edge cannot clear cost, **no model is called** (zero spend). |
| `evidence_store.py` | Append-only hash-chained JSONL (local). |
| `evaluation.py` | A/B/C comparison; outcomes learned before the decision are excluded and counted. |
| `dry_run.py` | Offline end-to-end run with mock providers; `--live-canary` runs only with an explicit budget. |

Model defaults: `claude-opus-5-5` ($4/$20 per MTok) and `grok-4.7` ($2/$6). Cheaper drop-ins via env:
`claude-haiku-5-5` ($0.10/$0.50), `grok-4.3` ($1.25/$2.50).

### Cost estimate (not measured)
Per candidate that passes the risk gate: Claude 2 calls ≈ $0.04 (Opus) or < $0.001 (Haiku); Grok
round 1 with X Search ≈ $0.01 tokens + ~$0.10 for ~20 posts, round 2 without search ≈ $0.01. Total
≈ $0.15 typical, ≤ $0.62 reserved worst case. With today's evidence **no** candidate passes the risk
gate, so the live cost would be $0/day; at 20 gated candidates/day ≈ $3/day.

### Tests
`tests/test_multi_agent.py`: 31 tests (config safety, budget/cap/cache/rate-limit, secret-free errors,
both adapters against fake SDKs, verifier/skeptic/risk/scanner, coordinator paths, record immutability,
evidence chain tamper detection, point-in-time evaluation, end-to-end dry run). Full suite locally:
494 passed, 62 skipped.

## Phase 2 — live wiring (prepared, not yet deployed)

| Piece | Status |
|---|---|
| `live.py` | Reads fresh OPEN event-reaction signals, their linked news, the segment evidence and today's treasury state over PostgREST; maps them to candidates (news class mirrors the treasury's rule); one candidate per asset and direction; cross-run de-duplication; inserts with `on_conflict=candidate_id` / ignore duplicates. `--dry-run` writes nothing. Crypto ALPHA is not routed (microstructure signals, no news catalyst). |
| Migration `20261105230000_brian_multi_agent_decisions.sql` | Insert-only table (update/delete rejected by `brian_reject_mutation`), RLS on, client roles revoked, `execution_allowed=false` enforced by a check. **Not applied to CORE.** |
| Workflow `brian-multi-agent-shadow.yml` | Every 15 min, only when repository variable `BRIAN_MULTI_AGENT_RUNNER=on`; manual runs default to dry run. Uses the existing CORE secret; model keys/budget only from repository secrets/variables. **Inactive until merged and switched on.** |
| Risk change | An unobserved live spread no longer blocks (the measured cost already includes the normal spread); an observed spread above 25 bps still blocks. |

Tests: `tests/test_multi_agent_live.py` (7) — mapping, news class, de-duplication, scanner and risk on
mapped rows, insert idempotency, errors without key echo, refusal without configuration.

### Check against real CORE data (read-only, 2026-11-05)
- Last 24 h: 46 OPEN event-reaction signals (20 asset/direction pairs); 6 moved ≥ 100 bps in 1 h and
  pass the scanner; **0 pass the risk gate** (best net −15.4 bps). Model calls that day: 0, cost $0.
- Two real signals (SILVER long 02:03, BRENT short 02:23 UTC) were run through `live.decide_all`
  offline: both NO_TRADE before any model call (net −25.6 and −15.4 bps; the second matches the SQL).
- **Finding: news relevance.** The SILVER and GOLD signals count as `OFFICIAL` news only because the
  engine linked SEC Form 144 filings of unrelated investment funds; BRENT was linked to, among others,
  a video-game delay headline. The segment evidence is therefore split by a news class that is partly
  noise. This is exactly the question the Grok analyst is meant to answer ("is the news about this
  instrument?"), and it can also be improved without any model by tightening the event-to-asset link.

## Not done (honest list)
- No live Claude or Grok call (no approved budget). The adapters are untested against real APIs.
- Phase 2 table and workflow are not live: they need the migration applied, the PR merged and the
  runner variable switched on (operator decisions).
- No dashboard panel (Phase 4; UI files belong to the separate UI owner).
- No A/B/C comparison result yet: needs live records plus later outcomes.

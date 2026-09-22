# Forward shadow scorecard v1

The observation window starts at database installation time, not at a retrospectively selected profitable date. It ends seven days later. Three independent paper portfolios are measured: Treasury, active DIP, and Arena. Existing cash, positions, losses, cooldowns and risk gates are retained. The two enabled paper DIP sessions have their end time extended to this window without resetting them.

The observer runs every five minutes, independently of an open dashboard, reading three current state records. Append-only protocols and observations have RLS and no anonymous or authenticated access. The collector uses SECURITY INVOKER and is callable only by its owner. It does not execute trades, invoke AI providers, or modify risk policy. Reports require the existing dashboard or cron credential, use a read-only repeatable-read transaction, an eight-second statement timeout, bounded rows, and a sixty-second per-isolate cache.

## Meaning of results

- Equity change is latest recorded equity minus the first recorded equity. The source timestamp is retained. This includes open-position valuation and is not realized trade profit. Flat initial portfolios were verified at installation.
- Closed DIP P&L uses the worker's ledger net, without deducting costs again. Treasury closes must match exactly one earlier opening with the same asset, direction and capital. Missing or ambiguous entry/cost stays unknown.
- The baseline is uninvested cash returning zero, excluding interest. Brian, DIP and Arena are different portfolios, not randomized treatment arms. Comparing them does not isolate news or psychology contribution.
- Drawdown uses the running peak of five-minute observations. It can miss intrainterval losses. Missing equity is never zero. Source age over ten minutes for Treasury / three minutes for DIP, observation gaps over eleven minutes, version/session changes and truncation prevent a clean conclusion.
- The 2% review threshold is a reporting flag, not a new liquidation or trading rule. Existing engine risk controls remain in force.
- Seven days and 100 closed trades are minimum review requirements, not significance or profitability guarantees. There is no automatic live promotion. Report truncation also prevents a clean verdict.
- ALPHA follows the first 500 chronological decisions after registration, with pending/overdue outcomes visible. Its one-hour directional return less decision-time round-trip cost is counterfactual, not treasury profit. Mixed compiler versions have no combined mean. Adjacent signals overlap and are not independent samples. The current auditor is scheduled hourly despite its legacy `5m` name.

## Limits and follow-up

This release establishes prospective measurement; it does not establish a profitable strategy. News-versus-no-news and psychology-versus-no-psychology ablations, out-of-sample regime evaluation and exchange fill validation remain separate experiments. Declared policy versions are monitored; an unversioned code change cannot be detected by this report. Protocols cannot be silently rewritten. A strategy or session change invalidates this cohort and requires a new registered experiment.

No real-money execution is enabled. The scorecard endpoint cannot place orders or reset sessions.

## Verification

`node --test tests/shadow-scorecard.test.cjs` covers costs, unknown values, resets, policy changes, source gaps, running-peak drawdown, sample insufficiency and non-shadow records. Existing Frontier, accounting, command, engineering and Scout suites remain release gates. Edge code is checked with Deno; the authenticated production endpoint and scheduled collector are checked separately. Browser authentication may limit inspection of private live data; backend verification does not bypass the dashboard lock.

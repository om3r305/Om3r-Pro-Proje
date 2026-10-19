# ALPHA outcome coverage — production release, 19 October 2026

## Delivered product behavior
The main Brian scorecard now reports the decision denominator and every missing outcome lane at 5/15/60 minutes, including per-asset detail. It also displays the fixed prospective window registered before it began: Berlin 15:00–16:00, final terminal window by 17:02. This is a coverage test, not a profitability experiment. WAIT/VETO remain in the denominator; unavailable costs are never zero.

The read-only query respects observation and persistence/capture times, original +120-second horizon tolerance, mode flags and immutable entry references. Query failures show UNAVAILABLE through a savepoint rather than fabricated complete coverage. Existing trading/scorecard calculations are unchanged. Other-chat engineering queue, controls, scheduler and worker were not modified.

## Exact release and validation
- PR https://github.com/om3r305/Om3r-Pro-Proje/pull/142.
- Tested candidate af36c207b8ad41bffe3b3b084412fbbba261d6df.
- Merge/production e4fd558a162d1f44c68c6f5cce1ba87f0567c5c2; merge Git tree equals candidate.
- CORE brian-shadow-scorecard v5 ACTIVE. Retrieved index.ts, model.ts and coverage.ts match exact local committed source. Existing verify_jwt=false preserved; body authentication retained.
- Frontend dpl_HzW1hY8o9wWVNMdhny1Tm4e7rfnW READY. Production alias verified against that deployment and commit.
- Served frontier-v4.html, outcome-coverage.js and shadow-scorecard.js each HTTP200 and exact source match.
- Existing engineering-room frontend also exactly matched the other conversation's 4727890b316f9e0cf9285c0883754d1f70967bbf source.
- Authenticated live endpoint returned HTTP200, forward-scorecard-v2-coverage and AVAILABLE using the existing server-side cron credential path. No credential returned or stored. Missing-key live POST returned HTTP401 UNAUTHORIZED_DASHBOARD.
- Main CI https://github.com/om3r305/Om3r-Pro-Proje/actions/runs/39249850460: all four jobs passed. Node: 23 files / 138 passing tests. PostgreSQL job: 50 passing tests, including 11 new exact-query scenarios against isolated fixture tables.
- ALPHA CI https://github.com/om3r305/Om3r-Pro-Proje/actions/runs/39249850486: both jobs passed.
- New actual-handler contracts verify authentication, reauthorization before cache access, snapshot time, fixed windows ignoring client dates and savepoint failure isolation.
- At 13:15 UTC, scoped frontend error/fatal log scan over preceding ten minutes returned no matching logs. This is a limited observation window, not a universal clean-log claim.
- Browser visual / user-dashboard-key end-to-end flow remains untested. Actual authenticated backend, HTTP assets and renderer tests are distinct evidence.
- Previous frontend rollback dpl_EZaJSiqF4RC8XYp494bZr2SEhBAg; prior scorecard v4 bundle preserved during deployment. Initial deployment failed on stale absolute import-map path, then succeeded with explicit relative deno.json. No authentication weakening.

## Publication coordination
At 13:13 UTC the main alias pointed to another production build dpl_Buu4qrYfCDYRvDdra3yaM8sRG2wc from older canonical 4727890b316f9e0cf9285c0883754d1f70967bbf, so the new coverage asset was absent there. The tested production deployment itself served the correct asset. Promoting that already-built production deployment restored the current canonical release including both conversations' changes; alias and served bytes were then rechecked. No other conversation's schedules were edited. Concurrent publication remains a coordination risk: a later older-source production build can regress the alias even when Git HEAD is current.

## First prospective observation (not final)
Snapshot 2026-10-19T13:13:18.733+00:00. The fixed cohort contains 32 decisions so far.
- 5 minutes: 24 mature, 0 resolved, 13 terminal-present awaiting auditor, 8 terminal windows open, 3 missing terminal after the original tolerance. Eight decisions not yet mature.
- 15 and 60 minutes: none mature at this snapshot.
- No missing entry or invalid outcome observed in this cohort snapshot.
- No inference of improved/worse profitability or causal attribution from these incomplete counts. Do not compare the rolling latest-300 baseline with this cohort as a paired experiment.

Raw non-secret coverage evidence: BRIAN_OUTCOME_RELEASE_20261019.json.
The API evaluates the fixed cohort on existing scorecard refresh; no new background schedule or automatic final archive was created.

## Product gaps still open
1. Clock-driven outcome price observation independent of radar/action rotation; the new panel diagnoses gaps but does not repair collection.
2. Independent primary/full-text claim verification, event-specific counter-evidence, priced-in assessment and actionable horizons.
3. Prospective psychology/strategy calibration and demonstrated advantage after realistic costs.
4. Complete authenticated visual flow and observation through the end of this fixed cohort.
5. Internal autonomous engineer CODE-timeout behavior belongs to the other conversation; this release does not claim it fixed.

No DIP/Burger/brian-paper changes, real-money activation, strategy thresholds, accounting resets or retrospective price invention.

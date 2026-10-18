# Brian provider capability canary — 19 October 2026 (Berlin)

The autonomous worker is still disabled. A separate operator-owned workflow now permits a single synthetic generation attempt without claiming a production task or changing the worker switch. It triggers only for changes to the canary workflow/scripts on brian-2026, or explicit manual invocation; it has no schedule.

## What this establishes

The existing hosted Copilot credential is tested with pinned CLI 1.0.98 (official GitHub release published 14 October). No new provider, paid subscription, model sweep or local model download is introduced. The requested model is the CLI default; the actual routed model remains UNVERIFIED unless separately evidenced. A generated file and eight independent acceptance tests must succeed to earn CAPABILITY_CANARY_PASSED. A successful response alone is insufficient.

The fixture asks for a deterministic evidence filter: valid point-in-time prices and observed costs, freshness boundaries, deduplication, stable ordering and immutable input. This is a synthetic tooling check, not a production improvement, strategy evaluation, profitability claim or permission to enable repeated claims.

## Boundaries

The model receives a temporary fixture directory, not the repository task queue. Only file-operation tools are exposed; write permission is limited to candidate.cjs, shell and URL tools are denied, built-in MCPs and custom instructions are disabled. The workflow has read-only repository access and existing Copilot request permission; checkout does not persist credentials. It has no database, deployment or exchange credentials.

After generation the scope checker rejects extra files, specification changes, unchanged stubs, symlinks and oversized candidates. Generated code executes only in a disposable Docker container, without network, writable root, credentials, capabilities or privileged user. Acceptance tests are mounted read-only outside the model workspace. Raw provider output is never printed or uploaded; the receipt contains fixed error classes, exit codes, timestamps and the generated file hash. A failure remains a failed workflow, not a healthy-worker label.

The container limits and tool permissions provide separate boundaries. Passing this small test does not establish full repository engineering, replay/stress review or production release capability.

## Local checks

Six behavioral harness tests passed: tool permissions, sanitized error classification, scope changes, symlinks/size limits, known-good implementation versus future-data contamination, and missing-cost/placeholder rejection. The existing baseline is also required. Docker is unavailable locally, so the actual generated-code container run must be evidenced by the hosted workflow. Live results will be recorded after the run; no success is asserted here.

Official references:
- https://github.com/github/copilot-cli/releases/tag/v1.0.98
- https://docs.github.com/en/copilot/reference/copilot-cli-reference/cli-command-reference
- https://docs.github.com/en/copilot/how-tos/copilot-cli/use-copilot-cli/allowing-tools

## Observed hosted result

At 00:31:34 Berlin, the actual hosted run completed CAPABILITY_CANARY_PASSED. One provider attempt wrote candidate SHA-256 `8990c3ab0964ceac65717c1fafc3946bb0b075edee7bf19cd8a9b93b366aa440`; the network-free container passed all eight acceptance tests. Provider and evaluator exit codes were both zero. The actual routed model remains UNVERIFIED. Source: https://github.com/om3r305/Om3r-Pro-Proje/actions/runs/39172553427 . All four regular CI jobs for this commit also passed.

## Exact real-task follow-up

The existing database claim function explicitly permits an exact operator request while autonomous claims are disabled. The workflow's generic automatic preflight previously prevented that intended manual path from being reached. Preflight now distinguishes a valid exact request from automatic queue polling. Server-side OIDC, concurrency, shadow-only eligibility, current-request and approval guards still decide whether a claim is accepted; no database permission or claim rule is relaxed.

The engineering CLI is pinned to the version just tested. The controlled candidate-only mode uses hosted Copilot only, skips the ineffective local-model bootstrap, and stops after independent verification/measurement before approval or release. An operator-owned, unscheduled one-shot workflow verifies the successful canary run/commit and dispatches only request `265eeba91eb4286e20701ffdfd9843e90566eba8f03de2bcfceab75a027aba0b` (measurement coverage). It cannot claim another task if the selected request has become ineligible. Autonomous claims remain false. This is an attempted real repository task, not a claim that the full task succeeded.

Seven canary harness tests and four manual/automatic preflight tests pass locally. The additional canary failure-path test verifies one sanitized receipt, no acceptance execution and temporary-workspace cleanup after a rejected provider call.

The first real task was claimed as run `eb2b6f0f-27d3-46b1-900b-2dd2710da238` and reached PLAN. Analysis completed successfully. Both CODE attempts hit the inherited 120-second timeout (exit 124); no measured candidate or PR was produced. The workflow returned success because the existing runner treats provider deferral as a handled state; the database correctly says BLOCKED. Therefore a green workflow alone is not accepted as engineering success. Evidence: https://github.com/om3r305/Om3r-Pro-Proje/actions/runs/39172925511 .

Follow-up repairs remove the duplicate selected-provider retry in hosted-only mode, explicitly classify timer exits as TIMEOUT, and record output byte counts / changed-file counts without raw model output. The single controlled resume uses the already-enqueued exact request `cac77b561916d0475e3886e1d7ac1e775e035503758edb2fa93291a657a7dcc9`. Its code phase has one 360-second attempt; other phases retain 120 seconds. This bounded adjustment is not an unlimited retry loop. Automatic claims and release remain disabled for this attempt. That release's local baseline was 17 files / 89 passing tests, and all four hosted CI jobs passed.

## Independent evaluator boundary

The guard already protected its scripts, migrations and workflows, but did not protect the fixed baseline test files invoked by those scripts. The guard now rejects autonomous changes to the operator-owned Node baseline and fixed Deno regression suite. Candidate-specific unit/replay/stress tests remain writable, and explicit operator maintenance retains the existing override while preserving other limits. Four behavioral checks exercise these boundaries and verify the actual invoked Deno paths. Guard version is `brian.engineer-guard.v1.1-operator-tests`; the expanded local baseline is 18 files / 93 passing tests.

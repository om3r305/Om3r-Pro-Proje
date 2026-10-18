# Brian engineering test gate — 18 October 2026

## Delivered

The engineering runner previously syntax-checked changed JavaScript, but only executed Deno and Python tests before recording UNIT_REGRESSION. A valid JavaScript file containing a failing assertion could therefore escape the test stage.

`scripts/brian_engineer_node_tests.cjs` now runs the required Brian Node regression baseline plus changed `.test`/`.spec` JavaScript files. Each file must exit successfully and report at least one passing test. Missing baseline files, symlinks, paths outside the repository, timeouts and all-skipped files block the stage. Candidate enumeration uses NUL-separated git output, includes additions/modifications/renames/copies, and excludes deleted paths. No shell interpolation is used for test filenames.

The runner calls this gate before UNIT_REGRESSION. Both the engineering workflow and the regular CI use Node 24. Regular CI now runs this gate on canonical pushes and PRs, independently of whether the autonomous worker is enabled. The script is within the existing operator-owned `scripts/brian_engineer_` protected prefix.

Local validation: **13 files, 65 passing tests**, zero skipped. Five new behavioral checks cover selection, an actual passing test, an actual failing assertion, all-skipped evidence, and missing/symlink/unsafe paths. Bash syntax and patch whitespace checks pass. Deno and database integration suites were not rerun locally because the required runtimes/test database are not available here; hosted CI remains the evidence source for those checks.

## Verified live state and remaining work

At this inspection the autonomous worker remains disabled. Its most recent recorded run was 29 September, blocked at PROTECTED_SCOPE_GUARD. Historical metadata cites exhausted Copilot quota and ineffective qwen2.5-coder:1.5b fallback. Metadata also names qwen3:4b while the workflow actually selects qwen2.5-coder:1.5b. The historical HEALTHY provider label is not a fresh capability test.

No new provider call, generated candidate, PR, release or successful autonomous end-to-end cycle has been observed in this inspection. GitHub's official BYOK documentation requires streaming and tool calling; accepting a model response alone cannot establish engineering capability. A future canary must demonstrate scoped edits, passing independent tests, replay, stress and exact-commit review before enabling repeated claims. No credentials were extracted and no paid provider was provisioned.

Research remains separate from engineering capability: 12 external source records were captured by 22:06 UTC. The independent price bridge had 266 samples with its newest observation at 22:10:47 UTC. These counts establish ingestion, not validated learning or trading profitability.

Open items remain: a capable provider canary and metadata reconciliation; an actual generated and independently verified code improvement; economic event dossiers and counter-evidence; prospectively evaluated psychology features; complete cost/outcome coverage; positive prospective strategy evidence. Full delivery is not established by this change. Real-money execution remains disabled.

Reference: https://docs.github.com/en/enterprise-cloud%40latest/copilot/how-tos/copilot-cli/customize-copilot/use-byok-models

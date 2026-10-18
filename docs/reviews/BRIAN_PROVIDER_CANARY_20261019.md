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

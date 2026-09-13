# Brian Engineer Frontier Console

This additive console exposes the existing PR #100 engineering state-machine to the Frontier V4 dashboard.

It is read-only and does not mutate engineering control, claim tasks, approve releases, merge branches, deploy candidates, touch DIP, or enable live execution.

The UI displays the canonical engineering sequence:

`CLAIMED -> UNDERSTAND -> PLAN -> CODE -> COMPILE -> TEST -> REPLAY -> STRESS -> REVIEW -> PR -> PREVIEW -> MEASURE -> HUMAN_APPROVAL -> DEPLOY -> MONITOR -> COMPLETE`

Production safety remains unchanged:

- `require_human_approval = true`
- `max_concurrent_runs = 1`
- DIP remains isolated
- candidate runs remain `shadow_only = true`
- `live_execution = false`
- autonomous claim is controlled only by the existing engineering control row

The new `brian-frontier-engineering-status` Edge Function uses the existing dashboard-key authentication and only reads engineering control, runs, events and codegen requests.
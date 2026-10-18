# Brian open-world foundation — 18/19 October 2026

Owner authorization: 18 October 23:47 Europe/Berlin. Work crossed local midnight. Product contract: ../BRIAN_VISION_AND_DELIVERY.md. This is the first implemented increment toward the agreed vision, not a completed self-improving investment intelligence system.

## Changes actually made

- Added append-only learning observations and research document receipts, a persistent research mission queue, and a service-only bounded diagnostic RPC. RLS enabled; anon/authenticated have no access. Existing system stop and an explicit open-world research enable flag are required before the new research loop does work.
- Integrated the loop into the existing researcher (24/54 UTC minutes), without introducing another overlapping cron. Learning observations are derived from measured outcome coverage and after-cost direction results; no thresholds are automatically retuned.
- Research mission topics derive from internal hypotheses/measurements and include economic causality, crowd psychology and continual evaluation. Public repository names are unrestricted; the implementation searches GitHub and Crossref, reading at most two README excerpts. This is not arbitrary full-web crawling or full-text literature review. No external package is installed or executed.
- Documents retain URI, capture time, provenance, content hash, available licensing metadata and explicit UNVERIFIED_EXTERNAL / RESEARCH_LEAD_NOT_TRADE_SIGNAL status. Retrieved text remains untrusted task data. Matching excerpts are attached to later hypothesis/codegen context; they cannot authorize trades or backdate knowledge.
- Source failures produce explicit status/receipts and delayed retries. Mission scheduling favors oldest due time before priority, preventing a permanently high-priority question from consuming every cycle.
- Separate operational repair: Realtime already held minute-level intrabar reference prices, but Core's auditor never received them. The authenticated ALPHA export now optionally exports at most 500 recent price observations. Existing Core sync preserves the source event/time in a new append-only, indexed audit price table. The auditor and its pending-decision eligibility query now both consult that path. Horizon definitions, +120 second terminal tolerance, decision-time cost rules and existing result rows are unchanged.

## Live evidence

21:56:38 UTC researcher response: HTTP 200 SUCCESS, 9 hypotheses, 9 experiments, 2 learning observations and 6 retrieved documents. The mission queue held 12 tasks. Two repository README files were received. The observed diagnostic sample had 292 mature decisions / 116 resolved at five minutes and 30 resolved directional outcomes averaging about -20.38 modeled net bps. It is a latest-300, nonrandom, overlapping sample; this is not an unbiased performance claim or causal diagnosis.

Search results included repository documentation and literature metadata. Relevance and research quality have not been independently assessed; some results concern general missing-data methods outside finance. Retrieval is not validation. No model weights were trained.

21:59:31 UTC Core sync: HTTP/logical SUCCESS, new exporter v3, 171 real price observations inserted. Subsequent sync showed 186 points across 51 assets; recent Core asset histories had approximately 9–11 points over 15 minutes. This is sampled evidence, not a tick-complete order book or guaranteed fill path.

The first auditor invocation after adding the price table wrote zero outcomes because the queue eligibility predicate still consulted only older price paths. That missing join was then corrected in all three horizon branches. At 22:02:49 UTC the real invocation returned RESOLVED: **61 additional horizon outcomes and 51 missed-opportunity receipts**; 47 horizon evaluations remained unresolved. These counts are evaluations, not trades or profit. Old missing prices were not fabricated. Full-day coverage has not suddenly become 100%.

Production versions: researcher v10 (v9 initial live proof), sandbox v11, RT ALPHA exporter v3, Core missed-opportunity auditor v10. The v10 researcher patch makes open-world failures explicit in parent collector metadata/status.

Applied migrations:

- 20261018215445_open_world_learning_foundation.sql
- 20261018215833_bridge_independent_audit_price_path.sql
- 20261018220147_include_independent_prices_in_audit_queue.sql

## Verification

60 Node tests passed. New tests cover diagnostic observations without fabricated causality, broad topic generation without leaking raw operational logs, untrusted repository/paper metadata, private/credential URL rejection, provider quotas and bounded responses, stopped-system no-network/no-write behavior, real cycle contract/context propagation, explicit failed-source receipts/retries, exporter authorization/backward compatibility, price-path failure, and unchanged cost/terminal rules. Existing scorecard, page separation, derivatives coverage, scheduler and main pipeline checks also passed. git diff --check passed.

Legacy Python engineering DB tests were not run: pytest is absent and the separate test database URL is not configured. No claim is made that those suites passed. Deno CLI is absent locally; Edge deployments bundled successfully and actual authenticated collector invocations were checked. Existing public SECURITY DEFINER advisor warnings remain; new tables intentionally expose no anon/authenticated policies. SQL checks confirmed anon cannot read research documents or invoke the diagnostic RPC, and all three evidence tables have enabled append-only triggers. No auth grants were broadened.

## Continuing development

The previous paused hourly monitoring task was repurposed, under the new continuous-development authorization, into **Brian araştırma ve geliştirme**, enabled every six hours from 19 October 06:00 Europe/Berlin. It reads the current contract and live evidence, selects at most one justified non-protected change, runs meaningful tests and saves a draft PR. It does not automatically merge/deploy, enable paid providers, change execution/risk/accounting, or restart a stopped system. Duplicate PRs and repeated unchanged alerts are excluded. The service accepted the schedule but returned next_run_time=null; no future execution/delivery is claimed as already tested.

This is ChatGPT-assisted development orchestration. The existing internal engineer worker was paused for provider quota/model-quality failures; autonomous code generation/review/release has not been restored or validated. No current capable model has been demonstrated by this work. Existing codegen requests/candidates remain planned until an actual artifact is produced and passes independent gates.

## What is still missing

General web reading inside a credential-free network-isolated reader; source-grounded economic event dossiers and entity/asset links; evaluated psychology measures; multi-horizon/multiasset expected-edge research; a capable coding provider/local worker; the complete candidate→independent review→prospective comparison→promotion→rollback path; full outcome coverage; measured net advantage. The treasury promotion gate remains closed and no real-money execution was enabled. No new paid service/model was provisioned. Existing infrastructure costs and free-tier limits still apply.

API references checked: https://docs.github.com/en/rest/search/search ; https://www.crossref.org/documentation/retrieve-metadata/rest-api/ ; https://supabase.com/docs/guides/cron/quickstart .

# Brian 2026 Cloud Research

Brian's source code lives in GitHub. GitHub Actions creates a temporary Ubuntu
runner with Python 3.12, so the user's Windows PC may be switched off and the
user does not need to install Linux.

The research workflow downloads only public Binance Spot archives, verifies
their official SHA-256 checksums, normalizes closed candles, applies the data
quality gate, and writes immutable Parquet partitions and manifests.
`research_data/` is ignored Git state: it is reproducible cache/build data, not
source code or scientific provenance. Official Binance archives, checksums, and
content-addressed manifests remain authoritative after every cache restore.

## Run Smoke

1. Open **Actions** in GitHub.
2. Select **Brian 2026 Cloud Research**.
3. Choose **Run workflow** on branch `brian-2026`.
4. Leave mode set to `smoke`.

Smoke verifies and builds BTCUSDT Spot for January 2024 only. It produces the
quality report, Parquet partitions, dataset manifest, and
`cloud_results/brian_cloud_summary.json`. It does not run the Phase 2.5 model
experiment.

## Run Full Development

Use the same manual workflow and select `full-development`. This fixes the
research interval at 2020-01-01 inclusive through 2026-01-01 exclusive,
rebuilds or safely reuses verified public archives and Parquet data, derives
5m/15m/1h, and runs the unchanged Phase 2.5 development experiment. Negative
candidate conclusions are retained; the workflow does not retune thresholds.

GitHub returns the cloud summary, dataset manifest, source-gap evidence,
Phase 2.5 manifest/preregistration, and equity curves as downloadable workflow
artifacts. Raw archive collections and the complete research dataset are not
uploaded as normal review artifacts.

## Permanent safety boundary

Any timestamp at or after `2026-01-01T00:00:00Z` is
`INVALID_CONTAMINATED`. The cloud runner has no end-date input and rejects
datasets or experiments reaching that cutoff. The declaration remains
`NO PRISTINE FINAL HOLDOUT EVALUATED`.

Brian remains `SHADOW_RESEARCH_ONLY`. The workflows have read-only repository
permissions, use no secrets, call no authenticated exchange endpoint, and
contain no live order or strategy-promotion step.

# Brian Disk IO mitigation — 2026-09-12

Supabase warned that `brian-market-intelligence` is depleting its Disk IO Budget. This is an IO-throughput warning, not a database-capacity warning.

Production observations before throttling:

- logical database size: ~712 MB;
- `pg_stat_database.temp_bytes`: ~21 GB accumulated temp spill IO;
- `brian_resolve_sensor_reliability_prospective_calibration`: 192 calls, ~624k shared blocks read, ~4.09M temp blocks read and ~4.10M temp blocks written;
- `brian_compact_retention_run`: 100 normal calls plus maintenance calls, ~2.36M shared blocks read on the normal path;
- high-churn tables include `learning_compact_samples`, `brian_sensor_observations`, reliability/outcome/decision tables, with retention keeping live sizes bounded.

Mitigation keeps the live decision path running. ALPHA compiler (2m), intrabar eye (2m), sensor mesh (10m) and Treasury (1m) remain unchanged. Only non-DIP learning/maintenance jobs are reduced:

- sensor reliability prospective calibration: 2x/hour -> 1x/hour;
- ALPHA reliability feature freeze: 5m -> 15m;
- ALPHA calibration challenger: 2x/hour -> 1x/hour;
- missed-opportunity auditor: 5m -> 15m;
- compact retention: hourly -> every 2h.

DIP is not referenced or changed. Re-evaluate Supabase Disk IO after a 24-hour observation window before making deeper query/index changes.

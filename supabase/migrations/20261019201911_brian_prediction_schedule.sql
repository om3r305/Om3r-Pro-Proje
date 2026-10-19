-- Brian Prediction Intelligence -- Phase 1d: low-frequency schedule, market snapshots only.
--
-- Operator-approved (2026-10-19): turn the module on for public market snapshots, keep wallet
-- tracking OFF, and run one bounded collection cycle every 15 minutes. This is still
-- READ-ONLY + SHADOW-ONLY: nothing here places orders or feeds a live decision.
--
--   * module_enabled := true, wallet_tracking_enabled := false (explicitly re-asserted).
--   * pg_cron job `brian-prediction-eye-15m` at minutes 5,20,35,50 (clear of the busy minutes
--     used by the other active CORE jobs). It calls brian-prediction-eye with the existing
--     vault-held cron key, exactly like the other CORE eyes, and skips a tick while a previous
--     request to the same function is still queued in pg_net.
--   * Stopping: set module_enabled=false (the eye returns MODULE_DISABLED), or the global
--     brian_system_enabled switch, or `select cron.unschedule('brian-prediction-eye-15m')`.
--   * The hard storage cap from Phase 1c (max_storage_mb) still applies to every write.
--
-- On a database without pg_cron (e.g. the CI Postgres) the schedule step is skipped with a notice.

update public.brian_prediction_control
   set module_enabled = true,
       wallet_tracking_enabled = false,
       updated_at = now()
 where control_id = 'prediction-intelligence';

do $$
begin
  if to_regnamespace('cron') is null then
    raise notice 'pg_cron not present; brian-prediction-eye-15m not scheduled';
    return;
  end if;
  perform cron.schedule(
    'brian-prediction-eye-15m',
    '5,20,35,50 * * * *',
    $cmd$
      select net.http_post(
        url := (select decrypted_secret || '/functions/v1/brian-prediction-eye'
                from vault.decrypted_secrets where name = 'brian_project_url' limit 1),
        headers := jsonb_build_object(
          'Content-Type', 'application/json',
          'Authorization', 'Bearer ' || (select decrypted_secret from vault.decrypted_secrets where name = 'brian_anon_jwt' limit 1),
          'apikey', (select decrypted_secret from vault.decrypted_secrets where name = 'brian_anon_jwt' limit 1),
          'x-brian-cron-key', (select decrypted_secret from vault.decrypted_secrets where name = 'brian_cron_key' limit 1)
        ),
        body := '{}'::jsonb,
        timeout_milliseconds := 60000
      )
      where not exists (
        select 1 from net.http_request_queue q
        where q.url like '%/functions/v1/brian-prediction-eye'
      );
    $cmd$
  );
end
$$;

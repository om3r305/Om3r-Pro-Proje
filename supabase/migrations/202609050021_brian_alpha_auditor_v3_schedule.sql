-- Switch ALPHA missed-opportunity audit to bounded v3 worker.
-- SHADOW ONLY. Auth remains JWT + x-brian-cron-key.

do $$
declare
  r record;
begin
  for r in
    select jobid from cron.job
    where jobname in ('brian-missed-opportunity-auditor-5m','brian-missed-opportunity-auditor-v3-5m')
  loop
    perform cron.unschedule(r.jobid);
  end loop;
end
$$;

select cron.schedule(
  'brian-missed-opportunity-auditor-v3-5m',
  '2-59/5 * * * *',
  $cron$
  select net.http_post(
    url := (
      select decrypted_secret || '/functions/v1/brian-missed-opportunity-auditor-v3'
      from vault.decrypted_secrets
      where name='brian_project_url'
      limit 1
    ),
    headers := jsonb_build_object(
      'Content-Type','application/json',
      'Authorization','Bearer ' || (
        select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1
      ),
      'apikey',(
        select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1
      ),
      'x-brian-cron-key',(
        select decrypted_secret from vault.decrypted_secrets where name='brian_dashboard_cron_key' limit 1
      )
    ),
    body := '{}'::jsonb,
    timeout_milliseconds := 45000
  );
  $cron$
);

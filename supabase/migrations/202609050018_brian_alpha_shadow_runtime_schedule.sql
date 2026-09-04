-- Brian ALPHA v2 production SHADOW runtime schedule.
-- No live execution. Phase 3.7 remains frozen/read-only.
-- Calls are protected by both Supabase gateway JWT verification and Brian's hashed cron key.
-- Idempotent by job name so migration-history drift can never multiply scheduled writers.

do $brian$
declare
  v_jobid bigint;
begin
  for v_jobid in
    select jobid
      from cron.job
     where jobname in (
       'brian-alpha-decision-compiler-1m',
       'brian-missed-opportunity-auditor-5m',
       'brian-official-macro-eye-10m'
     )
  loop
    perform cron.unschedule(v_jobid);
  end loop;
end
$brian$;

select cron.schedule(
  'brian-alpha-decision-compiler-1m',
  '* * * * *',
  $$
  select net.http_post(
    url := (
      select decrypted_secret || '/functions/v1/brian-alpha-decision-compiler'
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
  $$
);

select cron.schedule(
  'brian-missed-opportunity-auditor-5m',
  '2-59/5 * * * *',
  $$
  select net.http_post(
    url := (
      select decrypted_secret || '/functions/v1/brian-missed-opportunity-auditor'
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
  $$
);

select cron.schedule(
  'brian-official-macro-eye-10m',
  '6-59/10 * * * *',
  $$
  select net.http_post(
    url := (
      select decrypted_secret || '/functions/v1/brian-official-macro-eye'
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
    timeout_milliseconds := 90000
  );
  $$
);

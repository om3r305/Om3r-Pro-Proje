-- Schedule Source Architecture V2 content observer on odd minutes.
-- Registry health runs on even minutes; this separates health probing from content ingestion.
do $$
declare r record;
begin
  for r in select jobid from cron.job where jobname='brian-source-observer-v2-2m' loop
    perform cron.unschedule(r.jobid);
  end loop;
end $$;

select cron.schedule(
  'brian-source-observer-v2-2m',
  '1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59 * * * *',
  $cron$
  select net.http_post(
    url := (
      select decrypted_secret || '/functions/v1/brian-source-observer-v2'
      from vault.decrypted_secrets where name='brian_project_url' limit 1
    ),
    headers := jsonb_build_object(
      'Content-Type','application/json',
      'Authorization','Bearer ' || (select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),
      'apikey',(select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),
      'x-brian-cron-key',(select decrypted_secret from vault.decrypted_secrets where name='brian_dashboard_cron_key' limit 1)
    ),
    body := '{}'::jsonb,
    timeout_milliseconds := 90000
  );
  $cron$
);

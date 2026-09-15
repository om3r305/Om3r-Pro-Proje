do $$
declare r record;
begin
  for r in select jobid from cron.job where jobname='brian-source-registry-v2-5m' loop
    perform cron.unschedule(r.jobid);
  end loop;
end $$;

select cron.schedule(
  'brian-source-registry-v2-5m',
  '2-59/5 * * * *',
  $cmd$
  select net.http_post(
    url := (select decrypted_secret || '/functions/v1/brian-source-registry-v2' from vault.decrypted_secrets where name='brian_project_url' limit 1),
    headers := jsonb_build_object(
      'Content-Type','application/json',
      'Authorization','Bearer ' || (select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),
      'apikey',(select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),
      'x-brian-cron-key',(select decrypted_secret from vault.decrypted_secrets where name='brian_cron_key' limit 1)
    ),
    body := '{}'::jsonb,
    timeout_milliseconds := 120000
  );
  $cmd$
);
-- Phase127 cloud cadence: independent shadow-only horizon challenger.
do $$
declare j record;
begin
  for j in select jobid from cron.job where jobname='brian-evolution-alpha-horizon-5m'
  loop perform cron.unschedule(j.jobid); end loop;
end $$;

select cron.schedule(
  'brian-evolution-alpha-horizon-5m',
  '3-58/5 * * * *',
  $cron$
  select net.http_post(
    url := (select decrypted_secret || '/functions/v1/brian-evolution-alpha-horizon-challenger'
            from vault.decrypted_secrets where name='brian_project_url' limit 1),
    headers := jsonb_build_object(
      'Content-Type','application/json',
      'Authorization','Bearer ' || (select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),
      'apikey',(select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),
      'x-brian-cron-key',(select decrypted_secret from vault.decrypted_secrets where name='brian_cron_key' limit 1)
    ),
    body := '{}'::jsonb,
    timeout_milliseconds := 55000
  );
  $cron$
);

insert into public.brian_system_job_registry(job_name,component,operator_managed)
values ('brian-evolution-alpha-horizon-5m','EVOLUTION',true)
on conflict (job_name) do update
set component=excluded.component,operator_managed=excluded.operator_managed;

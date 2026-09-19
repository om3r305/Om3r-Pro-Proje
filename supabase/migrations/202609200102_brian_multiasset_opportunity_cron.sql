do $$
declare
  existing_jobid bigint;
begin
  select jobid into existing_jobid
  from cron.job
  where jobname='brian-multiasset-opportunity-10m'
  order by jobid
  limit 1;

  if existing_jobid is null then
    perform cron.schedule(
      'brian-multiasset-opportunity-10m',
      '3,13,23,33,43,53 * * * *',
      $cmd$
      select net.http_post(
        url := (select decrypted_secret || '/functions/v1/brian-multiasset-opportunity-engine'
                from vault.decrypted_secrets where name='brian_project_url' limit 1),
        headers := jsonb_build_object(
          'Content-Type','application/json',
          'Authorization','Bearer ' || (select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),
          'apikey',(select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),
          'x-brian-cron-key',(select decrypted_secret from vault.decrypted_secrets where name='brian_cron_key' limit 1)
        ),
        body := '{}'::jsonb,
        timeout_milliseconds := 30000
      )
      where not exists (
        select 1 from net.http_request_queue q
        where q.url like '%/functions/v1/brian-multiasset-opportunity-engine'
      );
      $cmd$
    );
  else
    perform cron.alter_job(
      existing_jobid,
      schedule := '3,13,23,33,43,53 * * * *',
      active := true
    );
  end if;
end $$;

insert into public.brian_system_job_registry(job_name,component,operator_managed)
values('brian-multiasset-opportunity-10m','ALPHA',true)
on conflict (job_name) do update
set component=excluded.component,operator_managed=true;

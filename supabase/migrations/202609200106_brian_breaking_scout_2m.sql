do $$
declare j bigint;
begin
  select jobid into j from cron.job where jobname='brian-breaking-scout-2m' limit 1;
  if j is null then
    perform cron.schedule(
      'brian-breaking-scout-2m',
      '*/2 * * * *',
      $cmd$
      select net.http_post(
        url := (select decrypted_secret || '/functions/v1/brian-breaking-scout' from vault.decrypted_secrets where name='brian_project_url' limit 1),
        headers := jsonb_build_object(
          'content-type','application/json',
          'x-brian-cron-key',(select decrypted_secret from vault.decrypted_secrets where name='brian_cron_key' limit 1)
        ),
        body := '{}'::jsonb,
        timeout_milliseconds := 30000
      );
      $cmd$
    );
  else
    perform cron.alter_job(j,schedule:='*/2 * * * *',active:=true);
  end if;
end $$;

insert into public.brian_system_job_registry(job_name,component,operator_managed)
values('brian-breaking-scout-2m','WORLD',true)
on conflict(job_name) do update set component='WORLD',operator_managed=true;

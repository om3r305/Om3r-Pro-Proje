-- Restore conditional exit protection without enabling the old full-scan cron.
-- No positions: return immediately. Disabled/expired sessions with positions still get exits.
create or replace function public.brian_dip_position_guardian_dispatch()
returns bigint language plpgsql security definer
set search_path=pg_catalog,public,net,vault
as $function$
declare v_request_id bigint; v_url text; v_anon text; v_key text;
begin
  if not exists (
    select 1 from public.brian_dip_multiasset_state
    where engine_id in ('dip-multiasset-v1','dip-aggressive-arena-v1')
      and coalesce(positions,'{}'::jsonb)<>'{}'::jsonb
  ) then return null; end if;
  perform pg_sleep(15);
  select decrypted_secret into v_url from vault.decrypted_secrets where name='brian_project_url' limit 1;
  select decrypted_secret into v_anon from vault.decrypted_secrets where name='brian_anon_jwt' limit 1;
  select decrypted_secret into v_key from vault.decrypted_secrets where name='brian_dashboard_cron_key' limit 1;
  select net.http_post(
    url:=v_url||'/functions/v1/brian-dip-position-guardian',
    headers:=jsonb_build_object('Content-Type','application/json','Authorization','Bearer '||v_anon,'apikey',v_anon,'x-brian-cron-key',v_key),
    body:='{}'::jsonb,timeout_milliseconds:=58000
  ) into v_request_id;
  return v_request_id;
end;
$function$;
revoke execute on function public.brian_dip_position_guardian_dispatch() from public,anon,authenticated;
grant execute on function public.brian_dip_position_guardian_dispatch() to service_role;
do $schedule$
declare j bigint;
begin
  select jobid into j from cron.job where jobname='brian-dip-position-guardian-1m';
  if j is null then
    perform cron.schedule('brian-dip-position-guardian-1m','* * * * *','select public.brian_dip_position_guardian_dispatch()');
  else
    perform cron.alter_job(j,schedule:='* * * * *',active:=true);
  end if;
end;
$schedule$;

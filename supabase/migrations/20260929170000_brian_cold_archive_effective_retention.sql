-- Target: brian-market-intelligence. Make the cold archive actually bound storage.
-- The archive had produced zero manifests: hot windows were 60-365 days on data
-- at most 27 days old, the largest tables (DIP multiasset evaluations, world
-- snapshots) had no policy, and the purge path deleted as service_role, which
-- the append-only trigger brian_reject_mutation rejects. The DB grew ~80 MB/day.

-- Owner-privileged purge. Re-checks every precondition itself so the edge
-- function cannot widen what is deleted: verified manifest older than the 24h
-- grace, purge-enabled policy, rows past the policy's hot window, and (for
-- sensor observations) rows not referenced by prospective calibration.
create or replace function public.brian_archive_purge_manifest(p_archive_id text)
returns integer language plpgsql security definer set search_path=pg_catalog,public as $$
declare
  m public.brian_archive_manifests;
  p public.brian_archive_policies;
  pk_type text;
  guard text:='';
  n integer;
begin
  select * into m from public.brian_archive_manifests where archive_id=p_archive_id for update;
  if not found or m.state<>'UPLOADED_VERIFIED' or m.verified_at>now()-interval '24 hours' then return 0; end if;
  select * into p from public.brian_archive_policies where table_name=m.table_name;
  if not found or p.purge_enabled is not true then return 0; end if;
  select format_type(a.atttypid,a.atttypmod) into pk_type from pg_attribute a
   where a.attrelid=format('public.%I',p.table_name)::regclass and a.attname=p.pk_column and not a.attisdropped;
  if pk_type is null then raise exception 'archive policy pk % not found on %',p.pk_column,p.table_name; end if;
  if p.table_name='brian_sensor_observations' then
    guard:=' and not exists(select 1 from public.brian_sensor_reliability_prospective_calibration c where c.observation_id=t.observation_id)';
  end if;
  execute format(
    'delete from public.%I t where t.%I = any(array(select jsonb_array_elements_text($1))::%s[]) and t.%I < now()-make_interval(days=>$2)%s',
    p.table_name,p.pk_column,pk_type,p.time_column,guard)
  using m.pk_values,p.hot_retention_days;
  get diagnostics n=row_count;
  update public.brian_archive_manifests
     set state='PURGED',purged_at=now(),updated_at=now(),
         metadata=coalesce(metadata,'{}'::jsonb)||jsonb_build_object('deleted_rows',n,'purged_by','brian_archive_purge_manifest')
   where archive_id=p_archive_id;
  return n;
end $$;
alter function public.brian_archive_purge_manifest(text) owner to postgres;
revoke all on function public.brian_archive_purge_manifest(text) from public,anon,authenticated;
grant execute on function public.brian_archive_purge_manifest(text) to service_role;

-- Hot windows sized to what readers use (edge functions read <=7 days of DIP
-- evaluations; learning jobs read <=36h). Archives stay restorable via
-- brian-cold-archive {"action":"restore","archive_id":...}.
insert into public.brian_archive_policies(table_name,time_column,pk_column,hot_retention_days,batch_size,archive_enabled,purge_enabled,priority,notes,updated_at) values
  ('brian_dip_multiasset_evaluations','observed_at','evaluation_id',14,1000,true,true,31,'largest table; readers use <=7d',now()),
  ('brian_world_event_frames','observed_at','frame_id',14,1000,true,true,32,'world snapshot history',now()),
  ('brian_world_causal_mechanisms','observed_at','mechanism_id',14,300,true,true,33,'world snapshot history',now()),
  ('brian_world_scenario_snapshots','observed_at','scenario_id',14,300,true,true,34,'world snapshot history',now()),
  ('brian_world_narrative_snapshots','observed_at','snapshot_id',14,300,true,true,35,'world snapshot history',now()),
  ('brian_world_asset_impact_candidates','observed_at','impact_id',14,500,true,true,36,'world snapshot history',now()),
  ('brian_world_entity_observations','observed_at','observation_id',14,1000,true,true,37,'world snapshot history',now()),
  ('brian_alpha_phase37_comparisons','observed_at','comparison_id',30,1000,true,true,38,'learning reads <=36h',now()),
  ('brian_alpha_reliability_shadow_features','decision_observed_at','assessment_id',30,500,true,true,39,'learning reads <=36h',now())
on conflict(table_name) do update set time_column=excluded.time_column,pk_column=excluded.pk_column,
  hot_retention_days=excluded.hot_retention_days,batch_size=excluded.batch_size,archive_enabled=true,
  purge_enabled=true,priority=excluded.priority,notes=excluded.notes,updated_at=now();

update public.brian_archive_policies set hot_retention_days=7,purge_enabled=true,updated_at=now() where table_name='brian_collector_runs';
update public.brian_archive_policies set hot_retention_days=7,batch_size=1000,purge_enabled=true,updated_at=now(),notes='no new rows here; live ticks moved to brian-realtime' where table_name='brian_micro_book_ticks';
update public.brian_archive_policies set hot_retention_days=30,batch_size=1000,purge_enabled=true,updated_at=now(),notes='calibration-referenced rows are never purged' where table_name='brian_sensor_observations';

-- Archive continuously instead of 3 archives/day.
select cron.alter_job(job_id:=(select jobid from cron.job where jobname='brian-cold-archive-daily'),schedule:='9,19,29,39,49,59 * * * *');

-- pg_cron run history is operational logging only (135 MB, 144k rows).
select cron.schedule('brian-cron-history-prune-daily','53 3 * * *',
  $$delete from cron.job_run_details where end_time < now() - interval '3 days'$$);

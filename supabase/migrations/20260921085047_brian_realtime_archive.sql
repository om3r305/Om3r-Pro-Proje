-- Target: brian-realtime. Reversible cold storage for append-only high-volume telemetry.
create table if not exists public.brian_realtime_archives (
  archive_id text primary key,
  table_name text not null check(table_name in ('brian_sensor_observations','brian_micro_book_ticks')),
  storage_path text not null unique,
  sha256 text not null,
  row_count integer not null,
  verified_at timestamptz not null,
  last_observed_at timestamptz not null,
  last_pk text not null,
  compacted_at timestamptz,
  restored_at timestamptz
);
alter table public.brian_realtime_archives enable row level security;
revoke all on public.brian_realtime_archives from public,anon,authenticated;
grant all on public.brian_realtime_archives to service_role;
create table if not exists public.brian_realtime_job_leases (
  job text primary key, token uuid not null, expires_at timestamptz not null
);
alter table public.brian_realtime_job_leases enable row level security;
revoke all on public.brian_realtime_job_leases from public,anon,authenticated;
grant all on public.brian_realtime_job_leases to service_role;
create or replace function public.brian_realtime_acquire_lease(p_job text,p_token uuid)
returns boolean language sql security invoker set search_path=pg_catalog,public as $$
 with acquired as (
   insert into public.brian_realtime_job_leases(job,token,expires_at)
   values(p_job,p_token,now()+interval '5 minutes')
   on conflict(job) do update set token=excluded.token,expires_at=excluded.expires_at
   where public.brian_realtime_job_leases.expires_at<now()
   returning job
 ) select exists(select 1 from acquired);
$$;
-- Only compact rows whose exact contents still match the verified archive.
create or replace function public.brian_realtime_compact_archive(p_archive text,p_rows jsonb)
returns integer language plpgsql security invoker set search_path=pg_catalog,public as $$
declare a public.brian_realtime_archives; n integer; pk text;
begin
 select * into a from public.brian_realtime_archives where archive_id=p_archive for update;
 if not found or a.compacted_at is not null or a.verified_at>now()-interval '24 hours' then return 0; end if;
 if jsonb_array_length(p_rows)<>a.row_count then raise exception 'archive row count mismatch'; end if;
 pk:=case a.table_name when 'brian_sensor_observations' then 'observation_id' when 'brian_micro_book_ticks' then 'tick_id' end;
 if pk is null then raise exception 'table not allowed'; end if;
 execute format('delete from public.%I t using jsonb_array_elements($1) r where t.%I=r->>%L and to_jsonb(t)=r and t.observed_at<now()-interval ''7 days''',a.table_name,pk,pk) using p_rows;
 get diagnostics n=row_count;
 update public.brian_realtime_archives set compacted_at=now() where archive_id=p_archive;
 return n;
end $$;
revoke all on function public.brian_realtime_acquire_lease(text,uuid),public.brian_realtime_compact_archive(text,jsonb) from public,anon,authenticated;
grant execute on function public.brian_realtime_acquire_lease(text,uuid),public.brian_realtime_compact_archive(text,jsonb) to service_role;
create index if not exists brian_rt_sensor_archive_idx on public.brian_sensor_observations(observed_at,observation_id);
create index if not exists brian_rt_tick_archive_idx on public.brian_micro_book_ticks(observed_at,tick_id);

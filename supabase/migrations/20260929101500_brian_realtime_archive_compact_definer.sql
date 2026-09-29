-- Target: brian-realtime. Fix archive compaction that never deleted a row.
-- brian_realtime_compact_archive ran as SECURITY INVOKER (service_role), but the
-- append-only trigger brian_reject_mutation only lets current_user='postgres'
-- delete. Every compaction raised, the archive job FAILED_CLOSED on the oldest
-- pending archive, and sensor/micro-book telemetry grew without bound.
-- Run as the owner (postgres) instead and record how many rows each archive
-- actually removed, so a content mismatch is visible rather than silent.
alter table public.brian_realtime_archives add column if not exists compacted_rows integer;

create or replace function public.brian_realtime_compact_archive(p_archive text,p_rows jsonb)
returns integer language plpgsql security definer set search_path=pg_catalog,public as $$
declare a public.brian_realtime_archives; n integer; pk text;
begin
 select * into a from public.brian_realtime_archives where archive_id=p_archive for update;
 if not found or a.compacted_at is not null or a.verified_at>now()-interval '24 hours' then return 0; end if;
 if jsonb_array_length(p_rows)<>a.row_count then raise exception 'archive row count mismatch'; end if;
 pk:=case a.table_name when 'brian_sensor_observations' then 'observation_id' when 'brian_micro_book_ticks' then 'tick_id' end;
 if pk is null then raise exception 'table not allowed'; end if;
 execute format('delete from public.%I t using jsonb_array_elements($1) r where t.%I=r->>%L and to_jsonb(t)=r and t.observed_at<now()-interval ''7 days''',a.table_name,pk,pk) using p_rows;
 get diagnostics n=row_count;
 update public.brian_realtime_archives set compacted_at=now(),compacted_rows=n where archive_id=p_archive;
 return n;
end $$;
alter function public.brian_realtime_compact_archive(text,jsonb) owner to postgres;
revoke all on function public.brian_realtime_compact_archive(text,jsonb) from public,anon,authenticated;
grant execute on function public.brian_realtime_compact_archive(text,jsonb) to service_role;

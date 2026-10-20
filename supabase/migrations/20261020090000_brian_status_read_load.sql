-- Status read load: same results, far less work per UI poll (20 Oct 2026 incident).
--
-- 07:33-07:51 UTC CORE ran out of CPU/IO headroom. The UI status endpoints (engineering status is
-- polled once a minute per open tab, 10 parallel reads) read two DISTINCT ON "latest" views with
-- ORDER BY <time> LIMIT n. That shape forces a full scan + sort of the base table on every call,
-- and an unindexed ORDER BY observed_at on brian_evolution_hypothesis_snapshots does the same.
--
-- 1) The two views are redefined as an index skip-scan (one probe per distinct source_id on the
--    existing (source_id, time desc) indexes). Columns, types, row set and output order (by
--    source_id) are unchanged; security_invoker and grants are unchanged. The assessment view gets
--    assessment_id desc as a deterministic tie-break where DISTINCT ON was arbitrary.
-- 2) An index on brian_evolution_hypothesis_snapshots(observed_at desc) for "newest N".
-- No data is changed or deleted.

create or replace view public.brian_world_source_latest
with (security_invoker = true)
as
with recursive ids as (
  (select c.source_id from public.brian_world_source_candidates c order by c.source_id limit 1)
  union all
  select (select c.source_id from public.brian_world_source_candidates c
          where c.source_id > ids.source_id order by c.source_id limit 1)
  from ids where ids.source_id is not null
)
select l.candidate_id, l.source_id, l.discovered_at, l.canonical_uri, l.provider, l.authority_class, l.access_mode, l.stage
from ids
cross join lateral (
  select c.candidate_id, c.source_id, c.discovered_at, c.canonical_uri, c.provider, c.authority_class, c.access_mode, c.stage
  from public.brian_world_source_candidates c
  where c.source_id = ids.source_id
  order by c.discovered_at desc, c.candidate_id desc
  limit 1
) l
where ids.source_id is not null
order by l.source_id;

create or replace view public.brian_world_source_assessment_latest
with (security_invoker = true)
as
with recursive ids as (
  (select a.source_id from public.brian_world_source_assessments a order by a.source_id limit 1)
  union all
  select (select a.source_id from public.brian_world_source_assessments a
          where a.source_id > ids.source_id order by a.source_id limit 1)
  from ids where ids.source_id is not null
)
select l.source_id, l.assessed_at, l.trust_score, l.eligible_for_research
from ids
cross join lateral (
  select a.source_id, a.assessed_at, a.trust_score, a.eligible_for_research
  from public.brian_world_source_assessments a
  where a.source_id = ids.source_id
  order by a.assessed_at desc, a.assessment_id desc
  limit 1
) l
where ids.source_id is not null
order by l.source_id;

revoke all on public.brian_world_source_latest from anon, authenticated;
revoke all on public.brian_world_source_assessment_latest from anon, authenticated;
grant select on public.brian_world_source_latest to service_role;
grant select on public.brian_world_source_assessment_latest to service_role;

comment on view public.brian_world_source_latest is 'Latest unique world-source candidate per source_id for Brian service-role status APIs (index skip-scan).';
comment on view public.brian_world_source_assessment_latest is 'Latest unique world-source assessment per source_id for Brian service-role status APIs (index skip-scan).';

create index if not exists brian_evolution_hypothesis_snapshots_observed_idx
  on public.brian_evolution_hypothesis_snapshots (observed_at desc);

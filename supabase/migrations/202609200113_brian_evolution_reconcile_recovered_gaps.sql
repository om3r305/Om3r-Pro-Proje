with healthy_caps as (
  select capability_id
  from public.brian_evolution_latest_capabilities
  where health='HEALTHY'
),
latest_gap as (
  select distinct on (g.gap_id)
    g.gap_id,g.capability_id,g.domain,g.severity,g.evidence_refs
  from public.brian_evolution_gap_snapshots g
  join healthy_caps h using(capability_id)
  where g.gap_id like 'runtime:%' or g.gap_id like 'planned:%'
  order by g.gap_id,g.observed_at desc
)
insert into public.brian_evolution_gap_snapshots(
  snapshot_id,gap_id,observed_at,capability_id,domain,severity,reason,suggested_action,
  evidence_refs,metadata,evidence_class,shadow_only,live_execution
)
select
  md5(gap_id||'|resolved-split-truth|'||now()::text),
  gap_id,now(),capability_id,domain,'LOW',
  case
    when gap_id like 'planned:%' then 'Resolved: this planned capability now exists in the canonical capability graph.'
    else 'Resolved: the latest evidence snapshot is HEALTHY.'
  end,
  'No repair action required. Keep monitoring evidence quality and freshness.',
  coalesce(evidence_refs,'{}'::text[]),
  jsonb_build_object(
    'resolved',true,
    'resolved_by','brian.evolution-core.v2-split-truth',
    'architecture','brian-realtime + brian-market-intelligence',
    'direct_alpha_influence',false
  ),
  'PROSPECTIVE_EVOLUTION_SHADOW',true,false
from latest_gap
where severity <> 'LOW';

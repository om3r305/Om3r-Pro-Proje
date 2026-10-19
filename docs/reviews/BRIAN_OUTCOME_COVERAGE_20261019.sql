-- Read-only diagnostics. Latest 300 decisions, nonrandom sample; no strategy performance claim.
with cfg as (select now() as at), d as materialized (
 select decision_id,asset_id,observed_at,observed_reference_price,action,estimated_round_trip_cost_bps
 from public.brian_alpha_decisions,cfg where observed_at between cfg.at-interval '6 hours' and cfg.at
 order by observed_at desc,decision_id limit 300
), e as materialized (
 select d.*,h.seconds, cfg.at,
 o.outcome_id, o.metadata->>'estimated_round_trip_cost_bps' as outcome_cost,
 exists(select 1 from public.brian_alpha_decisions p where p.asset_id=d.asset_id and p.observed_reference_price>0 and p.observed_at between d.observed_at+make_interval(secs=>h.seconds) and least(cfg.at,d.observed_at+make_interval(secs=>h.seconds+120)))
 or exists(select 1 from public.brian_intrabar_reaction_events p where p.asset_id=d.asset_id and p.observed_mid_price>0 and p.observed_at between d.observed_at+make_interval(secs=>h.seconds) and least(cfg.at,d.observed_at+make_interval(secs=>h.seconds+120)))
 or exists(select 1 from public.brian_audit_price_samples p where p.asset_id=d.asset_id and p.observed_mid_price>0 and p.observed_at between d.observed_at+make_interval(secs=>h.seconds) and least(cfg.at,d.observed_at+make_interval(secs=>h.seconds+120))) as terminal_present
 from d cross join (values(300),(900),(3600)) h(seconds) cross join cfg
 left join public.brian_alpha_decision_outcomes o on o.decision_id=d.decision_id and o.horizon_seconds=h.seconds and o.created_at<=cfg.at and o.resolved_at<=cfg.at
), tagged as (
 select *,case when observed_at+make_interval(secs=>seconds)>at then 'NOT_MATURE'
 when outcome_id is not null then 'RESOLVED'
 when observed_reference_price is null then 'MISSING_ENTRY_REFERENCE'
 when terminal_present then 'TERMINAL_PRESENT_AWAITING_AUDITOR'
 when observed_at+make_interval(secs=>seconds+120)>=at then 'TERMINAL_WINDOW_OPEN'
 else 'MISSING_TERMINAL_ALL_CORE_PATHS' end as status from e
), grouped as (
 select asset_id,seconds,count(*) as sampled,count(*) filter(where status<>'NOT_MATURE') as mature,
 count(*) filter(where status='RESOLVED') as resolved,
 count(*) filter(where status='MISSING_ENTRY_REFERENCE') as missing_entry,
 count(*) filter(where status='TERMINAL_PRESENT_AWAITING_AUDITOR') as awaiting_auditor,
 count(*) filter(where status='TERMINAL_WINDOW_OPEN') as window_open,
 count(*) filter(where status='MISSING_TERMINAL_ALL_CORE_PATHS') as missing_terminal,
 count(*) filter(where action in ('OPEN_LONG','OPEN_SHORT') and outcome_id is not null and outcome_cost is null) as resolved_action_missing_cost
 from tagged group by asset_id,seconds
)
select jsonb_build_object('as_of',(select at from cfg),'sample_policy','latest_300_last_6h_nonrandom','rows',(select jsonb_agg(to_jsonb(g) order by seconds,asset_id) from grouped g),
'freshness',jsonb_build_object('price_last',(select max(observed_at) from public.brian_audit_price_samples),'research_last',(select max(captured_at) from public.brian_research_documents),'research_documents',(select count(*) from public.brian_research_documents))) as coverage;

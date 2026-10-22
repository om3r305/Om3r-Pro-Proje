-- One honest "is everything connected?" status (operator 2026-10-23). Read-only, service_role only.
-- Every component reports OK / STALE / OFF / LOCAL_ONLY / SEPARATE plus the timestamp it was judged on;
-- integrated = every component that feeds the treasury is OK. Nothing here trades.

create or replace function public.brian_treasury_integration_status_v1(p_now timestamptz default now())
returns jsonb
language sql
stable
security definer
set search_path = pg_catalog, public
as $fn$
  with
  t as (select observed_at, equity_usd, cash_usd, deployment_pct, treasury_version, gate_version, blocked_reasons
        from public.brian_treasury_shadow_latest order by observed_at desc limit 1),
  cfg as (select config_value from public.brian_evolution_runtime_config where config_key = 'brian_treasury_unified_enabled'),
  cache as (select count(*) segments, max(computed_at) computed_at,
                   count(*) filter (where n >= 100 and gross_bps * n / (n + 100.0) - se_bps - cost_bps >= 3) clearing
            from public.brian_treasury_segment_edge_cache),
  lanes as (select coalesce(jsonb_object_agg(lane, jsonb_build_object('n7d', n, 'last_exit_at', last_at)), '{}'::jsonb) v
            from (select lane, count(*) n, max(exit_observed_at) last_at from public.brian_treasury_signal_outcomes
                  where observed_at >= p_now - interval '7 days' group by lane) x),
  eye as (select max(observed_at) at from public.brian_prediction_source_health where observed_at >= p_now - interval '1 day'),
  pilot as (select state, last_heartbeat_at from public.brian_niulai4_pilot_runs order by started_at desc limit 1),
  audit as (select max(created_at) at from public.brian_missed_opportunity_receipts where created_at >= p_now - interval '7 days'),
  jobs as (select coalesce(jsonb_object_agg(jobname, active), '{}'::jsonb) v from cron.job
           where jobname in ('brian-evolution-treasury-1m', 'brian-treasury-signal-outcomes-hourly', 'brian-prediction-eye-15m',
                             'brian-pm-calibration-shadow-hourly', 'brian-cold-archive-daily')),
  c as (
    select
      jsonb_build_object('status', case when t.observed_at >= p_now - interval '15 minutes' then 'OK' else 'STALE' end,
        'observed_at', t.observed_at, 'equity_usd', t.equity_usd, 'cash_usd', t.cash_usd, 'deployment_pct', t.deployment_pct,
        'treasury_version', t.treasury_version, 'gate_version', t.gate_version, 'blocked_reasons', t.blocked_reasons) treasury,
      jsonb_build_object('status', case when (select config_value from cfg) = 'true' then 'OK' else 'OFF' end,
        'lanes', jsonb_build_array('MULTIASSET_EVENT_REACTION', 'CANONICAL_ALPHA')) unified,
      jsonb_build_object('status', case when cache.computed_at >= p_now - interval '2 hours' then 'OK' else 'STALE' end,
        'computed_at', cache.computed_at, 'segments', cache.segments, 'segments_clearing_cost', cache.clearing,
        'lanes', (select v from lanes)) evidence,
      jsonb_build_object('status', case when eye.at >= p_now - interval '1 hour' then 'OK' else 'STALE' end,
        'observed_at', eye.at, 'role', 'news and prediction-market sensors; news trust class feeds the evidence split') prediction_eye,
      (select public.brian_pm_calibration_status_v1()) || jsonb_build_object('status', 'OK',
        'role', 'pre-registered Polymarket study; joins the treasury only if its pre-registered rule passes') pm_calibration,
      jsonb_build_object('status', case when pilot.last_heartbeat_at >= p_now - interval '30 minutes' then 'OK' else 'STALE' end,
        'state', pilot.state, 'last_heartbeat_at', pilot.last_heartbeat_at, 'host', 'LOCAL_PC',
        'role', 'shadow pilot; never copied, joins only after its own evaluation') niulai4,
      jsonb_build_object('status', 'LOCAL_ONLY', 'host', 'LOCAL_PC',
        'role', 'order-flow collection; gaps while the PC is off, cloud host pending (operator)') ofi,
      jsonb_build_object('status', case when audit.at is not null then 'OK' else 'STALE' end, 'last_receipt_at', audit.at,
        'role', 'hindsight report only; the treasury learns from the point-in-time evidence ledger instead') opportunity_audit,
      (select v from jobs) jobs
    from (select 1) one
    left join t on true left join cache on true left join eye on true left join pilot on true left join audit on true
  )
  select jsonb_build_object(
    'version', 'brian.integration-status.v1',
    'observed_at', p_now,
    'integrated', (treasury->>'status') = 'OK' and (unified->>'status') = 'OK' and (evidence->>'status') = 'OK'
                  and (prediction_eye->>'status') = 'OK' and not exists (select 1 from jsonb_each(jobs) j where j.value <> 'true'::jsonb),
    'feeds_treasury', jsonb_build_array('treasury', 'unified', 'evidence', 'prediction_eye'),
    'components', jsonb_build_object('treasury', treasury, 'unified', unified, 'evidence', evidence, 'prediction_eye', prediction_eye,
      'pm_calibration', pm_calibration, 'niulai4', niulai4, 'ofi', ofi, 'opportunity_audit', opportunity_audit),
    'jobs', jobs,
    'shadow_only', true, 'live_execution', false, 'trade_signal', false)
  from c
$fn$;
revoke all on function public.brian_treasury_integration_status_v1(timestamptz) from public, anon, authenticated;
grant execute on function public.brian_treasury_integration_status_v1(timestamptz) to service_role;

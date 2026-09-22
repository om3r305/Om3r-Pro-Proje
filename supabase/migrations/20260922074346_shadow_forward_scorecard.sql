-- Prospective observation only. Does not reset balances, change strategies or place orders.
create table public.brian_shadow_protocols (
  protocol_id text primary key,
  engine_id text not null,
  starts_at timestamptz not null default now(),
  ends_at timestamptz not null default (now() + interval '7 days'),
  rules jsonb not null,
  check (ends_at > starts_at)
);
create table public.brian_shadow_observations (
  protocol_id text not null references public.brian_shadow_protocols,
  captured_at timestamptz not null default now(),
  source_at timestamptz not null,
  session_id text,
  policy_version text,
  equity numeric,
  realized numeric,
  open_positions integer not null,
  shadow_only boolean not null,
  live_execution boolean not null,
  primary key (protocol_id,captured_at)
);
alter table public.brian_shadow_protocols enable row level security;
alter table public.brian_shadow_observations enable row level security;
revoke all on public.brian_shadow_protocols,public.brian_shadow_observations from public,anon,authenticated;
grant select on public.brian_shadow_protocols,public.brian_shadow_observations to service_role;
create policy shadow_protocol_service_read on public.brian_shadow_protocols for select to service_role using (true);
create policy shadow_observation_service_read on public.brian_shadow_observations for select to service_role using (true);

create function public.brian_shadow_append_only() returns trigger language plpgsql set search_path='' as $$
begin raise exception 'Shadow evaluation records are append only'; end $$;
revoke all on function public.brian_shadow_append_only() from public,anon,authenticated;
create trigger shadow_protocol_immutable before update or delete on public.brian_shadow_protocols for each row execute function public.brian_shadow_append_only();
create trigger shadow_observation_immutable before update or delete on public.brian_shadow_observations for each row execute function public.brian_shadow_append_only();

insert into public.brian_shadow_protocols(protocol_id,engine_id,rules)
select 'forward-20260922-v1-'||engine,engine,
  '{"version":"forward-scorecard-v1","minimum_closed_trades":100,"minimum_days":7,"review_drawdown_limit":0.02,"baseline":"uninvested_cash_zero_return","costs":"ledger_net_includes_simulated_fees_and_fill_model","automatic_promotion":false,"sample_interval_seconds":300,"alpha_horizon_seconds":3600,"inference":"descriptive_only_not_profitability_proof"}'::jsonb
from unnest(array['treasury','dip-multiasset-v1','dip-aggressive-arena-v1']) engine;

-- Keep the existing paper sessions running for the observation window; never reset cash or risk state.
update public.brian_dip_multiasset_state s set run_until=greatest(s.run_until,p.ends_at)
from public.brian_shadow_protocols p
where p.engine_id=s.engine_id and s.enabled and s.shadow_only and not s.live_execution;

create function public.brian_capture_shadow_observations() returns void
language sql security invoker set search_path='' set statement_timeout='8s' as $$
 insert into public.brian_shadow_observations(protocol_id,source_at,session_id,policy_version,equity,realized,open_positions,shadow_only,live_execution)
 select p.protocol_id,s.updated_at,s.source_session_id,s.last_scan->>'policy_version',
 case when jsonb_object_length_safe.n=0 then s.cash else (s.last_scan->>'equity')::numeric end,
 s.realized_pnl,jsonb_object_length_safe.n,s.shadow_only,s.live_execution
 from public.brian_shadow_protocols p join public.brian_dip_multiasset_state s on s.engine_id=p.engine_id
 cross join lateral (select count(*)::integer n from jsonb_object_keys(s.positions)) jsonb_object_length_safe
 where now() between p.starts_at and p.ends_at
 union all
 select p.protocol_id,s.observed_at,s.starting_equity_usd::text,s.treasury_version,
 s.equity_usd,s.realized_pnl_usd,jsonb_array_length(s.positions),s.shadow_only,s.live_execution
 from public.brian_shadow_protocols p cross join lateral
 (select observed_at,starting_equity_usd,treasury_version,equity_usd,realized_pnl_usd,positions,shadow_only,live_execution
 from public.brian_treasury_shadow_snapshots where observed_at<=now() order by observed_at desc limit 1) s
 where p.engine_id='treasury' and now() between p.starts_at and p.ends_at
 on conflict do nothing;
$$;
revoke all on function public.brian_capture_shadow_observations() from public,anon,authenticated,service_role;
select public.brian_capture_shadow_observations();
select cron.schedule('brian-shadow-scorecard-5m','*/5 * * * *','select public.brian_capture_shadow_observations()');

create index if not exists brian_treasury_action_position_idx on public.brian_treasury_shadow_actions(position_id,observed_at);
create index if not exists brian_dip_events_engine_time_idx on public.brian_dip_multiasset_events(engine_id,observed_at);

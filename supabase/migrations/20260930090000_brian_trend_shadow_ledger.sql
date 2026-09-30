-- Target: brian-market-intelligence. Paper ledger for the BTC/ETH trend ensemble
-- (policy trend-ensemble-v1, supabase/functions/_shared/trend_shadow.ts). One row per
-- closed UTC day, written by brian-trend-shadow. Append-only; shadow_only, never live.
create table if not exists public.brian_trend_shadow_ledger (
  day date not null,
  policy_version text not null,
  closes jsonb not null,
  signals jsonb not null,
  held_weights jsonb not null,
  target_weights jsonb not null,
  asset_returns jsonb not null,
  day_return double precision not null,
  turnover double precision not null check (turnover >= 0),
  cost double precision not null check (cost >= 0),
  nav double precision not null check (nav > 0),
  rebalanced boolean not null,
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now(),
  primary key (policy_version, day)
);
alter table public.brian_trend_shadow_ledger enable row level security;
revoke all on public.brian_trend_shadow_ledger from public, anon, authenticated;
grant select, insert on public.brian_trend_shadow_ledger to service_role;
drop trigger if exists brian_trend_shadow_ledger_append_only on public.brian_trend_shadow_ledger;
create trigger brian_trend_shadow_ledger_append_only before update or delete on public.brian_trend_shadow_ledger
  for each row execute function public.brian_reject_mutation();

-- Running performance vs holding BTC and a 50/50 basket over the same days.
create or replace view public.brian_trend_shadow_performance with (security_invoker=true) as
with l as (
  select policy_version, day, nav, day_return, cost, rebalanced, target_weights,
         (asset_returns->>'BTCUSDT')::float8 as btc_ret, (asset_returns->>'ETHUSDT')::float8 as eth_ret
  from public.brian_trend_shadow_ledger
), c as (
  select *,
    exp(sum(ln(1 + btc_ret)) over w) as btc_hold_index,
    exp(sum(ln(1 + (btc_ret + eth_ret) / 2)) over w) as half_half_index,
    max(nav) over w as peak_nav,
    first_value(nav) over w as first_nav,
    row_number() over w as days_live
  from l window w as (partition by policy_version order by day rows between unbounded preceding and current row)
)
select policy_version, day, days_live, nav,
  nav / first_nav - 1 as return_since_start,
  btc_hold_index / first_value(btc_hold_index) over p - 1 as btc_hold_return,
  half_half_index / first_value(half_half_index) over p - 1 as half_half_return,
  nav / peak_nav - 1 as drawdown,
  target_weights, rebalanced, cost
from c window p as (partition by policy_version order by day);
revoke all on public.brian_trend_shadow_performance from anon, authenticated;
grant select on public.brian_trend_shadow_performance to service_role;

-- 00:12 UTC, after the daily candle closes.
select cron.schedule('brian-trend-shadow-daily', '12 0 * * *', $$
  select net.http_post(
    url := (select decrypted_secret || '/functions/v1/brian-trend-shadow' from vault.decrypted_secrets where name='brian_project_url' limit 1),
    headers := jsonb_build_object(
      'Content-Type','application/json',
      'Authorization','Bearer ' || (select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),
      'apikey',(select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),
      'x-brian-cron-key',(select decrypted_secret from vault.decrypted_secrets where name='brian_cron_key' limit 1)
    ),
    body := '{}'::jsonb,
    timeout_milliseconds := 60000
  );
$$);

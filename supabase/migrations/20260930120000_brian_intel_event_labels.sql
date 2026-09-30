-- Target: brian-market-intelligence. Rule-based labels for brian_intel_events
-- (labeler rules-v1). The raw events carry no asset mapping or direction, so no
-- news effect can be measured. This adds a versioned, reproducible label per
-- event: noise flag, theme, crypto assets and a direction *from the crypto-price
-- point of view*, with the lexicon hit counts behind it. Raw events are untouched.
create table if not exists public.brian_intel_event_labels (
  event_id text not null,  -- no FK: labels are derived and must not block cold-archive purges of raw events
  labeler_version text not null,
  observed_at timestamptz not null,
  noise boolean not null,
  theme text not null check (theme in ('CRYPTO','CRYPTO_REG','MACRO','GEO_RISK','MARKET_STRESS','OTHER')),
  assets text[] not null,
  pos_hits smallint not null,
  neg_hits smallint not null,
  direction smallint not null check (direction in (-1,0,1)),
  labeled_at timestamptz not null default now(),
  primary key (event_id, labeler_version)
);
create index if not exists brian_intel_event_labels_theme_time_idx
  on public.brian_intel_event_labels(labeler_version, theme, observed_at) where not noise;
alter table public.brian_intel_event_labels enable row level security;
revoke all on public.brian_intel_event_labels from public, anon, authenticated;
grant select on public.brian_intel_event_labels to service_role;

create or replace function public.brian_intel_label_rules_v1(p_claim text)
returns table(noise boolean, theme text, assets text[], pos_hits smallint, neg_hits smallint, direction smallint)
language plpgsql immutable set search_path = pg_catalog as $$
declare
  c text := lower(coalesce(p_claim, ''));
  p int := 0; n int := 0;
  a text[] := '{}';
begin
  noise := p_claim ~ '^[0-9A-Z/-]{1,12} - .*\(\d{10}\)'                                 -- SEC EDGAR form feed
        or c ~ '^green (forest fire|earthquake|flood|tropical|drought|volcano)'           -- GDACS low-severity alerts
        or c ~ '(tägliche rendite|tenderverfahren|invitation to bid|auction result)';     -- routine debt-agency notices

  theme := case
    when c ~ '\m(bitcoin|btc|crypto|cryptocurrenc\w*|ethereum|ether|stablecoins?|binance|coinbase|blockchain|defi|solana|xrp|tether|usdt|usdc)\M' then
      case when c ~ '\m(sec|cftc|regulat\w*|ban|bans|banned|lawsuit|sues|charges|approv\w*|etf|bill|law|congress|senate|clarity act)\M' then 'CRYPTO_REG' else 'CRYPTO' end
    when c ~ '\m(fed|fomc|powell|rate cuts?|rate hikes?|interest rates?|inflation|cpi|pce|payrolls|jobs report|unemployment|treasur(y|ies)|yields?|ecb|lagarde|boj|central bank|recession|gdp)\M' then 'MACRO'
    when c ~ '\m(war|missiles?|strikes?|bomb\w*|invasion|invade\w*|troops|nuclear|sanctions?|attack\w*|drones?|ceasefire|truce|hostages?|coup)\M' then 'GEO_RISK'
    when c ~ '\m(sell-?off|crash(es|ed)?|plung\w*|tumbl(e|es|ed|ing)|rout|slump\w*|defaults?|defaulted|bankrupt\w*|liquidat\w*|hack(s|ed)?|exploit(s|ed)?|contagion|bank run|collapse\w*)\M' then 'MARKET_STRESS'
    else 'OTHER' end;

  if c ~ '\m(bitcoin|btc)\M' then a := array_append(a, 'BTC'); end if;
  if c ~ '\m(ethereum|ether|eth)\M' then a := array_append(a, 'ETH'); end if;
  if c ~ '\m(solana|sol)\M' then a := array_append(a, 'SOL'); end if;
  if c ~ '\mxrp\M' then a := array_append(a, 'XRP'); end if;
  if cardinality(a) = 0 and theme in ('CRYPTO','CRYPTO_REG') then a := '{CRYPTO}'; end if;
  if cardinality(a) = 0 and theme in ('MACRO','GEO_RISK','MARKET_STRESS') then a := '{RISK}'; end if;
  assets := a;

  if theme in ('CRYPTO','CRYPTO_REG') then
    p := (c ~ '\m(gains?|gained|rall(y|ies|ied)|surg(e|es|ed)|soar\w*|jump\w*|climb\w*|record high|all-time high|inflows?|approv(e|es|ed|al)|adopt\w*|embrac\w*|bullish|rebound\w*)\M')::int
       + (c ~ '\m(etf inflows?|buys?|purchas\w*|accumulat\w*)\M')::int;
    n := (c ~ '\m(slid(e|es)?|slump\w*|fall(s|ing)?|fell|drop(s|ped)?|plung\w*|crash\w*|tumbl(e|es|ed|ing)|sell-?off|outflows?|bearish)\M')::int
       + (c ~ '\m(hack\w*|exploit\w*|ban(s|ned)?|crackdown|lawsuit|sues|sued|charges|charged|fraud|delist\w*|disappoint\w*|liquidat\w*|reject\w*)\M')::int;
  elsif theme = 'MACRO' then
    p := (c ~ '(rate cuts?|cuts? rates|easing|dovish|inflation (eases|eased|slows|slowed|cools|cooled|falls|fell)|cooler[- ]than|better than (hoped|expected)|yields? (dip|dips|dipped|fall|falls|fell|drop|drops|dropped|ease|eases))')::int;
    n := (c ~ '(rate hikes?|hikes? rates|hawkish|tightening|inflation (rises|rose|jumps|jumped|accelerates|heats|hot)|hotter[- ]than|yields? (climb|climbs|climbed|rise|rises|rose|jump|jumps|jumped|surge|surges|soar|soars)|yields? hit|\d+-(year|month) high)')::int;
  elsif theme = 'GEO_RISK' then
    p := (c ~ '\m(ceasefire|truce|peace (deal|talks|plan|agreement)|de-?escalat\w*|release of hostages)\M')::int;
    n := (c ~ '\m(strikes?|missiles?|bomb\w*|invasion|invade\w*|attack\w*|drones?|killed|kills|escalat\w*|nuclear threat|war)\M')::int;
  elsif theme = 'MARKET_STRESS' then
    n := 1;
  end if;
  pos_hits := p; neg_hits := n;
  direction := sign(p - n);
  return next;
end $$;

create or replace function public.brian_label_intel_events_v1(p_limit int default 5000)
returns integer language plpgsql set search_path = pg_catalog, public as $$
declare inserted int;
begin
  insert into public.brian_intel_event_labels(event_id, labeler_version, observed_at, noise, theme, assets, pos_hits, neg_hits, direction)
  select e.event_id, 'rules-v1', e.first_observed_at, r.noise, r.theme, r.assets, r.pos_hits, r.neg_hits, r.direction
  from (select event_id, first_observed_at, claim from public.brian_intel_events ev
        where not exists (select 1 from public.brian_intel_event_labels l where l.event_id = ev.event_id and l.labeler_version = 'rules-v1')
        order by first_observed_at limit greatest(1, least(p_limit, 20000))) e
  cross join lateral public.brian_intel_label_rules_v1(e.claim) r
  on conflict do nothing;
  get diagnostics inserted = row_count;
  return inserted;
end $$;
revoke all on function public.brian_label_intel_events_v1(int) from public, anon, authenticated;
grant execute on function public.brian_label_intel_events_v1(int) to service_role;

select cron.schedule('brian-intel-label-5m', '2-59/5 * * * *',
  $$select set_config('statement_timeout','45000',false); select public.brian_label_intel_events_v1(5000);$$);

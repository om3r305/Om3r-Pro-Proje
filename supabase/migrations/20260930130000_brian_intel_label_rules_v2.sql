-- Target: brian-market-intelligence. Labeler rules-v2 (rules-v1 rows are kept for comparison).
-- Fixes measured on v1 samples:
--  * GEO_RISK fired on "bear attack", "home invasion", "alien invasion movie": now needs a
--    state/conflict actor in the headline.
--  * "Bitcoin slips while yields soar" scored bullish: rally verbs are ignored for crypto when
--    the headline is about yields/oil/dollar, and slips/dips/slides/sinks count as bearish.
--  * "Trump rejects ceasefire" scored de-escalation: rejected/collapsed/failed ceasefires and
--    talks now count as escalation.
create or replace function public.brian_intel_label_rules_v2(p_claim text)
returns table(noise boolean, theme text, assets text[], pos_hits smallint, neg_hits smallint, direction smallint)
language plpgsql immutable set search_path = pg_catalog as $$
declare
  c text := lower(coalesce(p_claim, ''));
  p int := 0; n int := 0;
  a text[] := '{}';
  actor boolean := c ~ '\m(russia\w*|ukrain\w*|kremlin|putin|zelensk\w*|iran\w*|israel\w*|gaza|hamas|hezbollah|houthis?|hormuz|china|chinese|taiwan|beijing|north korea\w*|pyongyang|nato|pentagon|us military|u\.s\. military|syria\w*|lebanon|yemen\w*|red sea|kashmir|venezuela\w*)\M';
  crypto_noun text := '(bitcoin|btc|crypto|ether|ethereum|eth|solana|xrp|token|tokens|altcoins?|stablecoins?)';
  cross_asset boolean := c ~ '\m(yields?|oil|crude|dollar|treasur(y|ies)|bonds?)\M';
begin
  noise := p_claim ~ '^[0-9A-Z/-]{1,12} - .*\(\d{10}\)'
        or c ~ '^green (forest fire|earthquake|flood|tropical|drought|volcano)'
        or c ~ '(tägliche rendite|tenderverfahren|invitation to bid|auction result)';

  theme := case
    when c ~ '\m(bitcoin|btc|crypto|cryptocurrenc\w*|ethereum|ether|stablecoins?|binance|coinbase|blockchain|defi|solana|xrp|tether|usdt|usdc)\M' then
      case when c ~ '\m(sec|cftc|regulat\w*|ban|bans|banned|lawsuit|sues|charges|approv\w*|etf|bill|law|congress|senate|clarity act)\M' then 'CRYPTO_REG' else 'CRYPTO' end
    when c ~ '\m(fed|fomc|powell|rate cuts?|rate hikes?|interest rates?|inflation|cpi|pce|payrolls|jobs report|unemployment|treasur(y|ies)|yields?|ecb|lagarde|boj|central bank|recession|gdp)\M' then 'MACRO'
    when actor and c ~ '\m(war|missiles?|strikes?|bomb\w*|invasion|invade\w*|troops|nuclear|sanctions?|attack\w*|drones?|ceasefire|truce|hostages?|coup|blockade|shelling)\M' then 'GEO_RISK'
    when c ~ '\m(sell-?off|crash(es|ed)?|plung\w*|tumbl(e|es|ed|ing)|rout|slump\w*|defaults?|defaulted|bankrupt\w*|liquidat\w*|contagion|bank run|collapse\w*)\M'
      or c ~ ('\m(hack(s|ed)?|exploit(s|ed)?)\M.*\m' || crypto_noun || '\M|\m' || crypto_noun || '\M.*\m(hack(s|ed)?|exploit(s|ed)?)\M') then 'MARKET_STRESS'
    else 'OTHER' end;

  if c ~ '\m(bitcoin|btc)\M' then a := array_append(a, 'BTC'); end if;
  if c ~ '\m(ethereum|ether|eth)\M' then a := array_append(a, 'ETH'); end if;
  if c ~ '\m(solana|sol)\M' then a := array_append(a, 'SOL'); end if;
  if c ~ '\mxrp\M' then a := array_append(a, 'XRP'); end if;
  if cardinality(a) = 0 and theme in ('CRYPTO','CRYPTO_REG') then a := '{CRYPTO}'; end if;
  if cardinality(a) = 0 and theme in ('MACRO','GEO_RISK','MARKET_STRESS') then a := '{RISK}'; end if;
  assets := a;

  if theme in ('CRYPTO','CRYPTO_REG') then
    if not cross_asset then
      p := (c ~ '\m(gains?|gained|rall(y|ies|ied)|surg(e|es|ed)|soar\w*|jump\w*|climb\w*|record high|all-time high|rebound\w*|reclaim\w*)\M')::int;
    else
      -- rally verbs only count when they sit right after a crypto noun ("bitcoin jumps")
      p := (c ~ ('\m' || crypto_noun || '\M\s+(\w+\s+)?(gains?|gained|rall(y|ies|ied)|surg(e|es|ed)|soar\w*|jump\w*|climb\w*|rebound\w*|reclaim\w*)'))::int;
    end if;
    p := p + (c ~ '\m(inflows?|approv(e|es|ed|al)|adopt\w*|embrac\w*|bullish|buys?|purchas\w*|accumulat\w*)\M')::int;
    n := (c ~ '\m(slid(e|es)?|slips?|slipped|dips?|dipped|sinks?|sank|slump\w*|fall(s|ing)?|fell|drop(s|ped)?|plung\w*|crash\w*|tumbl(e|es|ed|ing)|sell-?off|outflows?|bearish|weigh(s|ed)? on)\M')::int
       + (c ~ '\m(hack\w*|exploit\w*|ban(s|ned)?|crackdown|lawsuit|sues|sued|charges|charged|fraud|delist\w*|disappoint\w*|reject\w*|odds fade)\M')::int;
  elsif theme = 'MACRO' then
    p := (c ~ '(rate cuts?|cuts? rates|easing|dovish|inflation (eases|eased|slows|slowed|cools|cooled|falls|fell)|cooler[- ]than|(better|not as bad) than (hoped|expected)|yields? (dip|dips|dipped|fall|falls|fell|drop|drops|dropped|ease|eases|eased|retreat\w*))')::int;
    n := (c ~ '(rate hikes?|hikes? rates|hawkish|tightening|inflation (rises|rose|jumps|jumped|accelerates|heats|hot)|hotter[- ]than|yields? (climb|climbs|climbed|rise|rises|rose|jump|jumps|jumped|surge|surges|soar|soars|top|tops)|yields? hit|bond (sell-?off|rout))')::int;
  elsif theme = 'GEO_RISK' then
    if c ~ '(reject\w*|collaps\w*|fail\w*|stall\w*|breaks? down|broke down|violat\w*)\s+(\S+\s+){0,4}(ceasefire|truce|peace|talks)|(ceasefire|truce|peace|talks)\s+(\S+\s+){0,4}(reject\w*|collaps\w*|fail\w*|stall\w*|breaks? down|broke down|violat\w*)' then
      n := 1;
    else
      p := (c ~ '\m(ceasefire|truce|peace (deal|agreement)|de-?escalat\w*|hostages? (freed|released)|release of hostages)\M')::int;
    end if;
    n := n + (c ~ '\m(strikes?|missiles?|bomb\w*|invasion|invade\w*|attack\w*|drones?|killed|kills|escalat\w*|blockade|shelling)\M')::int;
  elsif theme = 'MARKET_STRESS' then
    n := 1;
  end if;
  pos_hits := p; neg_hits := n;
  direction := sign(p - n);
  return next;
end $$;

create or replace function public.brian_label_intel_events_v2(p_limit int default 5000)
returns integer language plpgsql set search_path = pg_catalog, public as $$
declare inserted int;
begin
  insert into public.brian_intel_event_labels(event_id, labeler_version, observed_at, noise, theme, assets, pos_hits, neg_hits, direction)
  select e.event_id, 'rules-v2', e.first_observed_at, r.noise, r.theme, r.assets, r.pos_hits, r.neg_hits, r.direction
  from (select event_id, first_observed_at, claim from public.brian_intel_events ev
        where not exists (select 1 from public.brian_intel_event_labels l where l.event_id = ev.event_id and l.labeler_version = 'rules-v2')
        order by first_observed_at limit greatest(1, least(p_limit, 20000))) e
  cross join lateral public.brian_intel_label_rules_v2(e.claim) r
  on conflict do nothing;
  get diagnostics inserted = row_count;
  return inserted;
end $$;
revoke all on function public.brian_label_intel_events_v2(int) from public, anon, authenticated;
grant execute on function public.brian_label_intel_events_v2(int) to service_role;

-- v1 stops labelling new events; its rows stay as the comparison baseline.
select cron.unschedule('brian-intel-label-5m');
select cron.schedule('brian-intel-label-v2-5m', '2-59/5 * * * *',
  $$select set_config('statement_timeout','45000',false); select public.brian_label_intel_events_v2(5000);$$);

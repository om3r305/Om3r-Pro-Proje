-- CORE receipt-time quotes share the immutable audit path; REALTIME history is untouched.
alter table public.brian_audit_price_samples
  drop constraint brian_audit_price_samples_source_project_check,
  add constraint brian_audit_price_samples_source_project_check check(source_project in ('REALTIME','CORE')),
  add column bid_price numeric,
  add column ask_price numeric,
  add column request_started_at timestamptz,
  add column provenance_uri text,
  add column source_metadata jsonb;
alter table public.brian_audit_price_samples add constraint brian_core_outcome_quote_check check(
  source_project<>'CORE' or (
    bid_price is not null and ask_price is not null and bid_price>0 and ask_price>=bid_price
    and bid_price::text not in ('NaN','Infinity','-Infinity') and ask_price::text not in ('NaN','Infinity','-Infinity')
    and observed_mid_price=(bid_price+ask_price)/2 and estimated_round_trip_cost_bps is null
    and request_started_at is not null and request_started_at<=observed_at
    and observed_at-request_started_at<=interval '15 seconds' and observed_at<=captured_at
    and provenance_uri is not null and provenance_uri='https://api.binance.com/api/v3/ticker/bookTicker'
    and source_metadata is not null and jsonb_typeof(source_metadata)='object'
    and source_metadata->>'quote_time_semantics' is not null
    and source_metadata->>'quote_time_semantics'='HTTP_RECEIPT_TIME_EXCHANGE_TIME_UNKNOWN'
    and source_metadata->>'provider_response_sha256' is not null
    and source_metadata->>'provider_response_sha256' ~ '^[0-9a-f]{64}$'
  )
);
alter table public.brian_audit_price_samples enable row level security;

create function public.brian_outcome_watch_assets_v1()
returns table(asset_id text)
language sql stable security invoker set search_path=pg_catalog,public as $fn$
  with cfg as (select statement_timestamp() at)
  select distinct d.asset_id from public.brian_alpha_decisions d,cfg
  where d.observed_at>=cfg.at-interval '62 minutes' and d.observed_at<=cfg.at and d.created_at<=cfg.at
    and d.shadow_only and not d.live_execution and d.asset_id ~ '^crypto:[A-Z0-9]{1,30}USDT$'
    and d.observed_reference_price>0 and d.observed_reference_price::text not in ('NaN','Infinity','-Infinity')
    and exists(select 1 from (values(300),(900),(3600)) h(seconds)
      where d.observed_at+make_interval(secs=>h.seconds+120)>=cfg.at
        and not exists(select 1 from public.brian_alpha_decision_outcomes o
          where o.decision_id=d.decision_id and o.horizon_seconds=h.seconds
          and o.created_at<=cfg.at and o.resolved_at<=cfg.at and o.shadow_only and not o.live_execution
          and o.resolved_at between d.observed_at+make_interval(secs=>h.seconds) and d.observed_at+make_interval(secs=>h.seconds+120)))
  order by d.asset_id limit 501
$fn$;
revoke all on function public.brian_outcome_watch_assets_v1() from public,anon,authenticated;
grant execute on function public.brian_outcome_watch_assets_v1() to service_role;

create function public.brian_record_outcome_prices_v1(p_rows jsonb)
returns jsonb language plpgsql security invoker set search_path=pg_catalog,public as $fn$
declare v_stored int; v_now timestamptz:=clock_timestamp();
begin
  if coalesce((select config_value='true' from public.brian_evolution_runtime_config where config_key='brian_system_enabled'),false) is not true then
    return jsonb_build_object('status','OPERATOR_STOPPED','stored',0);
  end if;
  if p_rows is null or jsonb_typeof(p_rows)<>'array' or jsonb_array_length(p_rows)>500 then
    raise exception 'OUTCOME_PRICE_BATCH_INVALID';
  end if;
  if exists(select 1 from jsonb_to_recordset(p_rows) r(source_event_id text,asset_id text,observed_at timestamptz,request_started_at timestamptz)
    where r.source_event_id is null or r.asset_id is null or r.observed_at is null or r.request_started_at is null
      or r.asset_id !~ '^crypto:[A-Z0-9]{1,30}USDT$'
      or r.source_event_id<>encode(sha256(convert_to('brian.outcome-price-eye.v1|'||r.asset_id||'|'||to_char(r.observed_at at time zone 'UTC','YYYY-MM-DD"T"HH24:MI'),'UTF8')),'hex')
      or r.observed_at<v_now-interval '30 seconds' or r.observed_at>v_now
      or r.request_started_at>r.observed_at or r.observed_at-r.request_started_at>interval '15 seconds'
      or not exists(select 1 from public.brian_alpha_decisions d where d.asset_id=r.asset_id
        and d.observed_at between r.observed_at-interval '62 minutes' and r.observed_at
        and d.created_at<=v_now and d.shadow_only and not d.live_execution and d.observed_reference_price>0
        and d.observed_reference_price::text not in ('NaN','Infinity','-Infinity'))) then
    raise exception 'OUTCOME_PRICE_RECEIPT_INVALID';
  end if;
  insert into public.brian_audit_price_samples(source_event_id,asset_id,observed_at,observed_mid_price,
    estimated_round_trip_cost_bps,source_project,bid_price,ask_price,request_started_at,provenance_uri,source_metadata)
  select source_event_id,asset_id,observed_at,(bid_price+ask_price)/2,null,'CORE',bid_price,ask_price,request_started_at,
    'https://api.binance.com/api/v3/ticker/bookTicker',source_metadata
  from jsonb_to_recordset(p_rows) r(source_event_id text,asset_id text,observed_at timestamptz,
    bid_price numeric,ask_price numeric,request_started_at timestamptz,source_metadata jsonb)
  on conflict(source_event_id) do nothing;
  get diagnostics v_stored=row_count;
  return jsonb_build_object('status','SUCCESS','received',jsonb_array_length(p_rows),'stored',v_stored,'shadow_only',true,'live_execution',false);
end
$fn$;
revoke all on function public.brian_record_outcome_prices_v1(jsonb) from public,anon,authenticated;
grant execute on function public.brian_record_outcome_prices_v1(jsonb) to service_role;

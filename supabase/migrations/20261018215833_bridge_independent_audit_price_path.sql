create table public.brian_audit_price_samples (
 source_event_id text primary key,asset_id text not null,observed_at timestamptz not null,
 observed_mid_price numeric not null check(observed_mid_price>0 and observed_mid_price::text not in ('NaN','Infinity','-Infinity')),
 estimated_round_trip_cost_bps double precision,source_project text not null check(source_project='REALTIME'),
 captured_at timestamptz not null default clock_timestamp(),
 shadow_only boolean not null default true check(shadow_only),live_execution boolean not null default false check(not live_execution)
);
create index brian_audit_price_samples_path_idx on public.brian_audit_price_samples(asset_id,observed_at);
alter table public.brian_audit_price_samples enable row level security;
revoke all on public.brian_audit_price_samples from public,anon,authenticated;
grant select,insert on public.brian_audit_price_samples to service_role;
create trigger brian_audit_price_samples_append_only before update or delete on public.brian_audit_price_samples
for each row execute function public.brian_reject_mutation();
CREATE OR REPLACE FUNCTION brian_private.sync_realtime_alpha_v1()
 RETURNS jsonb
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'pg_catalog', 'public', 'brian_private', 'vault', 'extensions'
AS $function$
declare
  v_key text;
  v_resp extensions.http_response;
  v_payload jsonb;
  v_rows jsonb;
  v_inserted integer := 0;
  v_received integer := 0;
  v_prices_inserted integer := 0;
  v_price_path jsonb;
  v_now timestamptz := clock_timestamp();
begin
  if not pg_try_advisory_xact_lock(hashtextextended('brian-realtime-alpha-sync-v1',0)) then
    return jsonb_build_object('status','BUSY','observed_at',v_now);
  end if;

  select decrypted_secret
  into v_key
  from vault.decrypted_secrets
  where name='brian_cron_key'
  limit 1;

  if v_key is null or length(v_key) < 20 then
    raise exception 'BRIAN_CORE_CRON_KEY_MISSING';
  end if;

  perform extensions.http_set_curlopt('CURLOPT_CONNECTTIMEOUT_MS','5000');
  perform extensions.http_set_curlopt('CURLOPT_TIMEOUT_MS','20000');

  select *
  into v_resp
  from extensions.http((
    'POST'::extensions.http_method,
    'https://dliediwlldojkfjzlznm.supabase.co/functions/v1/brian-realtime-alpha-export'::varchar,
    array[
      extensions.http_header('Content-Type','application/json'),
      extensions.http_header('x-brian-internal-key',v_key)
    ],
    'application/json'::varchar,
    jsonb_build_object('since_minutes',20,'limit',500,'include_price_path',true)::text::varchar
  )::extensions.http_request);

  perform extensions.http_reset_curlopt();

  if v_resp.status < 200 or v_resp.status >= 300 then
    return jsonb_build_object(
      'status','UPSTREAM_HTTP_FAILED',
      'http_status',v_resp.status,
      'observed_at',clock_timestamp()
    );
  end if;

  begin
    v_payload := coalesce(v_resp.content,'{}')::jsonb;
  exception when others then
    return jsonb_build_object(
      'status','UPSTREAM_JSON_INVALID',
      'http_status',v_resp.status,
      'observed_at',clock_timestamp()
    );
  end;

  if coalesce(v_payload->>'status','') <> 'SUCCESS' then
    return jsonb_build_object(
      'status','UPSTREAM_LOGICAL_FAILED',
      'upstream_status',v_payload->>'status',
      'observed_at',clock_timestamp()
    );
  end if;

  v_rows := coalesce(v_payload->'rows','[]'::jsonb);
  if jsonb_typeof(v_rows) <> 'array' then
    return jsonb_build_object('status','UPSTREAM_ROWS_INVALID','observed_at',clock_timestamp());
  end if;

  v_received := jsonb_array_length(v_rows);

  with src as (
    select *
    from jsonb_to_recordset(v_rows) as x(
      decision_id text,
      compiler_version text,
      observed_at timestamptz,
      asset_id text,
      observed_reference_price numeric,
      action text,
      direction smallint,
      evidence_score double precision,
      independent_group_count integer,
      support_groups text[],
      conflict_groups text[],
      source_observation_ids text[],
      source_intrabar_event_ids text[],
      requested_virtual_notional_usd numeric,
      gross_edge_bps double precision,
      estimated_round_trip_cost_bps double precision,
      net_edge_bps double precision,
      veto_reason text,
      reason text,
      metadata jsonb,
      evidence_class text,
      shadow_only boolean,
      live_execution boolean,
      created_at timestamptz
    )
  )
  insert into public.brian_alpha_decisions(
    decision_id, compiler_version, observed_at, asset_id, observed_reference_price,
    action, direction, evidence_score, independent_group_count, support_groups,
    conflict_groups, source_observation_ids, source_intrabar_event_ids,
    source_cost_quote_id, requested_virtual_notional_usd, gross_edge_bps,
    estimated_round_trip_cost_bps, net_edge_bps, veto_reason, reason, metadata,
    evidence_class, shadow_only, live_execution, created_at
  )
  select
    decision_id, compiler_version, observed_at, asset_id, observed_reference_price,
    action, direction, evidence_score, independent_group_count, support_groups,
    conflict_groups, source_observation_ids, source_intrabar_event_ids,
    null::text, requested_virtual_notional_usd, gross_edge_bps,
    estimated_round_trip_cost_bps, net_edge_bps, veto_reason, reason,
    coalesce(metadata,'{}'::jsonb) || jsonb_build_object(
      'core_sync_source','brian-realtime',
      'core_synced_at',clock_timestamp()
    ),
    evidence_class, shadow_only, live_execution,
    coalesce(created_at,observed_at,clock_timestamp())
  from src
  where decision_id is not null
  on conflict (decision_id) do nothing;

  get diagnostics v_inserted = row_count;

  v_price_path := coalesce(v_payload->'price_path','[]'::jsonb);
  if jsonb_typeof(v_price_path)<>'array' or jsonb_array_length(v_price_path)>500 then
    raise exception 'UPSTREAM_PRICE_PATH_INVALID';
  end if;
  insert into public.brian_audit_price_samples(source_event_id,asset_id,observed_at,observed_mid_price,estimated_round_trip_cost_bps,source_project)
  select event_id,asset_id,observed_at,observed_mid_price,estimated_round_trip_cost_bps,'REALTIME'
  from jsonb_to_recordset(v_price_path) as p(event_id text,asset_id text,observed_at timestamptz,
    observed_mid_price numeric,estimated_round_trip_cost_bps double precision,shadow_only boolean,live_execution boolean)
  where event_id is not null and asset_id ~ '^crypto:[A-Z0-9]+USDT$'
    and observed_mid_price>0 and observed_mid_price::text not in ('NaN','Infinity','-Infinity')
    and observed_at>=v_now-interval '25 minutes' and observed_at<=clock_timestamp()
    and shadow_only is true and live_execution is false
    and (estimated_round_trip_cost_bps is null or (estimated_round_trip_cost_bps>=0 and estimated_round_trip_cost_bps<'Infinity'::float8))
  on conflict(source_event_id) do nothing;
  get diagnostics v_prices_inserted = row_count;


  return jsonb_build_object(
    'status','SUCCESS',
    'upstream_version',v_payload->>'version',
    'received',v_received,
    'inserted',v_inserted,
    'price_path_received',jsonb_array_length(v_price_path),
    'price_samples_inserted',v_prices_inserted,
    'observed_at',clock_timestamp()
  );
exception when others then
  perform extensions.http_reset_curlopt();
  return jsonb_build_object(
    'status','FAILED_CLOSED',
    'error',sqlerrm,
    'observed_at',clock_timestamp()
  );
end;
$function$
;

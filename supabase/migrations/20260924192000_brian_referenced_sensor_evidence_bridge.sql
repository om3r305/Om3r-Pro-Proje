-- Brian Phase123: referenced sensor evidence bridge for split Realtime/Core topology.
--
-- Target: brian-market-intelligence.
-- SHADOW ONLY. No order, capital, execution, promotion, or mutable reliability
-- weighting surface is introduced.
--
-- Why this exists:
--   brian-realtime is now the authoritative high-cadence sensor project, while
--   brian-market-intelligence owns durable ALPHA outcome/reliability memory.
--   ALPHA decisions already sync from Realtime into Core, but their referenced
--   sensor observations stopped following them. That left Core's local
--   brian_sensor_observations stale, which in turn prevented the hourly
--   reliability snapshot and frozen reliability-feature joins from advancing.
--
-- Contract:
--   * sync ONLY sensor observation ids referenced by recent Core ALPHA decisions;
--   * request at most 500 exact ids from the dedicated Realtime export;
--   * accept only prospective shadow-only, non-live rows;
--   * validate every returned id is one of the requested ids;
--   * append with ON CONFLICT DO NOTHING; never rewrite a sensor observation;
--   * run every two minutes under an advisory lock so backlog recovery is bounded;
--   * keep a 36h rolling debt window, sufficient for the 24h reliability window
--     plus outcome/auditor lag.

create or replace function brian_private.sync_realtime_referenced_sensor_evidence_v1(
  p_limit integer default 500,
  p_lookback interval default interval '36 hours'
)
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public, brian_private, vault, extensions
as $$
declare
  v_started timestamptz := clock_timestamp();
  v_finished timestamptz;
  v_limit integer := greatest(1, least(coalesce(p_limit, 500), 500));
  v_ids text[];
  v_key text;
  v_resp extensions.http_response;
  v_payload jsonb;
  v_rows jsonb;
  v_returned integer := 0;
  v_inserted integer := 0;
  v_remaining_selected integer := 0;
  v_run_id text;
  v_error text;
begin
  if p_lookback < interval '1 hour' or p_lookback > interval '72 hours' then
    raise exception 'BRIAN_REFERENCED_SENSOR_SYNC_LOOKBACK_OUT_OF_RANGE';
  end if;

  if not pg_try_advisory_xact_lock(
    hashtextextended('brian-referenced-sensor-evidence-sync-v1', 0)
  ) then
    return jsonb_build_object(
      'status','BUSY',
      'version','brian.referenced-sensor-evidence-sync.v1',
      'shadow_only',true,
      'live_execution',false,
      'observed_at',clock_timestamp()
    );
  end if;

  select coalesce(array_agg(q.observation_id order by q.latest_ref desc, q.observation_id), '{}'::text[])
  into v_ids
  from (
    select
      src.observation_id,
      max(d.observed_at) as latest_ref
    from public.brian_alpha_decisions d
    cross join lateral unnest(coalesce(d.source_observation_ids, '{}'::text[]))
      as src(observation_id)
    left join public.brian_sensor_observations s
      on s.observation_id = src.observation_id
    where d.observed_at >= clock_timestamp() - p_lookback
      and d.observed_at <= clock_timestamp()
      and d.evidence_class = 'PROSPECTIVE_DEVELOPMENT_SHADOW'
      and d.shadow_only = true
      and d.live_execution = false
      and nullif(btrim(src.observation_id), '') is not null
      and s.observation_id is null
    group by src.observation_id
    order by max(d.observed_at) desc, src.observation_id
    limit v_limit
  ) q;

  if coalesce(array_length(v_ids, 1), 0) = 0 then
    return jsonb_build_object(
      'status','IDLE',
      'version','brian.referenced-sensor-evidence-sync.v1',
      'requested',0,
      'returned',0,
      'inserted',0,
      'remaining_selected',0,
      'shadow_only',true,
      'live_execution',false,
      'observed_at',clock_timestamp()
    );
  end if;

  select decrypted_secret
  into v_key
  from vault.decrypted_secrets
  where name = 'brian_cron_key'
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
    'https://dliediwlldojkfjzlznm.supabase.co/functions/v1/brian-realtime-referenced-sensor-export'::varchar,
    array[
      extensions.http_header('Content-Type','application/json'),
      extensions.http_header('x-brian-internal-key',v_key)
    ],
    'application/json'::varchar,
    jsonb_build_object('observation_ids', to_jsonb(v_ids))::text::varchar
  )::extensions.http_request);

  perform extensions.http_reset_curlopt();

  if v_resp.status < 200 or v_resp.status >= 300 then
    raise exception 'BRIAN_REFERENCED_SENSOR_EXPORT_HTTP_%', v_resp.status;
  end if;

  begin
    v_payload := coalesce(v_resp.content, '{}')::jsonb;
  exception when others then
    raise exception 'BRIAN_REFERENCED_SENSOR_EXPORT_JSON_INVALID';
  end;

  if coalesce(v_payload->>'status','') <> 'SUCCESS' then
    raise exception 'BRIAN_REFERENCED_SENSOR_EXPORT_LOGICAL_%',
      coalesce(v_payload->>'status','UNKNOWN');
  end if;

  v_rows := coalesce(v_payload->'rows', '[]'::jsonb);
  if jsonb_typeof(v_rows) <> 'array' then
    raise exception 'BRIAN_REFERENCED_SENSOR_EXPORT_ROWS_INVALID';
  end if;

  v_returned := jsonb_array_length(v_rows);
  if v_returned > coalesce(array_length(v_ids, 1), 0) then
    raise exception 'BRIAN_REFERENCED_SENSOR_EXPORT_OVERSUPPLIED';
  end if;

  if exists (
    select 1
    from jsonb_array_elements(v_rows) row_value
    where nullif(row_value->>'observation_id','') is null
       or not ((row_value->>'observation_id') = any(v_ids))
  ) then
    raise exception 'BRIAN_REFERENCED_SENSOR_EXPORT_ID_MISMATCH';
  end if;

  with src as (
    select *
    from jsonb_to_recordset(v_rows) as x(
      observation_id text,
      eye_id text,
      template_id text,
      asset_id text,
      market_domain text,
      sensor_family text,
      horizon text,
      independent_group text,
      observed_at timestamptz,
      direction smallint,
      strength double precision,
      confidence double precision,
      reliability double precision,
      available boolean,
      source_ids text[],
      reason text,
      evidence_class text,
      shadow_only boolean,
      live_execution boolean,
      metadata jsonb,
      created_at timestamptz
    )
  )
  insert into public.brian_sensor_observations(
    observation_id,
    eye_id,
    template_id,
    asset_id,
    market_domain,
    sensor_family,
    horizon,
    independent_group,
    observed_at,
    direction,
    strength,
    confidence,
    reliability,
    available,
    source_ids,
    reason,
    evidence_class,
    shadow_only,
    live_execution,
    metadata,
    created_at
  )
  select
    observation_id,
    eye_id,
    template_id,
    asset_id,
    market_domain,
    sensor_family,
    horizon,
    independent_group,
    observed_at,
    direction,
    strength,
    confidence,
    reliability,
    available,
    coalesce(source_ids, '{}'::text[]),
    reason,
    evidence_class,
    shadow_only,
    live_execution,
    coalesce(metadata, '{}'::jsonb) || jsonb_build_object(
      'core_referenced_sensor_sync', true,
      'core_synced_at', clock_timestamp(),
      'core_sync_source', 'brian-realtime',
      'core_sync_version', 'brian.referenced-sensor-evidence-sync.v1'
    ),
    coalesce(created_at, observed_at, clock_timestamp())
  from src
  where observation_id = any(v_ids)
    and evidence_class = 'PROSPECTIVE_DEVELOPMENT_SHADOW'
    and shadow_only is true
    and live_execution is false
    and observed_at <= clock_timestamp()
  on conflict (observation_id) do nothing;

  get diagnostics v_inserted = row_count;

  select count(*)
  into v_remaining_selected
  from unnest(v_ids) wanted(observation_id)
  left join public.brian_sensor_observations s
    on s.observation_id = wanted.observation_id
  where s.observation_id is null;

  v_finished := clock_timestamp();
  v_run_id := md5(
    'brian.referenced-sensor-evidence-sync.v1|' ||
    v_started::text || '|' || v_finished::text || '|' ||
    coalesce(array_length(v_ids, 1), 0)::text || '|' ||
    v_inserted::text
  );

  insert into public.brian_collector_runs(
    run_id,
    collector_id,
    started_at,
    finished_at,
    status,
    observed_records,
    stored_records,
    degraded_sources,
    error_class,
    error_message,
    metadata,
    evidence_class,
    shadow_only,
    live_execution
  )
  values(
    v_run_id,
    'brian-referenced-sensor-evidence-sync-v1',
    v_started,
    v_finished,
    'SUCCESS',
    coalesce(array_length(v_ids, 1), 0),
    v_inserted,
    '{}'::text[],
    null,
    null,
    jsonb_build_object(
      'version','brian.referenced-sensor-evidence-sync.v1',
      'target_project','brian-market-intelligence',
      'source_project','brian-realtime',
      'transport','exact_referenced_sensor_ids_over_internal_http',
      'requested',coalesce(array_length(v_ids, 1),0),
      'returned',v_returned,
      'inserted',v_inserted,
      'remaining_selected',v_remaining_selected,
      'lookback_seconds',extract(epoch from p_lookback)::bigint
    ),
    'PROSPECTIVE_DEVELOPMENT_SHADOW',
    true,
    false
  )
  on conflict (run_id) do nothing;

  return jsonb_build_object(
    'status','SUCCESS',
    'version','brian.referenced-sensor-evidence-sync.v1',
    'requested',coalesce(array_length(v_ids, 1),0),
    'returned',v_returned,
    'inserted',v_inserted,
    'remaining_selected',v_remaining_selected,
    'shadow_only',true,
    'live_execution',false,
    'observed_at',v_finished
  );
exception when others then
  perform extensions.http_reset_curlopt();
  v_error := left(sqlerrm, 1200);
  v_finished := clock_timestamp();
  v_run_id := md5(
    'brian.referenced-sensor-evidence-sync.v1|FAILED|' ||
    v_started::text || '|' || v_finished::text || '|' || v_error
  );
  begin
    insert into public.brian_collector_runs(
      run_id,
      collector_id,
      started_at,
      finished_at,
      status,
      observed_records,
      stored_records,
      degraded_sources,
      error_class,
      error_message,
      metadata,
      evidence_class,
      shadow_only,
      live_execution
    )
    values(
      v_run_id,
      'brian-referenced-sensor-evidence-sync-v1',
      v_started,
      v_finished,
      'FAILED',
      coalesce(array_length(v_ids, 1),0),
      0,
      '{}'::text[],
      'REFERENCED_SENSOR_SYNC_ERROR',
      v_error,
      jsonb_build_object(
        'version','brian.referenced-sensor-evidence-sync.v1',
        'source_project','brian-realtime',
        'target_project','brian-market-intelligence'
      ),
      'PROSPECTIVE_DEVELOPMENT_SHADOW',
      true,
      false
    )
    on conflict (run_id) do nothing;
  exception when others then
    null;
  end;

  return jsonb_build_object(
    'status','FAILED_CLOSED',
    'version','brian.referenced-sensor-evidence-sync.v1',
    'error',v_error,
    'shadow_only',true,
    'live_execution',false,
    'observed_at',v_finished
  );
end;
$$;

revoke all on function brian_private.sync_realtime_referenced_sensor_evidence_v1(
  integer, interval
) from public;
revoke all on function brian_private.sync_realtime_referenced_sensor_evidence_v1(
  integer, interval
) from anon;
revoke all on function brian_private.sync_realtime_referenced_sensor_evidence_v1(
  integer, interval
) from authenticated;
grant execute on function brian_private.sync_realtime_referenced_sensor_evidence_v1(
  integer, interval
) to service_role;

do $$
declare
  r record;
begin
  for r in
    select jobid
    from cron.job
    where jobname = 'brian-referenced-sensor-evidence-sync-2m'
  loop
    perform cron.unschedule(r.jobid);
  end loop;
end
$$;

select cron.schedule(
  'brian-referenced-sensor-evidence-sync-2m',
  '1-59/2 * * * *',
  $cron$
    select brian_private.sync_realtime_referenced_sensor_evidence_v1(
      500,
      interval '36 hours'
    );
  $cron$
);


-- Brian integrated event pipeline v1
-- Restores backend-owned meeting persistence and low-contention recovery for
-- source verification + BIG_MOVE. All market actions remain SHADOW-only.

create table if not exists brian_private.aux_dispatch_state (
  service_id text primary key,
  last_attempt_at timestamptz,
  last_request_id bigint,
  last_http_status integer,
  last_response_at timestamptz,
  attempt_count bigint not null default 0,
  updated_at timestamptz not null default now()
);
revoke all on table brian_private.aux_dispatch_state from public, anon, authenticated;

create or replace function brian_private.dispatch_stale_aux()
returns jsonb
language plpgsql
security definer
set search_path to 'pg_catalog','public','brian_private','net','vault','extensions'
as $$
declare
  now_ts timestamptz := clock_timestamp();
  svc record;
  project_url text;
  anon_jwt text;
  cron_key text;
  dashboard_cron_key text;
  req_id bigint;
  queue_depth integer := 0;
  system_enabled boolean := false;
begin
  if not pg_try_advisory_xact_lock(hashtextextended('brian-aux-recovery-dispatcher-v1',0)) then
    return jsonb_build_object('status','BUSY');
  end if;

  select coalesce(lower(config_value)='true',false)
    into system_enabled
  from public.brian_evolution_runtime_config
  where config_key='brian_system_enabled';

  if not coalesce(system_enabled,false) then
    return jsonb_build_object('status','SYSTEM_DISABLED','observed_at',now_ts);
  end if;

  update brian_private.aux_dispatch_state s
  set last_http_status=r.status_code,
      last_response_at=r.created,
      updated_at=now_ts
  from net._http_response r
  where s.last_request_id=r.id
    and (s.last_response_at is null or s.last_response_at<r.created);

  select count(*)::int into queue_depth from net.http_request_queue;
  if queue_depth>=8 then
    return jsonb_build_object('status','BACKPRESSURE_QUEUE','queue_depth',queue_depth,'observed_at',now_ts);
  end if;

  with services(service_id,collector_id,endpoint,max_age_seconds,retry_seconds,priority,use_dashboard_key,timeout_ms) as (
    values
      ('big_move','brian-big-move-hunter-v1','brian-big-move-hunter',180,150,10,true,60000),
      ('source_observer','brian-source-observer-v2','brian-source-observer-v2',300,240,20,true,90000),
      ('official_primary','brian-official-primary-eye-v2','brian-official-primary-eye-v2',600,480,30,false,120000),
      ('source_registry','brian-source-registry-v2','brian-source-registry-v2',900,600,40,true,90000)
  ), candidates as (
    select s.*,l.last_success_at,d.last_attempt_at,d.last_response_at,d.last_http_status,
      case when l.last_success_at is null then 1e12
           else extract(epoch from(now_ts-l.last_success_at))/greatest(s.max_age_seconds,1)
      end as overdue_ratio
    from services s
    left join lateral (
      select max(r.finished_at) as last_success_at
      from public.brian_collector_runs r
      where r.collector_id=s.collector_id and r.status='SUCCESS'
    ) l on true
    left join brian_private.aux_dispatch_state d on d.service_id=s.service_id
    where (l.last_success_at is null or l.last_success_at<now_ts-make_interval(secs=>s.max_age_seconds))
      and (d.last_attempt_at is null or d.last_attempt_at<now_ts-make_interval(secs=>s.retry_seconds))
  )
  select * into svc
  from candidates
  order by last_attempt_at asc nulls first, priority asc, overdue_ratio desc
  limit 1;

  if not found then
    return jsonb_build_object('status','IDLE','queue_depth',queue_depth,'observed_at',now_ts);
  end if;

  select decrypted_secret into project_url from vault.decrypted_secrets where name='brian_project_url' limit 1;
  select decrypted_secret into anon_jwt from vault.decrypted_secrets where name='brian_anon_jwt' limit 1;
  select decrypted_secret into cron_key from vault.decrypted_secrets where name='brian_cron_key' limit 1;
  select decrypted_secret into dashboard_cron_key from vault.decrypted_secrets where name='brian_dashboard_cron_key' limit 1;

  if project_url is null or anon_jwt is null or cron_key is null or dashboard_cron_key is null then
    raise exception 'BRIAN_AUX_DISPATCH_SECRETS_UNAVAILABLE';
  end if;

  req_id:=net.http_post(
    url:=project_url||'/functions/v1/'||svc.endpoint,
    headers:=jsonb_build_object(
      'Content-Type','application/json',
      'Authorization','Bearer '||anon_jwt,
      'apikey',anon_jwt,
      'x-brian-cron-key',case when svc.use_dashboard_key then dashboard_cron_key else cron_key end
    ),
    body:='{}'::jsonb,
    timeout_milliseconds:=svc.timeout_ms
  );

  insert into brian_private.aux_dispatch_state(service_id,last_attempt_at,last_request_id,attempt_count,updated_at)
  values(svc.service_id,now_ts,req_id,1,now_ts)
  on conflict(service_id) do update
  set last_attempt_at=excluded.last_attempt_at,
      last_request_id=excluded.last_request_id,
      attempt_count=brian_private.aux_dispatch_state.attempt_count+1,
      updated_at=excluded.updated_at;

  return jsonb_build_object(
    'status','ENQUEUED','service_id',svc.service_id,'collector_id',svc.collector_id,
    'request_id',req_id,'last_success_at',svc.last_success_at,'queue_depth',queue_depth,
    'timeout_ms',svc.timeout_ms,'observed_at',now_ts
  );
end;
$$;
revoke all on function brian_private.dispatch_stale_aux() from public, anon, authenticated;

do $$
declare r record;
begin
  for r in select jobid from cron.job where jobname='brian-aux-recovery-dispatcher-v1' loop
    perform cron.unschedule(r.jobid);
  end loop;
end $$;

select cron.schedule(
  'brian-aux-recovery-dispatcher-v1',
  '43 seconds',
  'select brian_private.dispatch_stale_aux();'
);

insert into public.brian_system_job_registry(job_name,component,operator_managed)
values('brian-aux-recovery-dispatcher-v1','WORLD',true)
on conflict(job_name) do update
set component=excluded.component,operator_managed=excluded.operator_managed;

create or replace function public.brian_sync_meeting_shadow_ledger()
returns jsonb
language plpgsql
security definer
set search_path to 'pg_catalog','public'
as $$
declare
  ev record;
  sensor_observation_id_v text;
  sensor_asset_id_v text;
  sensor_observed_at_v timestamptz;
  alpha_decision_id_v text;
  alpha_observed_at_v timestamptz;
  alpha_asset_id_v text;
  alpha_action_v text;
  alpha_evidence_v double precision;
  alpha_reason_v text;
  treasury_gate_v boolean;
  treasury_reason_v text;
  treasury_blocked_v text[];
  treasury_observed_at_v timestamptz;
  trade_action_id_v text;
  trade_kind_v text;
  trade_observed_at_v timestamptz;
  trade_reference_price_v numeric;
  trade_capital_usd_v numeric;
  trade_reason_v text;
  asset_norm text;
  alpha_norm text;
  source_ok boolean;
  urgency_v text;
  importance_v double precision;
  council_v text;
  status_v text;
  processed integer := 0;
  linked integer := 0;
  traded integer := 0;
begin
  select promotion_gate_open,promotion_gate_reason,blocked_reasons,observed_at
  into treasury_gate_v,treasury_reason_v,treasury_blocked_v,treasury_observed_at_v
  from public.brian_treasury_shadow_snapshots
  order by observed_at desc
  limit 1;

  for ev in
    select e.event_id,e.asset,e.event_kind,e.source_kind,e.source_id,e.published_at,
           e.first_observed_at,e.captured_at,e.claim,e.trust_class,e.provenance_uri
    from public.brian_intel_events e
    where e.first_observed_at >= now()-interval '6 hours'
      and (
        e.trust_class in ('OFFICIAL_PRIMARY','INSTITUTIONAL')
        or e.event_kind ilike 'OFFICIAL%'
        or e.claim ~* '(bitcoin|\bbtc\b|ethereum|ether|\beth\b|solana|\bsol\b|\bxrp\b|\bbnb\b|cardano|\bada\b|crypto|digital asset|stablecoin|\betf\b|central bank|federal reserve|\bfed\b|\becb\b|bank of japan|\bboj\b|interest rate|rate hike|rate cut|inflation|\bcpi\b|\bpce\b|employment|payroll|tariff|sanction|war|conflict|regulat|enforcement|hack|exploit|bankrupt|liquidat)'
      )
    order by
      case
        when e.event_kind='OFFICIAL_MACRO_RELEASE' then 4
        when e.trust_class='OFFICIAL_PRIMARY' then 3
        when e.trust_class='INSTITUTIONAL' then 2
        else 1
      end desc,
      e.first_observed_at desc
    limit 40
  loop
    processed := processed + 1;

    sensor_observation_id_v := null;
    sensor_asset_id_v := null;
    sensor_observed_at_v := null;
    alpha_decision_id_v := null;
    alpha_observed_at_v := null;
    alpha_asset_id_v := null;
    alpha_action_v := null;
    alpha_evidence_v := null;
    alpha_reason_v := null;
    trade_action_id_v := null;
    trade_kind_v := null;
    trade_observed_at_v := null;
    trade_reference_price_v := null;
    trade_capital_usd_v := null;
    trade_reason_v := null;

    select s.observation_id,s.asset_id,s.observed_at
    into sensor_observation_id_v,sensor_asset_id_v,sensor_observed_at_v
    from public.brian_sensor_observations s
    where s.metadata->>'event_id'=ev.event_id
      and s.available=true
    order by s.observed_at desc
    limit 1;

    if sensor_observation_id_v is not null then
      asset_norm := regexp_replace(upper(sensor_asset_id_v),'^CRYPTO:','','i');
    elsif coalesce(ev.asset,'') ~* '^crypto:' then
      asset_norm := regexp_replace(upper(ev.asset),'^CRYPTO:','','i');
    elsif ev.claim ~* '(bitcoin|\bbtc\b)' then asset_norm := 'BTCUSDT';
    elsif ev.claim ~* '(ethereum|ether|\beth\b)' then asset_norm := 'ETHUSDT';
    elsif ev.claim ~* '(solana|\bsol\b)' then asset_norm := 'SOLUSDT';
    elsif ev.claim ~* '\bxrp\b' then asset_norm := 'XRPUSDT';
    elsif ev.claim ~* '\bbnb\b' then asset_norm := 'BNBUSDT';
    elsif ev.claim ~* '(cardano|\bada\b)' then asset_norm := 'ADAUSDT';
    else asset_norm := regexp_replace(upper(coalesce(ev.asset,'GLOBALWORLD')),'[^A-Z0-9]','','g');
    end if;

    if sensor_observation_id_v is not null then
      select d.decision_id,d.observed_at,d.asset_id,d.action,d.evidence_score,d.reason
      into alpha_decision_id_v,alpha_observed_at_v,alpha_asset_id_v,alpha_action_v,alpha_evidence_v,alpha_reason_v
      from public.brian_alpha_decisions d
      where d.asset_id=sensor_asset_id_v
        and d.observed_at>=sensor_observed_at_v
        and d.observed_at<=sensor_observed_at_v+interval '10 minutes'
        and coalesce(d.metadata->'source_evidence_ids_all','[]'::jsonb) ? sensor_observation_id_v
      order by d.observed_at desc
      limit 1;
    end if;

    alpha_norm := case when alpha_asset_id_v is null then null else regexp_replace(upper(alpha_asset_id_v),'^CRYPTO:','','i') end;
    source_ok := ev.trust_class in ('OFFICIAL_PRIMARY','INSTITUTIONAL') or ev.event_kind ilike 'OFFICIAL%';

    if ev.event_kind='OFFICIAL_MACRO_RELEASE' then
      urgency_v := 'CRITICAL'; importance_v := 0.97;
    elsif ev.trust_class='OFFICIAL_PRIMARY' then
      urgency_v := 'CRITICAL'; importance_v := 0.90;
    elsif ev.claim ~* '(rate hike|rate cut|federal reserve|\bfed\b|bank of japan|\bboj\b|hack|exploit|bankrupt|enforcement|sanction|war|conflict)' then
      urgency_v := 'CRITICAL'; importance_v := 0.88;
    else
      urgency_v := 'HIGH'; importance_v := 0.78;
    end if;

    if alpha_decision_id_v is not null then
      linked := linked + 1;
      select a.action_id,a.kind,a.observed_at,a.reference_price,a.capital_usd,a.reason
      into trade_action_id_v,trade_kind_v,trade_observed_at_v,trade_reference_price_v,trade_capital_usd_v,trade_reason_v
      from public.brian_treasury_shadow_actions a
      where a.source_decision_id=alpha_decision_id_v
      order by a.observed_at desc
      limit 1;
    end if;

    if not source_ok then
      council_v := 'Kaynak doğrulaması bekleniyor; olay kalıcı gölge tutanağına alındı.';
    elsif alpha_decision_id_v is null then
      council_v := 'Kaynak doğrulandı; bu olay kimliğine bağlı ALPHA kanıtı bekleniyor.';
    elsif alpha_action_v in ('OPEN_LONG','OPEN_SHORT') and trade_action_id_v is not null then
      council_v := 'Kaynak doğrulandı; olay ALPHA kanıtına bağlandı ve Hazine SHADOW aksiyonu kaydetti.';
    elsif alpha_action_v in ('OPEN_LONG','OPEN_SHORT') then
      council_v := 'Kaynak doğrulandı; olay ALPHA kanıtına bağlandı. Hazine/risk sonucu bekleniyor.';
    elsif alpha_action_v='VETO' then
      council_v := 'Kaynak doğrulandı; ilişkili ALPHA kararı VETO.';
    else
      council_v := 'Kaynak doğrulandı; ilişkili ALPHA kararı WAIT.';
    end if;

    status_v := case
      when trade_kind_v='OPEN' then 'İŞLEM AÇILDI'
      when trade_kind_v='EXIT' then 'KAPANDI'
      else 'GÖZLEMLENİYOR'
    end;
    if trade_action_id_v is not null then traded := traded + 1; end if;

    insert into public.brian_meeting_shadow_ledger(
      event_key,event_time,first_seen_at,last_seen_at,urgency,importance,title,summary,original_claim,
      event_kind,source,publisher,source_uri,source_trust,source_verified,asset,alpha_asset,alpha_action,
      alpha_evidence,council_decision,treasury_gate,treasury_reason,status,trade_kind,trade_action_id,
      trade_observed_at,trade_reference_price,trade_capital_usd,trade_reason,payload,evidence_class,
      shadow_only,live_execution,created_at,updated_at
    ) values(
      'event:'||ev.event_id,
      coalesce(ev.published_at,ev.first_observed_at,ev.captured_at),
      ev.first_observed_at,
      now(),
      urgency_v,importance_v,
      left(ev.claim,1000),
      left(ev.claim,4000),
      left(ev.claim,4000),
      ev.event_kind,
      ev.source_id,
      ev.source_id,
      ev.provenance_uri,
      ev.trust_class,
      source_ok,
      asset_norm,
      alpha_norm,
      alpha_action_v,
      alpha_evidence_v,
      council_v,
      treasury_gate_v,
      coalesce(treasury_reason_v,array_to_string(treasury_blocked_v,' · ')),
      status_v,
      trade_kind_v,
      trade_action_id_v,
      trade_observed_at_v,
      trade_reference_price_v,
      trade_capital_usd_v,
      trade_reason_v,
      jsonb_build_object(
        'event_id',ev.event_id,
        'source_kind',ev.source_kind,
        'alpha_decision_id',alpha_decision_id_v,
        'alpha_observed_at',alpha_observed_at_v,
        'alpha_reason',alpha_reason_v,
        'sensor_observation_id',sensor_observation_id_v,
        'treasury_observed_at',treasury_observed_at_v,
        'backend_sync','brian.meeting-backend-sync.v1',
        'shadow_only',true,
        'live_execution',false
      ),
      'PROSPECTIVE_MEETING_SHADOW',
      true,false,now(),now()
    )
    on conflict(event_key) do update
    set last_seen_at=excluded.last_seen_at,
        urgency=excluded.urgency,
        importance=excluded.importance,
        title=excluded.title,
        summary=excluded.summary,
        original_claim=excluded.original_claim,
        source=excluded.source,
        publisher=excluded.publisher,
        source_uri=excluded.source_uri,
        source_trust=excluded.source_trust,
        source_verified=excluded.source_verified,
        asset=excluded.asset,
        alpha_asset=excluded.alpha_asset,
        alpha_action=excluded.alpha_action,
        alpha_evidence=excluded.alpha_evidence,
        council_decision=excluded.council_decision,
        treasury_gate=excluded.treasury_gate,
        treasury_reason=excluded.treasury_reason,
        status=excluded.status,
        trade_kind=excluded.trade_kind,
        trade_action_id=excluded.trade_action_id,
        trade_observed_at=excluded.trade_observed_at,
        trade_reference_price=excluded.trade_reference_price,
        trade_capital_usd=excluded.trade_capital_usd,
        trade_reason=excluded.trade_reason,
        payload=excluded.payload,
        updated_at=excluded.updated_at;
  end loop;

  return jsonb_build_object(
    'status','SUCCESS','processed',processed,'alpha_linked',linked,'trade_linked',traded,
    'shadow_only',true,'live_execution',false,'observed_at',now()
  );
end;
$$;
revoke all on function public.brian_sync_meeting_shadow_ledger() from public, anon, authenticated;

do $$
declare r record;
begin
  for r in select jobid from cron.job where jobname='brian-meeting-backend-sync-1m' loop
    perform cron.unschedule(r.jobid);
  end loop;
end $$;

select cron.schedule(
  'brian-meeting-backend-sync-1m',
  '* * * * *',
  'select public.brian_sync_meeting_shadow_ledger();'
);

insert into public.brian_system_job_registry(job_name,component,operator_managed)
values('brian-meeting-backend-sync-1m','WORLD',true)
on conflict(job_name) do update
set component=excluded.component,operator_managed=excluded.operator_managed;

select public.brian_sync_meeting_shadow_ledger();

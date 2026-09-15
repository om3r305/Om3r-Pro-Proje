create or replace function public.brian_meeting_shadow_upsert(p_dashboard_key text, p_record jsonb)
returns jsonb
language plpgsql
security definer
set search_path = public, extensions
as $$
declare
  v_expected text;
  v_event_key text;
  v_event_time timestamptz;
  v_title text;
  v_urgency text;
  v_asset text;
  v_alpha_asset text;
  v_trade public.brian_treasury_shadow_actions%rowtype;
  v_status text;
  v_row public.brian_meeting_shadow_ledger%rowtype;
begin
  select dashboard_key_sha256 into v_expected from public.brian_dashboard_auth where auth_id = 'control-v3';
  if v_expected is null or encode(extensions.digest(coalesce(p_dashboard_key,''), 'sha256'), 'hex') <> v_expected then
    raise exception 'UNAUTHORIZED_DASHBOARD' using errcode = '28000';
  end if;
  v_event_key := left(trim(coalesce(p_record->>'event_key','')),500);
  v_title := left(trim(coalesce(p_record->>'title','')),1000);
  v_event_time := nullif(p_record->>'event_time','')::timestamptz;
  if v_event_key = '' or v_title = '' or v_event_time is null then raise exception 'INVALID_EVENT_RECORD' using errcode = '22023'; end if;
  v_urgency := upper(coalesce(p_record->>'urgency','HIGH'));
  if v_urgency not in ('CRITICAL','HIGH','MEDIUM','LOW') then v_urgency := 'HIGH'; end if;
  v_asset := nullif(upper(regexp_replace(coalesce(p_record->>'asset',''), '[^A-Za-z0-9]', '', 'g')),'');
  v_alpha_asset := nullif(upper(regexp_replace(regexp_replace(coalesce(p_record->>'alpha_asset',''), '^crypto:', '', 'i'), '[^A-Za-z0-9]', '', 'g')),'');
  select a.* into v_trade from public.brian_treasury_shadow_actions a
  where upper(regexp_replace(a.asset_id, '^crypto:', '', 'i')) = coalesce(v_alpha_asset,v_asset)
    and a.observed_at between v_event_time - interval '5 minutes' and v_event_time + interval '12 hours'
    and a.shadow_only is true and a.live_execution is false order by a.observed_at desc limit 1;
  v_status := left(coalesce(nullif(p_record->>'status',''),'GÖZLEMLENİYOR'),120);
  if v_trade.action_id is not null then
    if v_trade.kind = 'EXIT' then v_status := 'KAPANDI'; elsif v_trade.kind = 'OPEN' then v_status := 'İŞLEM AÇILDI'; end if;
  end if;
  insert into public.brian_meeting_shadow_ledger(event_key,event_time,first_seen_at,last_seen_at,urgency,importance,title,summary,original_claim,event_kind,source,publisher,source_uri,source_trust,source_verified,asset,alpha_asset,alpha_action,alpha_evidence,council_decision,treasury_gate,treasury_reason,status,trade_kind,trade_action_id,trade_observed_at,trade_reference_price,trade_capital_usd,trade_reason,payload,evidence_class,shadow_only,live_execution,updated_at)
  values(v_event_key,v_event_time,coalesce(nullif(p_record->>'first_seen_at','')::timestamptz,v_event_time),now(),v_urgency,nullif(p_record->>'importance','')::double precision,v_title,nullif(left(p_record->>'summary',4000),''),nullif(left(p_record->>'original_claim',4000),''),nullif(left(p_record->>'event_kind',200),''),nullif(left(p_record->>'source',300),''),nullif(left(p_record->>'publisher',300),''),nullif(left(p_record->>'source_uri',3000),''),nullif(left(p_record->>'source_trust',300),''),coalesce((p_record->>'source_verified')::boolean,false),v_asset,v_alpha_asset,nullif(left(p_record->>'alpha_action',120),''),nullif(p_record->>'alpha_evidence','')::double precision,nullif(left(p_record->>'council_decision',4000),''),nullif(p_record->>'treasury_gate','')::boolean,nullif(left(p_record->>'treasury_reason',4000),''),v_status,case when v_trade.action_id is not null then v_trade.kind end,case when v_trade.action_id is not null then v_trade.action_id end,case when v_trade.action_id is not null then v_trade.observed_at end,case when v_trade.action_id is not null then v_trade.reference_price end,case when v_trade.action_id is not null then v_trade.capital_usd end,case when v_trade.action_id is not null then v_trade.reason end,coalesce(p_record->'payload','{}'::jsonb),'PROSPECTIVE_MEETING_SHADOW',true,false,now())
  on conflict (event_key) do update set last_seen_at=excluded.last_seen_at,urgency=excluded.urgency,importance=excluded.importance,title=excluded.title,summary=excluded.summary,original_claim=excluded.original_claim,event_kind=excluded.event_kind,source=excluded.source,publisher=excluded.publisher,source_uri=excluded.source_uri,source_trust=excluded.source_trust,source_verified=excluded.source_verified,asset=excluded.asset,alpha_asset=excluded.alpha_asset,alpha_action=excluded.alpha_action,alpha_evidence=excluded.alpha_evidence,council_decision=excluded.council_decision,treasury_gate=excluded.treasury_gate,treasury_reason=excluded.treasury_reason,status=excluded.status,trade_kind=excluded.trade_kind,trade_action_id=excluded.trade_action_id,trade_observed_at=excluded.trade_observed_at,trade_reference_price=excluded.trade_reference_price,trade_capital_usd=excluded.trade_capital_usd,trade_reason=excluded.trade_reason,payload=excluded.payload,updated_at=now()
  returning * into v_row;
  return jsonb_build_object('status','UPSERTED','event_key',v_row.event_key,'updated_at',v_row.updated_at,'shadow_only',true,'live_execution',false);
end;
$$;
revoke all on function public.brian_meeting_shadow_upsert(text,jsonb) from public;
grant execute on function public.brian_meeting_shadow_upsert(text,jsonb) to anon;
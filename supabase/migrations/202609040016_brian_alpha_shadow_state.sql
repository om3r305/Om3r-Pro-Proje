-- Brian ALPHA v2 stateful shadow position book and frozen Phase 3.7 prospective comparator.
-- SHADOW ONLY. This migration only reads frozen Phase 3.7 outputs; it does not mutate Phase 3.7.

create table if not exists public.brian_alpha_shadow_position_events (
  event_id uuid primary key default gen_random_uuid(),
  decision_id text not null unique references public.brian_alpha_decisions(decision_id),
  asset_id text not null,
  event_ts timestamptz not null,
  event_type text not null check (event_type in ('OPEN','REAFFIRM','FLIP')),
  position_before smallint not null check (position_before between -1 and 1),
  position_after smallint not null check (position_after in (-1,1)),
  entry_price_before numeric,
  entry_price_after numeric not null check (entry_price_after > 0),
  entry_ts_before timestamptz,
  entry_ts_after timestamptz not null,
  action text not null check (action in ('OPEN_LONG','OPEN_SHORT')),
  reference_price numeric not null check (reference_price > 0),
  realized_gross_bps double precision,
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW'
    check (evidence_class='PROSPECTIVE_DEVELOPMENT_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now(),
  constraint brian_alpha_shadow_before_state check (
    (position_before=0 and entry_price_before is null and entry_ts_before is null) or
    (position_before in (-1,1) and entry_price_before > 0 and entry_ts_before is not null)
  ),
  constraint brian_alpha_shadow_reaffirm_consistency check (
    event_type <> 'REAFFIRM' or (
      position_before=position_after and
      entry_price_before=entry_price_after and
      entry_ts_before=entry_ts_after and
      realized_gross_bps is null
    )
  ),
  constraint brian_alpha_shadow_open_consistency check (
    event_type <> 'OPEN' or (
      position_before=0 and
      entry_price_before is null and
      entry_ts_before is null and
      realized_gross_bps is null
    )
  ),
  constraint brian_alpha_shadow_flip_consistency check (
    event_type <> 'FLIP' or (
      position_before=-position_after and
      entry_price_before > 0 and
      entry_ts_before is not null and
      realized_gross_bps is not null
    )
  )
);

create index if not exists brian_alpha_shadow_position_asset_time_idx
  on public.brian_alpha_shadow_position_events(asset_id,event_ts desc,created_at desc);

alter table public.brian_alpha_shadow_position_events enable row level security;
revoke all on public.brian_alpha_shadow_position_events from anon,authenticated;
revoke insert,update,delete,truncate,references,trigger on public.brian_alpha_shadow_position_events from service_role;
grant select on public.brian_alpha_shadow_position_events to service_role;

drop trigger if exists brian_alpha_shadow_position_events_append_only on public.brian_alpha_shadow_position_events;
create trigger brian_alpha_shadow_position_events_append_only
before update or delete on public.brian_alpha_shadow_position_events
for each row execute function public.brian_reject_mutation();

create or replace function brian_private.capture_alpha_shadow_position_event()
returns trigger
language plpgsql
security definer
set search_path = pg_catalog, public, brian_private
as $$
declare
  target_position smallint;
  previous_row public.brian_alpha_shadow_position_events%rowtype;
  realized_bps double precision;
begin
  if new.action not in ('OPEN_LONG','OPEN_SHORT') then
    return new;
  end if;
  if new.observed_reference_price is null or new.observed_reference_price <= 0 then
    return new;
  end if;

  target_position := case when new.action='OPEN_LONG' then 1 else -1 end;
  perform pg_advisory_xact_lock(hashtextextended('brian-alpha-shadow:' || new.asset_id,0));

  select *
    into previous_row
    from public.brian_alpha_shadow_position_events
   where asset_id=new.asset_id
   order by event_ts desc,created_at desc,event_id desc
   limit 1;

  if found and new.observed_at <= previous_row.event_ts then
    -- Never let an out-of-order or same-time late insert rewrite prospective state.
    return new;
  end if;

  if not found then
    insert into public.brian_alpha_shadow_position_events(
      decision_id,asset_id,event_ts,event_type,position_before,position_after,
      entry_price_before,entry_price_after,entry_ts_before,entry_ts_after,
      action,reference_price,realized_gross_bps,metadata
    ) values (
      new.decision_id,new.asset_id,new.observed_at,'OPEN',0,target_position,
      null,new.observed_reference_price,null,new.observed_at,
      new.action,new.observed_reference_price,null,
      jsonb_build_object('state_source','brian_alpha_decisions','position_semantics','direction_only_no_notional')
    );
    return new;
  end if;

  if previous_row.position_after=target_position then
    insert into public.brian_alpha_shadow_position_events(
      decision_id,asset_id,event_ts,event_type,position_before,position_after,
      entry_price_before,entry_price_after,entry_ts_before,entry_ts_after,
      action,reference_price,realized_gross_bps,metadata
    ) values (
      new.decision_id,new.asset_id,new.observed_at,'REAFFIRM',
      previous_row.position_after,target_position,
      previous_row.entry_price_after,previous_row.entry_price_after,
      previous_row.entry_ts_after,previous_row.entry_ts_after,
      new.action,new.observed_reference_price,null,
      jsonb_build_object('state_source','brian_alpha_decisions','position_semantics','direction_only_no_notional')
    );
    return new;
  end if;

  realized_bps := ((new.observed_reference_price::double precision / previous_row.entry_price_after::double precision) - 1.0)
    * 10000.0 * previous_row.position_after;

  insert into public.brian_alpha_shadow_position_events(
    decision_id,asset_id,event_ts,event_type,position_before,position_after,
    entry_price_before,entry_price_after,entry_ts_before,entry_ts_after,
    action,reference_price,realized_gross_bps,metadata
  ) values (
    new.decision_id,new.asset_id,new.observed_at,'FLIP',
    previous_row.position_after,target_position,
    previous_row.entry_price_after,new.observed_reference_price,
    previous_row.entry_ts_after,new.observed_at,
    new.action,new.observed_reference_price,realized_bps,
    jsonb_build_object('state_source','brian_alpha_decisions','position_semantics','direction_only_no_notional')
  );
  return new;
end;
$$;

revoke all on function brian_private.capture_alpha_shadow_position_event() from public,anon,authenticated,service_role;

drop trigger if exists brian_alpha_capture_shadow_position on public.brian_alpha_decisions;
create trigger brian_alpha_capture_shadow_position
after insert on public.brian_alpha_decisions
for each row execute function brian_private.capture_alpha_shadow_position_event();

create or replace view public.brian_alpha_shadow_position_book
with (security_invoker=true)
as
select distinct on (asset_id)
  asset_id,
  position_after as position,
  entry_price_after as entry_price,
  entry_ts_after as entry_ts,
  event_ts as last_action_at,
  event_type as last_event_type,
  decision_id as last_decision_id,
  reference_price as last_reference_price,
  realized_gross_bps as last_flip_realized_gross_bps,
  shadow_only,
  live_execution
from public.brian_alpha_shadow_position_events
order by asset_id,event_ts desc,created_at desc,event_id desc;

revoke all on public.brian_alpha_shadow_position_book from public,anon,authenticated;
grant select on public.brian_alpha_shadow_position_book to service_role;

create table if not exists public.brian_alpha_phase37_comparisons (
  comparison_id uuid primary key default gen_random_uuid(),
  decision_id text not null references public.brian_alpha_decisions(decision_id),
  asset_id text not null,
  observed_at timestamptz not null,
  alpha_action text not null,
  alpha_direction smallint not null check (alpha_direction between -1 and 1),
  alpha_position_intent smallint not null check (alpha_position_intent between -1 and 1),
  phase37_experiment_id text not null,
  phase37_policy_kind text not null check (phase37_policy_kind in ('NATIVE','PROFIT')),
  phase37_tick_id text,
  phase37_observed_at timestamptz,
  phase37_age_seconds double precision check (phase37_age_seconds is null or phase37_age_seconds >= 0),
  phase37_target_weight numeric,
  phase37_direction smallint check (phase37_direction is null or phase37_direction between -1 and 1),
  relationship text not null check (relationship in (
    'AGREES','DISAGREES','ALPHA_ONLY','PHASE37_ONLY','BOTH_FLAT','PHASE37_UNAVAILABLE'
  )),
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW'
    check (evidence_class='PROSPECTIVE_DEVELOPMENT_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now(),
  unique(decision_id,phase37_policy_kind),
  constraint brian_alpha_phase37_causal check (
    phase37_observed_at is null or phase37_observed_at <= observed_at
  )
);

create index if not exists brian_alpha_phase37_comparison_asset_time_idx
  on public.brian_alpha_phase37_comparisons(asset_id,observed_at desc);
create index if not exists brian_alpha_phase37_comparison_relation_idx
  on public.brian_alpha_phase37_comparisons(relationship,observed_at desc);

alter table public.brian_alpha_phase37_comparisons enable row level security;
revoke all on public.brian_alpha_phase37_comparisons from anon,authenticated;
revoke insert,update,delete,truncate,references,trigger on public.brian_alpha_phase37_comparisons from service_role;
grant select on public.brian_alpha_phase37_comparisons to service_role;

drop trigger if exists brian_alpha_phase37_comparisons_append_only on public.brian_alpha_phase37_comparisons;
create trigger brian_alpha_phase37_comparisons_append_only
before update or delete on public.brian_alpha_phase37_comparisons
for each row execute function public.brian_reject_mutation();

create or replace function brian_private.capture_alpha_phase37_comparisons()
returns trigger
language plpgsql
security definer
set search_path = pg_catalog, public, brian_private
as $$
declare
  source_experiment constant text := 'phase37-prospective-live-20260903';
  policy text;
  symbol text;
  alpha_intent smallint;
  phase_tick record;
  phase_weight numeric;
  phase_direction smallint;
  age_seconds double precision;
  relation text;
begin
  alpha_intent := case
    when new.action='OPEN_LONG' then 1
    when new.action='OPEN_SHORT' then -1
    else 0
  end;
  symbol := case when position(':' in new.asset_id)>0 then split_part(new.asset_id,':',2) else new.asset_id end;

  for policy in select unnest(array['NATIVE','PROFIT']::text[]) loop
    select tick_id,observed_at,target_weights
      into phase_tick
      from public.brian_live_shadow_ticks
     where experiment_id=source_experiment
       and policy_kind=policy
       and observed_at <= new.observed_at
     order by observed_at desc,tick_id desc
     limit 1;

    if not found or new.observed_at - phase_tick.observed_at > interval '10 minutes' then
      insert into public.brian_alpha_phase37_comparisons(
        decision_id,asset_id,observed_at,alpha_action,alpha_direction,alpha_position_intent,
        phase37_experiment_id,phase37_policy_kind,relationship,metadata
      ) values (
        new.decision_id,new.asset_id,new.observed_at,new.action,new.direction,alpha_intent,
        source_experiment,policy,'PHASE37_UNAVAILABLE',
        jsonb_build_object('comparison_mode','at_or_before_only','max_phase37_age_seconds',600,'future_leakage_allowed',false)
      );
      continue;
    end if;

    age_seconds := extract(epoch from (new.observed_at-phase_tick.observed_at));
    phase_weight := coalesce((phase_tick.target_weights ->> symbol)::numeric,0);
    phase_direction := case when phase_weight>0 then 1 when phase_weight<0 then -1 else 0 end;

    relation := case
      when alpha_intent=0 and phase_direction=0 then 'BOTH_FLAT'
      when alpha_intent=0 and phase_direction<>0 then 'PHASE37_ONLY'
      when alpha_intent<>0 and phase_direction=0 then 'ALPHA_ONLY'
      when alpha_intent=phase_direction then 'AGREES'
      else 'DISAGREES'
    end;

    insert into public.brian_alpha_phase37_comparisons(
      decision_id,asset_id,observed_at,alpha_action,alpha_direction,alpha_position_intent,
      phase37_experiment_id,phase37_policy_kind,phase37_tick_id,phase37_observed_at,
      phase37_age_seconds,phase37_target_weight,phase37_direction,relationship,metadata
    ) values (
      new.decision_id,new.asset_id,new.observed_at,new.action,new.direction,alpha_intent,
      source_experiment,policy,phase_tick.tick_id,phase_tick.observed_at,
      age_seconds,phase_weight,phase_direction,relation,
      jsonb_build_object('comparison_mode','at_or_before_only','max_phase37_age_seconds',600,'future_leakage_allowed',false)
    );
  end loop;
  return new;
end;
$$;

revoke all on function brian_private.capture_alpha_phase37_comparisons() from public,anon,authenticated,service_role;

drop trigger if exists brian_alpha_capture_phase37_comparison on public.brian_alpha_decisions;
create trigger brian_alpha_capture_phase37_comparison
after insert on public.brian_alpha_decisions
for each row execute function brian_private.capture_alpha_phase37_comparisons();

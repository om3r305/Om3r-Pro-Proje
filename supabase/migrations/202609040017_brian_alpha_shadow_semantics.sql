-- Brian ALPHA v2 shadow-state semantic hardening.
-- SHADOW ONLY. Phase 3.7 remains frozen/read-only.
--
-- WAIT/VETO mean "no new position action": if a causal ALPHA shadow position already exists,
-- keep that position intent for the frozen Phase 3.7 comparison instead of pretending WAIT is
-- flat. An out-of-order/same-time OPEN that the shadow-position trigger will ignore must likewise
-- not rewrite comparator intent. All lookups used to recover prior state are causal at-or-before.

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
  latest_event_ts timestamptz;
  causal_prior_position smallint;
  ignored_out_of_order boolean := false;
  phase_tick record;
  phase_weight numeric;
  phase_direction smallint;
  age_seconds double precision;
  relation text;
begin
  -- First determine whether this decision is later than the latest already-materialized state.
  -- This mirrors capture_alpha_shadow_position_event()'s <= out-of-order/same-time guard without
  -- reading a future event as if it existed at the decision timestamp.
  select event_ts
    into latest_event_ts
    from public.brian_alpha_shadow_position_events
   where asset_id=new.asset_id
   order by event_ts desc,created_at desc,event_id desc
   limit 1;

  if latest_event_ts is not null and new.observed_at <= latest_event_ts then
    ignored_out_of_order := true;
    select position_after
      into causal_prior_position
      from public.brian_alpha_shadow_position_events
     where asset_id=new.asset_id
       and event_ts < new.observed_at
     order by event_ts desc,created_at desc,event_id desc
     limit 1;
    alpha_intent := coalesce(causal_prior_position,0);
  elsif new.action='OPEN_LONG' then
    alpha_intent := 1;
  elsif new.action='OPEN_SHORT' then
    alpha_intent := -1;
  else
    -- WAIT/VETO hold the latest causal shadow position. If no position has ever been opened,
    -- holding means flat (0); it does not synthesize a position.
    select position_after
      into causal_prior_position
      from public.brian_alpha_shadow_position_events
     where asset_id=new.asset_id
       and event_ts <= new.observed_at
     order by event_ts desc,created_at desc,event_id desc
     limit 1;
    alpha_intent := coalesce(causal_prior_position,0);
  end if;

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
        jsonb_build_object(
          'comparison_mode','at_or_before_only',
          'max_phase37_age_seconds',600,
          'future_leakage_allowed',false,
          'wait_veto_semantics','hold_prior_causal_shadow_position',
          'out_of_order_decision_ignored_for_position_intent',ignored_out_of_order
        )
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
      jsonb_build_object(
        'comparison_mode','at_or_before_only',
        'max_phase37_age_seconds',600,
        'future_leakage_allowed',false,
        'wait_veto_semantics','hold_prior_causal_shadow_position',
        'out_of_order_decision_ignored_for_position_intent',ignored_out_of_order
      )
    );
  end loop;
  return new;
end;
$$;

revoke all on function brian_private.capture_alpha_phase37_comparisons() from public,anon,authenticated,service_role;

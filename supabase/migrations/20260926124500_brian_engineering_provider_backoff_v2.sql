-- Brian Engineer provider backoff should represent provider exhaustion only.
-- A downstream analysis-format/contract failure is not provider unavailability and must
-- not poison the provider circuit. Human approval, shadow-only and DIP isolation stay unchanged.

create or replace function brian_private.engineering_provider_guard_event_trigger()
returns trigger
language plpgsql
security definer
set search_path to 'public','brian_private','pg_temp'
as $function$
declare
  prior_phase text;
  current_streak integer := 0;
  next_streak integer := 1;
  backoff_hours integer := 1;
begin
  select phase into prior_phase
  from public.brian_evolution_engineering_runs
  where run_id=new.run_id;

  if new.phase='UNDERSTAND' and new.passed is true then
    update public.brian_evolution_engineering_control
    set metadata=(coalesce(metadata,'{}'::jsonb)
          - 'engineering_provider_backoff_until'
          - 'engineering_provider_backoff_reason')
          || jsonb_build_object(
            'engineering_provider_state','HEALTHY',
            'engineering_provider_failure_streak',0,
            'engineering_provider_last_recovered_at',now(),
            'engineering_provider_guard','EXPONENTIAL_PROVIDER_EXHAUSTION_BACKOFF_V2'
          ),
        updated_at=now()
    where control_id='default';
    return new;
  end if;

  if new.phase='BLOCKED'
     and prior_phase='CLAIMED'
     and coalesce(new.payload->>'provider_exhausted','false')='true' then
    select case
      when coalesce(metadata->>'engineering_provider_failure_streak','') ~ '^[0-9]+$'
        then (metadata->>'engineering_provider_failure_streak')::integer
      else 0
    end
    into current_streak
    from public.brian_evolution_engineering_control
    where control_id='default';

    next_streak := least(6,greatest(1,current_streak+1));
    backoff_hours := least(24,(power(2,next_streak-1))::integer);

    update public.brian_evolution_engineering_control
    set metadata=coalesce(metadata,'{}'::jsonb) || jsonb_build_object(
          'engineering_provider_state','BACKOFF_PROVIDER_EXHAUSTED',
          'engineering_provider_failure_streak',next_streak,
          'engineering_provider_backoff_hours',backoff_hours,
          'engineering_provider_backoff_until',now()+make_interval(hours=>backoff_hours),
          'engineering_provider_backoff_reason','ALL_AI_PROVIDERS_UNAVAILABLE',
          'engineering_provider_last_failure_at',now(),
          'engineering_provider_guard','EXPONENTIAL_PROVIDER_EXHAUSTION_BACKOFF_V2'
        ),
        updated_at=now()
    where control_id='default';
  end if;

  return new;
end;
$function$;

-- The latest blocked run had provider_exhausted=false and failed only at the
-- response-format validation layer. Clear the stale provider backoff it created.
update public.brian_evolution_engineering_control
set metadata=(coalesce(metadata,'{}'::jsonb)
      - 'engineering_provider_backoff_until'
      - 'engineering_provider_backoff_reason'
      - 'engineering_provider_backoff_hours')
      || jsonb_build_object(
        'engineering_provider_state','AVAILABLE_NON_PROVIDER_FAILURE',
        'engineering_provider_failure_streak',0,
        'engineering_provider_guard','EXPONENTIAL_PROVIDER_EXHAUSTION_BACKOFF_V2',
        'engineering_provider_last_recovered_at',now()
      ),
    updated_at=now()
where control_id='default'
  and coalesce(metadata->>'engineering_provider_state','')='BACKOFF_PRE_UNDERSTAND_FAILURE';

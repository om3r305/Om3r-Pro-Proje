-- Brian ALPHA calibration challenger v1.
-- SHADOW measurement only: canonical ALPHA decisions/weights are never mutated here.

create table if not exists public.brian_alpha_calibration_challenger (
  challenger_id text primary key,
  decision_id text not null,
  observed_at timestamptz not null,
  evaluated_at timestamptz not null default now(),
  asset_id text not null,
  canonical_action text not null,
  canonical_direction integer not null check (canonical_direction in (-1,0,1)),
  canonical_evidence_score numeric not null,
  challenger_action text not null check (challenger_action in (
    'KEEP_VETO','ALLOW_ACTION','DOWNGRADE_TO_WAIT',
    'PROMOTE_CANDIDATE_LONG','PROMOTE_CANDIDATE_SHORT','KEEP_WAIT'
  )),
  calibration_score numeric not null check (calibration_score >= 0 and calibration_score <= 1),
  mature_support_count integer not null check (mature_support_count >= 0),
  avg_support_cost_adjusted_bps numeric not null,
  avg_support_bayesian_hit_rate numeric not null check (avg_support_bayesian_hit_rate >= 0 and avg_support_bayesian_hit_rate <= 1),
  reliability_window_end timestamptz not null,
  rationale jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW' check (evidence_class='PROSPECTIVE_DEVELOPMENT_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now()
);

create unique index if not exists brian_alpha_calibration_challenger_decision_unique
  on public.brian_alpha_calibration_challenger(decision_id);
create index if not exists brian_alpha_calibration_challenger_observed_idx
  on public.brian_alpha_calibration_challenger(observed_at desc, challenger_id desc);
create index if not exists brian_alpha_calibration_challenger_action_idx
  on public.brian_alpha_calibration_challenger(challenger_action, evaluated_at desc);

alter table public.brian_alpha_calibration_challenger enable row level security;
revoke all on public.brian_alpha_calibration_challenger from anon, authenticated;
revoke update,delete,truncate,references,trigger on public.brian_alpha_calibration_challenger from service_role;
grant select,insert on public.brian_alpha_calibration_challenger to service_role;

drop trigger if exists brian_alpha_calibration_challenger_append_only on public.brian_alpha_calibration_challenger;
create trigger brian_alpha_calibration_challenger_append_only
  before update or delete on public.brian_alpha_calibration_challenger
  for each row execute function public.brian_reject_mutation();

comment on table public.brian_alpha_calibration_challenger is
  'Append-only SHADOW challenger decisions derived from prospective reliability calibration. Never mutates canonical ALPHA.';

do $brian$
declare
  v_jobid bigint;
begin
  for v_jobid in
    select jobid from cron.job where jobname = 'brian-alpha-calibration-challenger-v1-5m'
  loop
    perform cron.unschedule(v_jobid);
  end loop;
end
$brian$;

select cron.schedule(
  'brian-alpha-calibration-challenger-v1-5m',
  '4-59/5 * * * *',
  $$
  select net.http_post(
    url := (
      select decrypted_secret || '/functions/v1/brian-alpha-calibration-challenger'
      from vault.decrypted_secrets
      where name='brian_project_url'
      limit 1
    ),
    headers := jsonb_build_object(
      'Content-Type','application/json',
      'Authorization','Bearer ' || (
        select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1
      ),
      'apikey',(
        select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1
      ),
      'x-brian-cron-key',(
        select decrypted_secret from vault.decrypted_secrets where name='brian_dashboard_cron_key' limit 1
      )
    ),
    body := '{}'::jsonb,
    timeout_milliseconds := 45000
  );
  $$
);

create table public.brian_learning_episodes (
 episode_id text primary key check(episode_id ~ '^[a-f0-9]{64}$'),
 observed_at timestamptz not null, window_start timestamptz not null, window_end timestamptz not null,
 kind text not null, observation text not null, metrics jsonb not null, evidence_refs text[] not null,
 next_experiment text not null, evidence_class text not null,
 shadow_only boolean not null default true check(shadow_only),
 live_execution boolean not null default false check(not live_execution),
 created_at timestamptz not null default now(), check(window_start<=window_end)
);
create table public.brian_research_missions (
 mission_id text primary key check(mission_id ~ '^[a-f0-9]{64}$'), hypothesis_id text,
 question text not null, search_query text not null check(length(search_query)<=256), priority double precision not null check(priority between 0 and 1),
 evidence_refs text[] not null default '{}', success_criteria text[] not null default '{}',
 state text not null check(state in ('QUEUED','EVIDENCE_COLLECTED','PARTIAL','NO_EVIDENCE')),
 attempts integer not null default 0 check(attempts>=0), created_at timestamptz not null default now(),
 last_researched_at timestamptz, next_research_at timestamptz not null, last_error text, version text not null
);
create index brian_research_missions_due_idx on public.brian_research_missions(next_research_at,priority desc);
create table public.brian_research_documents (
 document_id text primary key check(document_id ~ '^[a-f0-9]{64}$'),
 mission_id text not null references public.brian_research_missions(mission_id), provider text not null,
 canonical_uri text not null check(canonical_uri ~ '^https://'), title text not null,
 excerpt text not null check(length(excerpt)<=1200), published_at timestamptz, captured_at timestamptz not null,
 content_sha256 text not null check(content_sha256 ~ '^[a-f0-9]{64}$'), license text,
 metadata jsonb not null default '{}', trust_class text not null check(trust_class='UNVERIFIED_EXTERNAL'),
 evidence_role text not null check(evidence_role='RESEARCH_LEAD_NOT_TRADE_SIGNAL'),
 code_executed boolean not null default false check(not code_executed),
 shadow_only boolean not null default true check(shadow_only),
 live_execution boolean not null default false check(not live_execution), created_at timestamptz not null default now()
);
create index brian_research_documents_context_idx on public.brian_research_documents(captured_at desc,mission_id);
create index brian_learning_episodes_observed_idx on public.brian_learning_episodes(observed_at desc);
alter table public.brian_learning_episodes enable row level security;
alter table public.brian_research_missions enable row level security;
alter table public.brian_research_documents enable row level security;
revoke all on public.brian_learning_episodes,public.brian_research_missions,public.brian_research_documents from public,anon,authenticated;
grant select,insert on public.brian_learning_episodes,public.brian_research_documents to service_role;
grant select,insert,update on public.brian_research_missions to service_role;
create trigger brian_learning_episodes_append_only before update or delete on public.brian_learning_episodes
for each row execute function public.brian_reject_mutation();
create trigger brian_research_documents_append_only before update or delete on public.brian_research_documents
for each row execute function public.brian_reject_mutation();

create function public.brian_learning_window_v1(p_now timestamptz default now()) returns jsonb
language sql stable security invoker set search_path=pg_catalog,public as $fn$
with decisions as materialized (
 select decision_id,observed_at,action from public.brian_alpha_decisions
 where observed_at>=p_now-interval '6 hours' and observed_at<=p_now
 order by observed_at desc,decision_id limit 300
), outcomes as materialized (
 select o.decision_id,o.direction_adjusted_return,o.metadata from public.brian_alpha_decision_outcomes o
 join decisions d on d.decision_id=o.decision_id
 where o.observed_at>=p_now-interval '6 hours' and o.observed_at<=p_now
 and o.created_at<=p_now and o.resolved_at<=p_now and o.horizon_seconds=300
)
select jsonb_build_object('window_start',p_now-interval '6 hours','window_end',p_now,
 'sample_policy','latest_300_decisions_not_random','sampled_decisions',(select count(*) from decisions),
 'mature_5m',(select count(*) from decisions where observed_at+interval '5 minutes'<=p_now),
 'resolved_5m',(select count(*) from outcomes),
 'action_samples_5m',(select count(*) from outcomes o join decisions d using(decision_id)
   where d.action in ('OPEN_LONG','OPEN_SHORT') and o.metadata->>'estimated_round_trip_cost_bps' is not null),
 'mean_action_net_bps_5m',(select avg(o.direction_adjusted_return*10000-(o.metadata->>'estimated_round_trip_cost_bps')::double precision)
   from outcomes o join decisions d using(decision_id) where d.action in ('OPEN_LONG','OPEN_SHORT')),
 'learning_is_observation_not_causal_proof',true,'shadow_only',true,'live_execution',false)
$fn$;
revoke all on function public.brian_learning_window_v1(timestamptz) from public,anon,authenticated;
grant execute on function public.brian_learning_window_v1(timestamptz) to service_role;
insert into public.brian_evolution_runtime_config(config_key,config_value,updated_at)
values('brian_open_world_research_enabled','true',clock_timestamp())
on conflict(config_key) do update set config_value=excluded.config_value,updated_at=excluded.updated_at;

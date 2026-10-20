-- BRIAN Research Lab ledger (append-only): papers, atomic claims, assumptions, and test results.
--
-- Fills the one missing link in the existing research chain (hypothesis -> experiment -> result ->
-- decision live in brian_evolution_*): what a paper actually claims, which unstated choices we
-- made to test it, and every outcome of every test — including failures. Source of truth for
-- experiment code and raw results is the repository (research/research_lab/); these tables index
-- them so the UI can show status. Nothing here schedules, executes or trades anything.

-- Requires public.brian_reject_mutation() (created by earlier BRIAN migrations).

create table if not exists public.brian_research_papers (
  paper_id text primary key check (paper_id ~ '^(arxiv|doi|url):[A-Za-z0-9._/:#-]{3,200}$'),
  title text not null check (length(title) between 3 and 500),
  authors text[] not null default '{}',
  url text not null check (url ~ '^https://'),
  license text,
  code_url text check (code_url is null or code_url ~ '^https://'),
  code_commit text check (code_commit is null or code_commit ~ '^[0-9a-f]{7,40}$'),
  data_cutoff timestamptz,
  missing_info text[] not null default '{}',
  ledger_path text not null check (ledger_path ~ '^research/research_lab/ledger/[A-Za-z0-9._-]+\.json$'),
  ledger_sha256 text not null check (ledger_sha256 ~ '^[0-9a-f]{64}$'),
  created_at timestamptz not null default now()
);

create table if not exists public.brian_research_claims (
  claim_id text primary key check (claim_id ~ '^(arxiv|doi|url):[A-Za-z0-9._/:-]{3,200}#C-[0-9]{2,3}$'),
  paper_id text not null references public.brian_research_papers(paper_id),
  tier text not null check (tier in ('central', 'supporting', 'ancillary')),
  statement text not null check (length(statement) between 10 and 1000),
  metric text,
  dataset text,
  reported jsonb not null default '{}'::jsonb check (jsonb_typeof(reported) = 'object'),
  comparison jsonb check (comparison is null or jsonb_typeof(comparison) = 'object'),
  repro_risk text not null check (repro_risk in ('low', 'medium', 'high', 'blocked')),
  depends_on text[] not null default '{}',
  notes text,
  created_at timestamptz not null default now(),
  check (split_part(claim_id, '#', 1) = paper_id)
);

create table if not exists public.brian_research_assumptions (
  paper_id text not null references public.brian_research_papers(paper_id),
  assumption_id text not null check (assumption_id ~ '^A-[0-9]{2,3}$'),
  claim_id text references public.brian_research_claims(claim_id),
  leverage text not null check (leverage in ('low', 'medium', 'high')),
  detail text not null,
  chosen text not null,
  rationale text not null,
  created_at timestamptz not null default now(),
  primary key (paper_id, assumption_id)
);

create table if not exists public.brian_research_claim_results (
  result_id text primary key check (result_id ~ '^[A-Za-z0-9._:-]{8,120}$'),
  claim_id text not null references public.brian_research_claims(claim_id),
  experiment_id text not null check (experiment_id ~ '^[a-z0-9._-]{3,80}$'),
  status text not null check (status in ('NOT_STARTED', 'RUNNING', 'INSUFFICIENT_DATA', 'PASS', 'FAIL', 'UNVERIFIED', 'REVIEW_REQUIRED')),
  status_reason text not null check (length(status_reason) between 3 and 500),
  observed_at timestamptz not null,
  code_commit text not null check (code_commit ~ '^[0-9a-f]{7,40}$'),
  run_url text check (run_url is null or run_url ~ '^https://'),
  dataset_sha256 text check (dataset_sha256 is null or dataset_sha256 ~ '^[0-9a-f]{64}$'),
  evidence_class text not null check (evidence_class in ('OUT_OF_SAMPLE_ADAPTED_REPLICATION', 'OUT_OF_SAMPLE_REPLICATION', 'IN_SAMPLE_REPRODUCTION', 'DEMO')),
  metrics jsonb not null default '{}'::jsonb check (jsonb_typeof(metrics) = 'object' and octet_length(metrics::text) <= 65536),
  trade_signal boolean not null default false check (not trade_signal),
  shadow_only boolean not null default true check (shadow_only),
  created_at timestamptz not null default now()
);
create index if not exists brian_research_claim_results_claim_idx
  on public.brian_research_claim_results(claim_id, observed_at desc, result_id desc);

do $$
declare t text;
begin
  foreach t in array array['brian_research_papers', 'brian_research_claims', 'brian_research_assumptions', 'brian_research_claim_results'] loop
    execute format('alter table public.%I enable row level security', t);
    execute format('revoke all on public.%I from public, anon, authenticated, service_role', t);
    execute format('grant select on public.%I to service_role', t);
    execute format('drop trigger if exists %I on public.%I', t || '_append_only', t);
    execute format('create trigger %I before update or delete on public.%I for each row execute function public.brian_reject_mutation()', t || '_append_only', t);
  end loop;
end $$;

-- Register a ledger file (paper + claims + assumptions). Idempotent for identical content; a
-- changed paper/claim/assumption under an existing id is REJECTED (ids are immutable — publish a
-- new id instead), so history can never be rewritten silently.
create or replace function public.brian_record_research_ledger_v1(p_ledger jsonb, p_ledger_path text, p_ledger_sha256 text)
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  v_paper jsonb := p_ledger->'paper';
  v_pid text := v_paper->>'paper_id';
  v_c jsonb;
  v_a jsonb;
  v_new_claims integer := 0;
  v_new_assumptions integer := 0;
  v_existing record;
begin
  if p_ledger->>'schema' is distinct from 'brian.research-ledger.v1' then
    return jsonb_build_object('status', 'REJECTED', 'reason', 'UNKNOWN_SCHEMA');
  end if;
  select * into v_existing from public.brian_research_papers where paper_id = v_pid;
  if found then
    if v_existing.title <> v_paper->>'title' or v_existing.url <> v_paper->>'url' then
      return jsonb_build_object('status', 'REJECTED', 'reason', 'PAPER_ID_CONTENT_CHANGED');
    end if;
  else
    insert into public.brian_research_papers (paper_id, title, authors, url, license, code_url, code_commit, data_cutoff, missing_info, ledger_path, ledger_sha256)
    values (v_pid, v_paper->>'title',
            coalesce(array(select jsonb_array_elements_text(v_paper->'authors')), '{}'),
            v_paper->>'url', v_paper->>'license', v_paper->'code'->>'url', v_paper->'code'->>'commit',
            (v_paper->>'data_cutoff')::timestamptz,
            coalesce(array(select jsonb_array_elements_text(v_paper->'missing_info')), '{}'),
            p_ledger_path, p_ledger_sha256);
  end if;

  for v_c in select * from jsonb_array_elements(coalesce(p_ledger->'claims', '[]'::jsonb)) loop
    if exists (select 1 from public.brian_research_claims where claim_id = v_c->>'claim_id') then
      if not exists (select 1 from public.brian_research_claims where claim_id = v_c->>'claim_id' and statement = v_c->>'statement') then
        raise exception 'BRIAN_RESEARCH: claim % changed under the same id', v_c->>'claim_id';
      end if;
      continue;
    end if;
    insert into public.brian_research_claims (claim_id, paper_id, tier, statement, metric, dataset, reported, comparison, repro_risk, depends_on, notes)
    values (v_c->>'claim_id', v_pid, v_c->>'tier', v_c->>'statement', v_c->>'metric', v_c->>'dataset',
            coalesce(v_c->'reported', '{}'::jsonb), v_c->'comparison', v_c->>'repro_risk',
            coalesce(array(select jsonb_array_elements_text(v_c->'depends_on')), '{}'), v_c->>'notes');
    v_new_claims := v_new_claims + 1;
  end loop;

  for v_a in select * from jsonb_array_elements(coalesce(p_ledger->'assumptions', '[]'::jsonb)) loop
    if exists (select 1 from public.brian_research_assumptions where paper_id = v_pid and assumption_id = v_a->>'assumption_id') then
      if not exists (select 1 from public.brian_research_assumptions where paper_id = v_pid and assumption_id = v_a->>'assumption_id' and chosen = v_a->>'chosen') then
        raise exception 'BRIAN_RESEARCH: assumption % changed under the same id', v_a->>'assumption_id';
      end if;
      continue;
    end if;
    insert into public.brian_research_assumptions (paper_id, assumption_id, claim_id, leverage, detail, chosen, rationale)
    values (v_pid, v_a->>'assumption_id', v_a->>'claim_id', v_a->>'leverage', v_a->>'detail', v_a->>'chosen', v_a->>'rationale');
    v_new_assumptions := v_new_assumptions + 1;
  end loop;

  return jsonb_build_object('status', 'STORED', 'paper_id', v_pid, 'new_claims', v_new_claims, 'new_assumptions', v_new_assumptions);
end;
$$;

-- Record one experiment outcome for a claim. Every run is kept (PASS and FAIL alike).
create or replace function public.brian_record_research_result_v1(p_result jsonb)
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
begin
  if coalesce(p_result->>'trade_signal', 'false') <> 'false' or coalesce(p_result->>'shadow_only', 'true') <> 'true' then
    return jsonb_build_object('status', 'REJECTED', 'reason', 'NOT_SHADOW_ONLY');
  end if;
  insert into public.brian_research_claim_results (result_id, claim_id, experiment_id, status, status_reason, observed_at, code_commit, run_url, dataset_sha256, evidence_class, metrics)
  values (p_result->>'result_id', p_result->>'claim_id', p_result->>'experiment_id', p_result->>'status', p_result->>'status_reason',
          (p_result->>'observed_at')::timestamptz, p_result->>'code_commit', p_result->>'run_url', p_result->>'dataset_sha256',
          p_result->>'evidence_class', coalesce(p_result->'metrics', '{}'::jsonb))
  on conflict (result_id) do nothing;
  return jsonb_build_object('status', case when found then 'STORED' else 'DUPLICATE' end, 'result_id', p_result->>'result_id');
end;
$$;

-- Read-only status for the UI: every claim with its latest result (or NOT_STARTED), counts by
-- status, and the assumptions behind it. Failures are listed exactly like successes.
create or replace function public.brian_research_status_v1(p_as_of timestamptz default now())
returns jsonb
language sql
stable
security definer
set search_path = pg_catalog, public
as $$
  with latest as (
    select distinct on (r.claim_id) r.*
    from public.brian_research_claim_results r
    where r.observed_at <= p_as_of
    order by r.claim_id, r.observed_at desc, r.result_id desc
  ),
  claims as (
    select c.claim_id, c.paper_id, c.tier, c.statement, c.metric, c.repro_risk, c.reported,
           coalesce(l.status, 'NOT_STARTED') as status,
           l.status_reason, l.observed_at as result_at, l.experiment_id, l.code_commit, l.run_url,
           l.evidence_class, l.metrics,
           (select count(*) from public.brian_research_claim_results x where x.claim_id = c.claim_id and x.observed_at <= p_as_of) as runs
    from public.brian_research_claims c
    left join latest l on l.claim_id = c.claim_id
    where c.created_at <= p_as_of
  )
  select jsonb_build_object(
    'as_of', p_as_of,
    'status', case when exists (select 1 from claims) then 'OK' else 'NO_DATA' end,
    'papers', coalesce((select jsonb_agg(jsonb_build_object(
        'paper_id', p.paper_id, 'title', p.title, 'authors', to_jsonb(p.authors), 'url', p.url, 'license', p.license,
        'code_url', p.code_url, 'code_commit', p.code_commit, 'data_cutoff', p.data_cutoff,
        'missing_info', to_jsonb(p.missing_info), 'ledger_path', p.ledger_path, 'ledger_sha256', p.ledger_sha256,
        'assumptions', coalesce((select jsonb_agg(jsonb_build_object('assumption_id', a.assumption_id, 'claim_id', a.claim_id,
            'leverage', a.leverage, 'detail', a.detail, 'chosen', a.chosen, 'rationale', a.rationale) order by a.assumption_id)
          from public.brian_research_assumptions a where a.paper_id = p.paper_id), '[]'::jsonb)
      ) order by p.created_at, p.paper_id) from public.brian_research_papers p where p.created_at <= p_as_of), '[]'::jsonb),
    'claims', coalesce((select jsonb_agg(to_jsonb(c) order by c.paper_id, c.claim_id) from claims c), '[]'::jsonb),
    'counts', coalesce((select jsonb_object_agg(status, n) from (select status, count(*) n from claims group by status) s), '{}'::jsonb),
    'shadow_only', true, 'trade_signal', false
  );
$$;

revoke all on function public.brian_record_research_ledger_v1(jsonb, text, text) from public, anon, authenticated;
revoke all on function public.brian_record_research_result_v1(jsonb) from public, anon, authenticated;
revoke all on function public.brian_research_status_v1(timestamptz) from public, anon, authenticated;
grant execute on function public.brian_research_status_v1(timestamptz) to service_role;
-- Recording is an operator action (SQL editor / MCP as postgres), not an API surface:
-- service_role gets no EXECUTE on the two writers.

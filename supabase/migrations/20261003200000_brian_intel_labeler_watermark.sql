-- Target: brian-market-intelligence. The rules-v2 labeler anti-joined all ~91k events against
-- all labels every 5 minutes (~24 s per call, 3.4M blocks read) on a t3a.nano that was already
-- at 100% disk IO. Label only events observed after the newest labelled one (1 h overlap for
-- late commits), and run every 15 minutes.
create index if not exists brian_intel_event_labels_version_time_idx
  on public.brian_intel_event_labels(labeler_version, observed_at desc);

create or replace function public.brian_label_intel_events_v2(p_limit int default 5000)
returns integer language plpgsql set search_path = pg_catalog, public as $$
declare inserted int; watermark timestamptz;
begin
  select max(observed_at) - interval '1 hour' into watermark
    from public.brian_intel_event_labels where labeler_version = 'rules-v2';
  insert into public.brian_intel_event_labels(event_id, labeler_version, observed_at, noise, theme, assets, pos_hits, neg_hits, direction)
  select e.event_id, 'rules-v2', e.first_observed_at, r.noise, r.theme, r.assets, r.pos_hits, r.neg_hits, r.direction
  from (select event_id, first_observed_at, claim from public.brian_intel_events ev
        where ev.first_observed_at > coalesce(watermark, '-infinity'::timestamptz)
          and not exists (select 1 from public.brian_intel_event_labels l where l.event_id = ev.event_id and l.labeler_version = 'rules-v2')
        order by first_observed_at limit greatest(1, least(p_limit, 20000))) e
  cross join lateral public.brian_intel_label_rules_v2(e.claim) r
  on conflict do nothing;
  get diagnostics inserted = row_count;
  return inserted;
end $$;

select cron.alter_job(job_id := (select jobid from cron.job where jobname = 'brian-intel-label-v2-5m'), schedule := '7,22,37,52 * * * *');

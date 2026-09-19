-- Repair long-tail next checkpoints if an older reaction worker cleared them during rollout.
update public.brian_catalyst_sentinel_watches
set next_recheck_at = case recheck_count
  when 0 then started_at + interval '5 minutes'
  when 1 then started_at + interval '10 minutes'
  when 2 then started_at + interval '15 minutes'
  when 3 then started_at + interval '30 minutes'
  when 4 then started_at + interval '120 minutes'
  when 5 then started_at + interval '360 minutes'
  when 6 then started_at + interval '720 minutes'
  else null
end,
updated_at=now()
where expires_at>now()
  and metadata->>'long_tail_watch'='true'
  and next_recheck_at is null
  and recheck_count between 0 and 6;

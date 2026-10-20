# Event rewind and decision-source inspection — isolated preview

Clicking a real event or journey record opens a nonmodal history drawer. The earlier electron artwork, core arrival ripple and live read-only polling are retained. The main scene explicitly labels historical notification playback. The drawer has observed-version selection, back/next steps, play/pause, a seek slider, rewind-to-start and exact source JSON.

History is bounded to 180 observed record versions in this tab's memory. It is not a server audit log and is reset on page reload. Each version preserves its original raw fields, record timestamp, public response time, local first observation and actual observed notification arrival (when present). Initial records and imports do not fabricate a core arrival. Public polling continues while reviewing a frozen version. New versions are added to the selector without silently replacing the selected version.

Rationale inspection uses only fields actually supplied in the preserved record and uniquely matched explicit record references. A simultaneous news record and ALPHA decision are not treated as causally linked. Non-decision source thresholds are labeled as source-state fields, not trading reasons.

Material source limitation verified in the live public ALPHA envelope: action=WAIT, asset_id, direction, observed_at, evidence_score and a null estimated_round_trip_cost_bps were present. No explanatory reason, veto cause, decision threshold or evidence-source links were supplied. The drawer therefore says the reason for WAIT cannot be verified from this response. No backend reason is invented or exposed through a new endpoint. Full decision explanation needs those fields to be provided by the source in a future authorized backend change.

Browser verification: real news versions 23:03:00 and 23:04:03 were retained separately; the second was observed at 23:04:54 with a core notification at 23:04:57. Three initial-observation steps and four observed-arrival steps were shown correctly. Opening ALPHA displayed its actual WAIT and evidence_score=0.0481699293155155. Back/start/play controls worked; no WebGL errors were reported.

Six new history tests passed; 18 history/journey/atmosphere tests passed in total. Fullscreen benchmark with review open: 60.0372 FPS at 6108 × 1374 drawing resolution; p95 16.8 ms; mean JS 2.585 ms and GPU draw 1.316 ms. These are work durations, not device utilization percentages. No production deployment, experiment or execution occurred.

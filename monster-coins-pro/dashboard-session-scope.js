/* Brian dashboard session-window view patch.
   Historical DB evidence remains append-only. This only scopes what the current tracking
   session shows. PAUSE freezes the window; Start after PAUSE resumes the same window. */
(() => {
  const inWindow = (value, startMs, endMs) => {
    const t = Date.parse(String(value || ''));
    return Number.isFinite(t) && t >= startMs && t <= endMs;
  };
  const scopedData = (data) => {
    const s = data?.session, a = data?.alpha_v2;
    if (!s?.started_at || !a) return data;
    const startMs = Date.parse(s.started_at);
    if (!Number.isFinite(startMs)) return data;
    const pausedEnd = s.status === 'PAUSED' && s.ended_at ? Date.parse(s.ended_at) : Infinity;
    const endMs = Number.isFinite(pausedEnd) ? pausedEnd : Infinity;
    return {
      ...data,
      alpha_v2: {
        ...a,
        decisions: (a.decisions || []).filter((x) => inWindow(x.observed_at, startMs, endMs)),
        positions: (a.positions || []).filter((x) => inWindow(x.last_action_at || x.entry_ts, startMs, endMs)),
        phase37_comparisons: (a.phase37_comparisons || []).filter((x) => inWindow(x.observed_at, startMs, endMs)),
        outcomes: (a.outcomes || []).filter((x) => inWindow(x.observed_at, startMs, endMs)),
        costs: (a.costs || []).filter((x) => inWindow(x.observed_at, startMs, endMs)),
        session_scoped: true,
        session_started_at: s.started_at,
        session_ended_at: s.status === 'PAUSED' ? (s.ended_at || null) : null,
      },
    };
  };

  const originalOverview = renderAlphaOverview;
  renderAlphaOverview = function(data) {
    return originalOverview(scopedData(data));
  };

  const originalAlpha = renderAlpha;
  renderAlpha = function(data) {
    return originalAlpha(scopedData(data));
  };

  const originalSession = renderSession;
  renderSession = function(data) {
    const result = originalSession(data);
    const s = data?.session;
    if ($('startBtn')) {
      $('startBtn').textContent = s?.status === 'PAUSED' ? '▶ Takibe Devam' : '▶ Takibi Başlat';
    }
    if ($('restartBtn')) $('restartBtn').textContent = '↻ Yeni Oturum';
    return result;
  };
})();

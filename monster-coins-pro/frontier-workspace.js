'use strict';
/* Navigation to the existing Brian rooms. No collector, experiment, or control mutation. */
(() => {
  const $ = id => document.getElementById(id);
  if (!$('brianOperations') || $('workspaceNav')) return;
  const style = document.createElement('link');
  style.rel = 'stylesheet'; style.href = '/frontier-workspace.css?v=20261020-ui1';
  document.head.appendChild(style);
  const nav = document.createElement('nav');
  nav.id = 'workspaceNav'; nav.setAttribute('aria-label', 'Brian çalışma alanları');
  nav.innerHTML = '<a href="#opsOverview">Genel bakış</a><a href="#opsFlow">Haberler</a>' +
    '<button type="button" data-workspace-room="meeting" data-ops-room="meeting">Toplantı</button>' +
    '<button type="button" data-workspace-room="accounting">Muhasebe</button>' +
    '<button type="button" data-workspace-room="engineering">Yazılım</button>' +
    '<button type="button" data-workspace-room="anatomy">Gelişim</button>' +
    '<a href="#predictionResearch">Araştırma</a><a href="#shadowScorecard">Performans</a>' +
    '<a href="#opsControl">Sistem</a>';
  document.querySelector('.ticker')?.after(nav);
  const status = document.createElement('section');
  status.id = 'workspaceStatus'; status.setAttribute('aria-label', 'Veri bağlantısı');
  status.innerHTML = '<span id="workspaceConnection">Bağlantı doğrulanıyor</span>' +
    '<button type="button" id="workspaceRefresh">Verileri yenile</button>' +
    '<p id="workspaceFeedback" role="status" aria-live="polite"></p>';
  $('opsOverview').prepend(status);
  function linkNewsSources() {
    const rows = typeof news === 'function' ? news() : [];
    const cards = $('criticalNews')?.querySelectorAll('.news') || [];
    cards.forEach((card, i) => {
      if (card.querySelector('.workspace-source') || !rows[i]?.uri) return;
      let url;
      try { url = new URL(rows[i].uri); } catch { return; }
      if (!['https:', 'http:'].includes(url.protocol) || url.username || url.password) return;
      const a = document.createElement('a');
      a.className = 'workspace-source'; a.href = url.href; a.target = '_blank'; a.rel = 'noopener noreferrer';
      a.textContent = 'Kaynağı aç ↗'; a.setAttribute('aria-label', 'Haber kaynağını aç: ' + (rows[i].title || url.hostname));
      card.appendChild(a);
    });
  }
  if (typeof renderNews === 'function') {
    const original = renderNews;
    renderNews = function () { original.apply(this, arguments); linkNewsSources(); };
  }
  linkNewsSources();
  const handlers = {
    meeting: () => $('openMeeting')?.click(),
    accounting: () => window.openTreasuryAccounting?.(),
    engineering: () => window.openBrianEngineeringRoom?.(),
    anatomy: () => window.openBrianAnatomy?.()
  };
  nav.addEventListener('click', e => {
    const button = e.target.closest('[data-workspace-room]');
    if (button) { handlers[button.dataset.workspaceRoom]?.(); return; }
    const a = e.target.closest('a[href^="#"]');
    if (!a) return;
    const target = $(a.hash.slice(1));
    if (!target) { e.preventDefault(); $('workspaceFeedback').textContent = 'Bu alan henüz yüklenemedi.'; return; }
    e.preventDefault();
    for (let n = target; n; n = n.parentElement) if (n.tagName === 'DETAILS') n.open = true;
    target.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth', block: 'start' });
    target.setAttribute('tabindex', '-1'); target.focus({ preventScroll: true });
  });
  let refreshing = false;
  $('workspaceRefresh').onclick = async () => {
    if (refreshing) return;
    if (typeof key !== 'function' || !key()) { $('workspaceFeedback').textContent = 'Verileri almak için dashboard kilidini aç.'; return; }
    refreshing = true; $('workspaceRefresh').disabled = true;
    $('workspaceFeedback').textContent = 'Mevcut durum kayıtları yenileniyor…';
    try {
      await Promise.allSettled([
        typeof refresh === 'function' ? refresh() : Promise.reject(),
        typeof refreshAutonomyV4 === 'function' ? refreshAutonomyV4() : Promise.reject(),
        typeof refreshEngineerConsole === 'function' ? refreshEngineerConsole() : Promise.reject()
      ]);
      // Existing refreshers retain errors internally; do not label completion as success.
      $('workspaceFeedback').textContent = 'Yenileme tamamlandı. Bağlantı ve hata durumlarını ilgili alandan kontrol et.';
    } finally { refreshing = false; $('workspaceRefresh').disabled = false; update(); }
  };
  function update() {
    if (document.hidden) return;
    const names = { meeting: !!$('openMeeting'), accounting: typeof window.openTreasuryAccounting === 'function', engineering: typeof window.openBrianEngineeringRoom === 'function', anatomy: typeof window.openBrianAnatomy === 'function' };
    for (const b of nav.querySelectorAll('[data-workspace-room]')) {
      b.disabled = !names[b.dataset.workspaceRoom];
      b.title = b.disabled ? 'Oda yüklenemedi; sayfayı yenileyin.' : '';
    }
    const hb = typeof STABILITY !== 'undefined' ? STABILITY.heartbeat : null;
    const error = typeof STABILITY !== 'undefined' ? STABILITY.error : null;
    const stamp = Date.parse(hb?.observed_at || '');
    const age = Date.now() - stamp;
    const recent = Number.isFinite(stamp) && age >= 0 && age <= 120000;
    const state = !hb ? 'unknown' : error || !recent || hb.__cached ? 'warn' : 'ok';
    status.dataset.state = state;
    $('workspaceConnection').textContent = (state === 'ok' ? 'Durum bağlantısı güncel' : state === 'warn' ? 'Son kayıt gösteriliyor · bağlantıyı kontrol et' : 'Canlı durum henüz alınmadı') +
      (Number.isFinite(stamp) ? ' · ' + new Date(stamp).toLocaleString('tr-TR', { timeZone: 'Europe/Berlin' }) : '');
  }
  // Optional deep link opens an existing room; never starts a run or a control action.
  const requestedRoom = new URLSearchParams(location.search).get('room');
  let roomOpened = false;
  const sections = { research: 'predictionResearch', performance: 'shadowScorecard', news: 'opsFlow', system: 'opsControl' };
  const announce = state => { if (parent !== window) parent.postMessage({ type: 'brian-room-status', room: requestedRoom, state }, location.origin); };
  function openRequestedRoom() {
    if (roomOpened || !requestedRoom) return;
    if (typeof key !== 'function' || !key()) { announce('locked'); return; }
    const actions = { meeting: () => $('openMeeting')?.click(), accounting: () => window.openTreasuryAccounting?.(), engineering: () => window.openBrianEngineeringRoom?.(), anatomy: () => window.openBrianAnatomy?.() };
    const available = { meeting: !!$('openMeeting'), accounting: typeof window.openTreasuryAccounting === 'function', engineering: typeof window.openBrianEngineeringRoom === 'function', anatomy: typeof window.openBrianAnatomy === 'function' };
    try {
      if (actions[requestedRoom]) { if (!available[requestedRoom]) return; actions[requestedRoom](); }
      else { const target = $(sections[requestedRoom]); if (!target) return; for (let n = target; n; n = n.parentElement) if (n.tagName === 'DETAILS') n.open = true; target.scrollIntoView({ block: 'start' }); target.setAttribute('tabindex', '-1'); target.focus({ preventScroll: true }); }
      roomOpened = true; announce('ready');
    } catch { announce('error'); }
  }
  openRequestedRoom(); setInterval(openRequestedRoom, 1000);
  update(); setInterval(update, 3000);
  document.addEventListener('visibilitychange', update);
})();

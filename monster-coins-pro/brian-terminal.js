'use strict';
/* Presentation over the existing BRIAN records. No new service, worker or experiment. */
(() => {
  const $ = id => document.getElementById(id);
  if (!$('workspaceNav') || $('terminalMarkets')) return;
  document.body.classList.add('brian-terminal');
  const css = document.createElement('link');
  css.rel = 'stylesheet'; css.href = '/brian-terminal.css?v=20261020-terminal3';
  document.head.appendChild(css);
  const E = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const icons = {
    overview:'M3 3h7v7H3zM14 3h7v7h-7zM3 14h7v7H3zM14 14h7v7h-7z',
    news:'M4 3h13v18H4zM17 8h4v13h-4M7 7h7M7 11h7M7 15h7',
    meeting:'M4 4h16v12H8l-4 4zM8 8h8M8 12h5',
    accounting:'M3 7h18v14H3zM3 7l3-4h12l3 4M15 12h6v5h-6z',
    engineering:'M8 5l-6 7 6 7M16 5l6 7-6 7M14 3l-4 18',
    anatomy:'M12 3v18M3 12h18M5 5l14 14M19 5L5 19',
    research:'M8 3h8M10 3v7l-6 10h16l-6-10V3M8 15h8',
    performance:'M4 21V11h4v10M10 21V6h4v15M16 21V2h4v19',
    system:'M9 3h6l1 4 4 2v6l-4 2-1 4H9l-1-4-4-2V9l4-2zM12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6',
    globe:'M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0M3 12h18M12 3c-5 5-5 13 0 18 5-5 5-13 0-18'
  };
  const icon = name => `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="${icons[name] || icons.overview}"/></svg>`;
  const keys = ['overview','news','meeting','accounting','engineering','anatomy','research','performance','system'];
  [...$('workspaceNav').children].forEach((el, i) => el.insertAdjacentHTML('afterbegin', icon(keys[i])));
  $('workspaceNav').querySelector('a[href="#opsFlow"]').href='#worldPanel';
  $('workspaceNav').querySelector('a').setAttribute('aria-current','location');
  $('workspaceNav').addEventListener('click', e => {const a=e.target.closest('a');if(!a)return;for(const link of $('workspaceNav').querySelectorAll('a'))link.removeAttribute('aria-current');a.setAttribute('aria-current','location');});
  const brand = document.querySelector('.topbar .brand');
  if (brand) {
    brand.querySelector('img').src = '/brian-mark.svg';
    brand.querySelector('strong').textContent = 'BRIAN';
    const tagline = document.createElement('span'); tagline.className = 'terminal-tagline';
    tagline.textContent = 'FİNANSAL ZEKÂ MERKEZİ';
    brand.querySelector('strong').after(tagline);
  }
  const clock = document.createElement('time'); clock.id = 'terminalClock';
  document.querySelector('.top-actions')?.prepend(clock);
  const clockTick = () => { clock.dateTime = new Date().toISOString(); clock.textContent = new Intl.DateTimeFormat('tr-TR', {timeZone:'Europe/Berlin',day:'2-digit',month:'short',hour:'2-digit',minute:'2-digit'}).format(new Date()); };
  clockTick(); setInterval(clockTick, 30000);

  const markets = document.createElement('section'); markets.id = 'terminalMarkets';
  markets.setAttribute('aria-label', 'Piyasa radarı · 24 saatlik değişimler');
  markets.innerHTML = '<div class="terminal-market-label">'+icon('performance')+'<b>PİYASA RADARI</b><small id="terminalMarketState">Veri bekleniyor</small></div><div id="terminalQuotes"><p>Doğrulanmış radar kaydı bekleniyor.</p></div>';
  $('workspaceNav').after(markets);
  let marketSignature = '';
  function renderMarkets() {
    if (document.hidden) return;
    const r = typeof S !== 'undefined' ? S.control?.market_radar || S.control?.alpha_v2?.market_radar : null;
    const {rows,current,label,at:stamp}=window.BrianTerminalModel.radar(r);
    markets.dataset.state = current ? 'ok' : 'warn';
    $('terminalMarketState').textContent = label;
    $('terminalMarketState').title = stamp!==null ? 'Kaynak zamanı: '+new Date(stamp).toLocaleString('tr-TR',{timeZone:'Europe/Berlin'}) : 'Kaynak zamanı bilinmiyor';
    const signature = JSON.stringify([rows.slice(0,8),label]);
    if (signature === marketSignature) return;
    marketSignature = signature;
    $('terminalQuotes').innerHTML = rows.length ? rows.slice(0,8).map(row => {
      const v = row.change,valid=v!==null;
      return `<a class="terminal-quote" href="#marketRadarPanel"><span class="terminal-coin" aria-hidden="true">${E(row.letter)}</span><span><b>${E(row.symbol)}</b><small>${current?'24 saatlik değişim':'Son radar · eski kayıt'}</small></span><strong data-tone="${valid ? v >= 0 ? 'up' : 'down' : 'unknown'}">${valid ? (v > 0 ? '+' : '') + v.toLocaleString('tr-TR',{maximumFractionDigits:2}) + '%' : '—'}</strong></a>`;
    }).join('') : '<p>Radar kaydı bekleniyor.</p>';
  }
  renderMarkets(); setInterval(renderMarkets, 5000);
  const radar=$('marketRadarPanel');if(radar)document.querySelector('main.shell')?.appendChild(radar);

  // Native SVG geography is an orientation graphic. It never assigns a news event to a location.
  function globe() {
    const project = ([lon,lat]) => { const l=(lon-20)*Math.PI/180,p=lat*Math.PI/180,c=12*Math.PI/180; return [250+177*Math.cos(p)*Math.sin(l),210-177*(Math.cos(c)*Math.sin(p)-Math.sin(c)*Math.cos(p)*Math.cos(l)),Math.sin(c)*Math.sin(p)+Math.cos(c)*Math.cos(p)*Math.cos(l)]; };
    const path = points => points.map((point,i) => {const p=project(point);return (i?'L':'M')+p[0].toFixed(1)+','+p[1].toFixed(1);}).join(' ');
    const continents = [
      [[-17,37],[-5,36],[10,37],[25,32],[34,31],[35,24],[44,12],[51,11],[43,0],[40,-12],[34,-26],[19,-35],[12,-20],[9,-2],[-4,5],[-15,12],[-17,25]],
      [[-10,36],[-9,44],[-1,49],[5,50],[8,55],[18,55],[20,61],[29,70],[32,61],[44,57],[48,47],[39,41],[28,41],[24,35],[18,40],[12,43],[8,39],[2,43]],
      [[29,70],[56,72],[87,73],[113,66],[140,60],[162,63],[178,52],[146,48],[136,35],[122,24],[112,21],[108,8],[100,5],[98,20],[89,22],[80,8],[73,20],[66,25],[56,25],[50,15],[43,12],[35,30],[40,40],[48,47],[44,57]],
      [[-81,12],[-68,10],[-61,5],[-51,2],[-35,-7],[-39,-20],[-50,-30],[-61,-52],[-70,-55],[-75,-40],[-72,-18],[-80,-3]],
      [[-168,68],[-140,70],[-126,60],[-125,48],[-117,33],[-104,24],[-97,16],[-86,10],[-80,9],[-84,21],[-81,26],[-75,35],[-64,48],[-53,50],[-64,60],[-90,70],[-118,73]],
      [[-53,60],[-45,61],[-25,73],[-35,83],[-55,81],[-65,68]],
      [[113,-22],[122,-13],[136,-12],[143,-11],[153,-26],[147,-39],[131,-33],[115,-35]],
      [[47,-13],[50,-17],[47,-25],[44,-24]], [[-8,50],[-2,51],[0,56],[-5,59],[-7,55]]
    ];
    let grid='';
    for(let lat=-60;lat<=60;lat+=20){ const points=[];for(let lon=-180;lon<=180;lon+=3){const p=project([lon,lat]);if(p[2]>=0)points.push([lon,lat]);}grid+=`<path d="${path(points)}"/>`; }
    for(let lon=-60;lon<=100;lon+=20){const points=[];for(let lat=-89;lat<=89;lat+=3)points.push([lon,lat]);grid+=`<path d="${path(points)}"/>`;}
    const land=continents.map(points=>`<path d="${path(points.filter(p=>project(p)[2]>-.04))}Z"/>`).join('');
    return `<svg class="terminal-globe" viewBox="0 0 500 420" role="img" aria-label="Dünya küresi · dekoratif coğrafya, olay konumu göstermez"><defs><radialGradient id="tgOcean" cx="38%" cy="30%"><stop stop-color="#083d66"/><stop offset=".7" stop-color="#03233f"/><stop offset="1" stop-color="#010b18"/></radialGradient><radialGradient id="tgHalo"><stop offset=".72" stop-color="#009dff" stop-opacity="0"/><stop offset=".83" stop-color="#008aff" stop-opacity=".23"/><stop offset="1" stop-color="#008aff" stop-opacity="0"/></radialGradient><clipPath id="tgClip"><circle cx="250" cy="210" r="177"/></clipPath></defs><circle cx="250" cy="210" r="209" fill="url(#tgHalo)"/><ellipse cx="250" cy="210" rx="222" ry="83" transform="rotate(-27 250 210)" fill="none" stroke="#04aefa" stroke-opacity=".5"/><ellipse cx="250" cy="210" rx="222" ry="100" transform="rotate(28 250 210)" fill="none" stroke="#006bba" stroke-dasharray="3 7"/><circle cx="250" cy="210" r="177" fill="url(#tgOcean)" stroke="#009deb" stroke-width="1.4"/><g clip-path="url(#tgClip)"><g class="tg-grid">${grid}</g><g class="tg-land">${land}</g><path d="M82 167Q250 40 411 206M100 286Q260 132 401 279M173 54Q360 180 200 380" fill="none" stroke="#36c9ff" stroke-opacity=".35"/></g><circle cx="250" cy="210" r="184" fill="none" stroke="#1184bd" stroke-dasharray="1 8"/><g fill="#59dfff"><circle cx="207" cy="110" r="3"/><circle cx="315" cy="174" r="3"/><circle cx="210" cy="245" r="3"/><circle cx="356" cy="237" r="3"/></g></svg>`;
  }
  const core=document.querySelector('#command .nc-core');
  if(core){core.querySelector('.nc-brain')?.remove();core.insertAdjacentHTML('afterbegin',globe());}
  const heading=document.querySelector('#command .nc-head h2'); if(heading)heading.textContent='Küresel istihbarat ağı';
  const kicker=document.querySelector('#command .nc-kicker');if(kicker)kicker.textContent='BRIAN / BAĞLANTILI SİSTEMLER';
  const center=document.querySelector('#command .nc-center-sub');if(center)center.textContent='VERİ · ANALİZ · ÖNGÖRÜ';
  const note=document.querySelector('#command .nc-note');if(note)note.textContent='Küre ve coğrafya görseldir. Modül durumları gerçek kayıtlardan gelir; ışık yalnız yeni kanıtta akar. Güncellik, kârlılık değildir.';
  document.querySelectorAll('#command [data-module] i').forEach(el=>{const k=el.parentElement.dataset.module;el.innerHTML=icon(({world:'globe',behavior:'anatomy',alpha:'performance',treasury:'accounting',research:'research',ocean:'system'})[k]);});
  const title=(selector,text)=>{const el=document.querySelector(selector);if(el)el.textContent=text;};
  title('#opsFlow > .ops-heading h2','Canlı haber akışı');
  title('#opsRooms > .ops-heading h2','Çalışma odaları');
  title('#opsOverview .ops-heading h2','Hazine & sistem özeti');
  title('#worldPanel .section-title','CANLI HABER AKIŞI');
  title('#predictionResearch h3','Araştırma & analiz');
  title('#command .nc-center-label','BRIAN');
  const filter=document.createElement('label');filter.className='terminal-news-search';
  filter.innerHTML=icon('news')+'<input type="search" id="terminalNewsSearch" placeholder="Haberlerde ara…" aria-label="Görünür haberlerde ara"><span id="terminalSearchCount" role="status"></span>';
  $('criticalNews')?.before(filter);
  function filterNews(){const q=$('terminalNewsSearch').value.trim().toLocaleLowerCase('tr-TR');let count=0;for(const row of $('criticalNews')?.children||[]){const show=!q||row.textContent.toLocaleLowerCase('tr-TR').includes(q);row.hidden=!show;if(show)count++;} $('criticalNews')?.classList.toggle('terminal-searching',!!q);$('terminalSearchCount').textContent=q?count+' sonuç':'';}
  $('terminalNewsSearch').addEventListener('input',filterNews);
  if(typeof renderNews==='function'){const original=renderNews;renderNews=function(){original.apply(this,arguments);filterNews();};}
  const bottom=document.createElement('nav');bottom.id='terminalMobileNav';bottom.setAttribute('aria-label','Mobil ana bölümler');
  bottom.innerHTML=[['opsOverview','overview','Özet'],['marketRadarPanel','performance','Piyasalar'],['worldPanel','news','Haberler'],['predictionResearch','research','Araştırma']].map(([id,i,label],index)=>`<a href="#${id}"${index===0?' aria-current="location"':''}>${icon(i)}<span>${label}</span></a>`).join('');
  document.body.appendChild(bottom);
  function jump(e){const a=e.target.closest('a[href^="#"]');if(!a)return;const target=$(a.hash.slice(1));if(!target)return;e.preventDefault();for(let n=target;n;n=n.parentElement)if(n.tagName==='DETAILS')n.open=true;target.scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'start'});target.setAttribute('tabindex','-1');target.focus({preventScroll:true});}
  bottom.addEventListener('click',jump);markets.addEventListener('click',jump);
  bottom.addEventListener('click',e=>{const a=e.target.closest('a');if(!a)return;for(const link of bottom.querySelectorAll('a'))link.removeAttribute('aria-current');a.setAttribute('aria-current','location');});
  // Preserve all controls and report details; put the long performance report behind a disclosure.
  const score=$('shadowScorecard');
  if(score){const d=document.createElement('details');d.id='terminalPerformance';d.className='terminal-report';d.innerHTML='<summary>'+icon('performance')+'<span>Performans & maliyet sonrası sonuçlar<small>Gerçek ölçümler, kapsam ve test koşulları</small></span><b>Raporu aç ↗</b></summary>';score.before(d);d.appendChild(score);}
})();

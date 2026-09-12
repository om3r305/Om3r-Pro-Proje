'use strict';

/* Brian Anatomy / Development Analytics — evidence-weighted, MAIN Brian only. */
(() => {
  const ENDPOINT = `${ROOT}/brian-development-status`;
  const D = { data: null, error: null, lastSync: null, busy: false };

  function installStyle(){
    if(document.getElementById('brianDevelopmentStyle')) return;
    const style=document.createElement('style');
    style.id='brianDevelopmentStyle';
    style.textContent=`
      .dev-launch{position:relative;overflow:hidden;background:radial-gradient(circle at 10% 0%,rgba(21,223,178,.13),transparent 35%),radial-gradient(circle at 95% 100%,rgba(46,147,255,.13),transparent 40%)}
      .dev-launch:before{content:'';position:absolute;inset:-60%;background:conic-gradient(from 0deg,transparent,rgba(83,239,206,.05),transparent 22%);animation:devspin 16s linear infinite;pointer-events:none}
      @keyframes devspin{to{transform:rotate(360deg)}}
      .dev-launch-inner{position:relative;z-index:1;display:grid;grid-template-columns:auto 1fr auto;gap:14px;align-items:center}
      .dev-ring{width:82px;height:82px;border-radius:50%;display:grid;place-items:center;background:conic-gradient(#30efbe var(--p,0%),rgba(255,255,255,.07) 0);box-shadow:0 0 30px rgba(43,240,188,.12)}
      .dev-ring:after{content:'';width:64px;height:64px;border-radius:50%;background:#06111f;border:1px solid rgba(86,220,255,.18);position:absolute}.dev-ring b{position:relative;z-index:2;font-size:20px;color:#eafffa}.dev-ring small{position:absolute;z-index:2;margin-top:30px;color:#7795a9;font-size:8px}
      .dev-launch h3{margin:0;color:#eaffff;font-size:15px}.dev-launch p{margin:5px 0 0;color:#7f9aab;font-size:10px;line-height:1.45}.dev-mini{display:flex;gap:7px;flex-wrap:wrap;margin-top:9px}.dev-chip{font-size:9px;padding:5px 7px;border:1px solid rgba(89,191,225,.18);border-radius:999px;color:#9bdcec;background:rgba(4,28,43,.58)}
      .dev-open{border:1px solid rgba(48,239,190,.35);color:#8effdd;background:rgba(19,107,88,.18);padding:10px 12px;border-radius:11px;font-weight:800;font-size:10px;cursor:pointer;white-space:nowrap}
      .dev-open:hover{background:rgba(25,142,114,.26)}
      #developmentModal{position:fixed;inset:0;z-index:9999;display:none;background:rgba(0,5,13,.94);backdrop-filter:blur(16px);overflow:auto;padding:16px}.dev-modal-show{display:block!important}
      .dev-shell{max-width:1220px;margin:0 auto 40px;border:1px solid rgba(81,206,238,.18);border-radius:20px;background:linear-gradient(180deg,rgba(4,18,32,.98),rgba(2,9,18,.99));box-shadow:0 30px 100px rgba(0,0,0,.5);overflow:hidden}
      .dev-head{position:sticky;top:0;z-index:5;background:rgba(3,15,27,.94);backdrop-filter:blur(12px);display:flex;justify-content:space-between;gap:12px;align-items:center;padding:14px 16px;border-bottom:1px solid rgba(87,196,226,.13)}
      .dev-title{font-size:16px;font-weight:900;color:#eaffff}.dev-sub{font-size:9px;color:#7590a4;margin-top:3px}.dev-close{border:1px solid rgba(255,255,255,.14);border-radius:10px;background:rgba(255,255,255,.04);color:#d9eff6;padding:8px 11px;cursor:pointer}
      .dev-body{padding:14px}.dev-hero{display:grid;grid-template-columns:minmax(300px,.9fr) minmax(0,1.1fr);gap:14px}.dev-anatomy,.dev-overview,.dev-panel{border:1px solid rgba(71,184,220,.14);border-radius:16px;background:rgba(3,18,31,.7);padding:13px}
      .dev-anatomy{display:grid;place-items:center;min-height:530px;position:relative;background:radial-gradient(circle at 50% 35%,rgba(28,171,210,.12),transparent 45%),rgba(3,18,31,.7)}
      .dev-figure{width:min(100%,430px);height:auto;overflow:visible}.dev-part{stroke:rgba(220,250,255,.28);stroke-width:2;transition:.25s;filter:drop-shadow(0 0 8px rgba(0,0,0,.25))}.dev-nerve{fill:none;stroke-width:3;stroke-linecap:round;opacity:.72;stroke-dasharray:4 8;animation:nerve 1.8s linear infinite}@keyframes nerve{to{stroke-dashoffset:-24}}
      .dev-label{fill:#b7dbe6;font:700 11px system-ui}.dev-label-small{fill:#718c9f;font:500 9px system-ui}.dev-score-svg{fill:#fff;font:900 14px system-ui}
      .dev-topline{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}.dev-kpi{padding:12px;border-radius:12px;background:rgba(5,29,45,.72);border:1px solid rgba(72,205,238,.13)}.dev-kpi span{display:block;color:#7693a6;font-size:8px;text-transform:uppercase;letter-spacing:.08em}.dev-kpi b{display:block;margin-top:5px;font-size:21px;color:#efffff}.dev-kpi em{display:block;margin-top:2px;color:#7390a2;font-size:8px;font-style:normal}
      .dev-formula{margin-top:10px;padding:10px;border-radius:11px;background:rgba(6,23,37,.7);color:#91aaba;font-size:9px;line-height:1.55;border:1px dashed rgba(79,184,218,.15)}
      .dev-bars{margin-top:12px;display:grid;gap:9px}.dev-bar-row{display:grid;grid-template-columns:minmax(135px,.34fr) 1fr auto;gap:9px;align-items:center}.dev-bar-name{font-size:10px;color:#c6e4ed}.dev-bar-track{height:10px;border-radius:999px;background:rgba(255,255,255,.05);overflow:hidden}.dev-bar-fill{height:100%;border-radius:inherit;box-shadow:0 0 12px currentColor}.dev-bar-value{font:800 10px ui-monospace,monospace;color:#dff8ff;min-width:42px;text-align:right}
      .dev-legend{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}.dev-legend span{font-size:8px;color:#8aa2b2}.dev-dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:4px}
      .dev-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:12px}.dev-panel h4{margin:0 0 9px;font-size:12px;color:#e9fbff}.dev-panel-sub{font-size:8px;color:#6f8a9e;margin:-4px 0 9px}
      .dev-component{padding:10px 0;border-bottom:1px solid rgba(120,190,220,.08)}.dev-component:last-child{border-bottom:0}.dev-comp-head{display:flex;align-items:flex-start;justify-content:space-between;gap:8px}.dev-comp-name{font-size:10px;font-weight:800;color:#dff7ff}.dev-comp-anatomy{font-size:8px;color:#718fa3;margin-top:2px}.dev-comp-score{font:900 12px ui-monospace,monospace}.dev-triple{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:5px;margin-top:8px}.dev-tiny{background:rgba(4,25,39,.68);border-radius:8px;padding:7px}.dev-tiny span{display:block;color:#6d879a;font-size:7px;text-transform:uppercase}.dev-tiny b{display:block;font-size:10px;color:#dff8ff;margin-top:3px}.dev-rationale{color:#879fac;font-size:8px;line-height:1.5;margin-top:7px}
      .dev-list{display:grid;gap:7px}.dev-item{padding:9px;border-radius:10px;background:rgba(4,26,40,.66);border:1px solid rgba(76,173,207,.10);font-size:9px;color:#b7d2db;line-height:1.45}.dev-item b{color:#ebfbff}.dev-item strong{float:right}.dev-good{color:#42efb2!important}.dev-mid{color:#ffcb63!important}.dev-bad{color:#ff667b!important}.dev-unproven{color:#8998a4!important}
      .dev-arms{max-height:360px;overflow:auto}.dev-arm{display:grid;grid-template-columns:minmax(0,1fr) 60px 45px;gap:8px;padding:8px 0;border-bottom:1px solid rgba(120,190,220,.08);align-items:center}.dev-arm:last-child{border-bottom:0}.dev-arm-name{font-size:9px;color:#b9d8e2;overflow-wrap:anywhere}.dev-arm-meta{font-size:7px;color:#718c9e;margin-top:2px}.dev-arm-score{font:800 10px ui-monospace,monospace;text-align:right}.dev-spark{height:5px;background:rgba(255,255,255,.05);border-radius:99px;overflow:hidden}.dev-spark i{display:block;height:100%;border-radius:inherit}
      .dev-warning{padding:9px;border-radius:10px;border-left:3px solid #ffbd55;background:rgba(93,62,13,.12);font-size:8px;color:#c9b98d;line-height:1.5}.dev-empty{padding:15px;color:#758c9c;font-size:9px;text-align:center;border:1px dashed rgba(120,190,220,.13);border-radius:10px}
      .dev-refresh{border:1px solid rgba(57,219,184,.24);background:rgba(17,94,81,.15);color:#6af2d1;border-radius:9px;padding:7px 9px;font-size:8px;cursor:pointer;margin-right:6px}
      @media(max-width:800px){#developmentModal{padding:0}.dev-shell{border-radius:0;margin:0;min-height:100vh}.dev-hero,.dev-grid{grid-template-columns:1fr}.dev-anatomy{min-height:470px}.dev-launch-inner{grid-template-columns:auto 1fr}.dev-open{grid-column:1/-1}.dev-topline{grid-template-columns:repeat(3,1fr)}.dev-kpi{padding:9px}.dev-kpi b{font-size:17px}.dev-bar-row{grid-template-columns:105px 1fr 38px}}
    `;
    document.head.appendChild(style);
  }

  const colorFor=(c)=>{
    const ev=Number(c?.evidence_pct||0),m=Number(c?.maturity_pct||0);
    if(ev<10) return '#66717c';
    if(m>=70) return '#36efae';
    if(m>=45) return '#31d7e8';
    if(m>=25) return '#ffc857';
    return '#ff5f73';
  };
  const clsFor=(c)=>{
    const ev=Number(c?.evidence_pct||0),m=Number(c?.maturity_pct||0);
    if(ev<10)return 'dev-unproven';if(m>=45)return 'dev-good';if(m>=25)return 'dev-mid';return 'dev-bad';
  };
  const fmt=(n,d=1)=>Number.isFinite(Number(n))?Number(n).toFixed(d):'—';
  const pct1=(n)=>Number.isFinite(Number(n))?`${fmt(n)}%`:'—';
  const comp=(id)=>D.data?.components?.find(x=>x.id===id)||null;

  function ensureUI(){
    installStyle();
    if(!document.getElementById('developmentLaunch')){
      const target=document.getElementById('autonomyRow') || document.getElementById('chatPanel')?.closest('section');
      if(target?.parentNode){
        const section=document.createElement('section');
        section.id='developmentLaunch';
        section.className='card card-pad dev-launch';
        section.innerHTML=`<div class="dev-launch-inner">
          <div class="dev-ring" id="devMiniRing" style="--p:0%"><b id="devMiniScore">—</b><small>GELİŞİM</small></div>
          <div><h3>🧬 Brian Analiz & Gelişim</h3><p>Başarı, kanıt derinliği, veri akışı, laboratuvar ve hazineyi birlikte ölçen canlı beyin olgunluk raporu.</p><div class="dev-mini"><span class="dev-chip" id="devMiniTier">KANIT OKUNUYOR</span><span class="dev-chip" id="devMiniQuality">Kalite —</span><span class="dev-chip" id="devMiniFlow">Veri akışı —</span></div></div>
          <button class="dev-open" id="devOpenBtn">ANATOMİ RAPORUNU AÇ →</button>
        </div>`;
        target.parentNode.insertBefore(section,target);
      }
    }
    if(!document.getElementById('developmentModal')){
      const modal=document.createElement('div');
      modal.id='developmentModal';
      modal.innerHTML=`<div class="dev-shell"><div class="dev-head"><div><div class="dev-title">🧬 Brian Gelişim Anatomisi</div><div class="dev-sub">Güçlü organlar, zayıf halkalar, kanıt seviyesi ve hangi özelliklerin gerçekten işe yaradığı.</div></div><div><button class="dev-refresh" id="devRefreshBtn">↻ CANLI YENİLE</button><button class="dev-close" id="devCloseBtn">KAPAT ✕</button></div></div><div class="dev-body" id="developmentBody"><div class="dev-empty">Canlı analiz hazırlanıyor…</div></div></div>`;
      document.body.appendChild(modal);
    }
    document.getElementById('devOpenBtn')?.addEventListener('click',openModal,{once:true});
    document.getElementById('devCloseBtn')?.addEventListener('click',closeModal,{once:true});
    document.getElementById('devRefreshBtn')?.addEventListener('click',()=>refresh(true),{once:true});
    document.getElementById('developmentModal')?.addEventListener('click',(e)=>{if(e.target?.id==='developmentModal')closeModal()});
  }

  function openModal(){ensureUI();document.getElementById('developmentModal')?.classList.add('dev-modal-show');document.body.style.overflow='hidden';render();if(!D.data)refresh(true)}
  function closeModal(){document.getElementById('developmentModal')?.classList.remove('dev-modal-show');document.body.style.overflow=''}

  function anatomySvg(){
    const a=comp('alpha'),s=comp('sensors'),w=comp('world'),l=comp('lab'),t=comp('treasury');
    const ca=colorFor(a),cs=colorFor(s),cw=colorFor(w),cl=colorFor(l),ct=colorFor(t);
    const nerve=Number(D.data?.overall?.data_exchange_health_pct||0)>=75?'#38efb2':Number(D.data?.overall?.data_exchange_health_pct||0)>=45?'#ffc75a':'#ff6277';
    const score=(c)=>c?.evidence_pct<10?'?':Math.round(Number(c?.maturity_pct||0));
    return `<svg class="dev-figure" viewBox="0 0 430 560" role="img" aria-label="Brian sistem anatomisi">
      <defs><filter id="glow"><feGaussianBlur stdDeviation="4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
      <path class="dev-nerve" stroke="${nerve}" d="M215 122 C215 170 215 210 215 255 M215 255 C170 285 150 315 128 352 M215 255 C260 285 280 315 302 352 M215 255 C202 330 184 390 167 473 M215 255 C228 330 246 390 263 473"/>
      <circle class="dev-part" cx="215" cy="78" r="50" fill="${ca}" fill-opacity=".58" filter="url(#glow)"/>
      <path class="dev-part" fill="${ca}" fill-opacity=".83" d="M181 71c0-20 15-35 34-35 19 0 34 15 34 35 0 11-5 20-13 27-5 4-9 11-10 19h-22c-1-8-5-15-10-19-8-7-13-16-13-27z"/>
      <circle class="dev-part" cx="198" cy="78" r="7" fill="${cs}"/><circle class="dev-part" cx="232" cy="78" r="7" fill="${cs}"/>
      <path class="dev-part" fill="${cw}" fill-opacity=".66" d="M169 145 Q215 126 261 145 L276 277 Q215 306 154 277 Z"/>
      <path class="dev-part" fill="${cl}" fill-opacity=".67" d="M165 151 Q132 170 109 229 L88 326 116 335 144 249 175 202Z"/><path class="dev-part" fill="${cl}" fill-opacity=".67" d="M265 151 Q298 170 321 229 L342 326 314 335 286 249 255 202Z"/>
      <path class="dev-part" fill="${ct}" fill-opacity=".68" d="M169 280 L207 286 196 406 175 515 142 510 156 392Z"/><path class="dev-part" fill="${ct}" fill-opacity=".68" d="M261 280 L223 286 234 406 255 515 288 510 274 392Z"/>
      <text class="dev-label" x="15" y="44">BEYİN · ALPHA</text><text class="dev-score-svg" x="33" y="64">${score(a)}%</text><line x1="96" y1="58" x2="169" y2="72" stroke="${ca}" opacity=".6"/>
      <text class="dev-label" x="306" y="83">GÖZ / SİNİR</text><text class="dev-score-svg" x="333" y="103">${score(s)}%</text><line x1="301" y1="92" x2="244" y2="82" stroke="${cs}" opacity=".6"/>
      <text class="dev-label" x="300" y="210">DÜNYA / HAFIZA</text><text class="dev-score-svg" x="332" y="230">${score(w)}%</text><line x1="294" y1="216" x2="266" y2="216" stroke="${cw}" opacity=".6"/>
      <text class="dev-label" x="14" y="293">LAB / GELİŞİM</text><text class="dev-score-svg" x="39" y="313">${score(l)}%</text><line x1="102" y1="304" x2="121" y2="286" stroke="${cl}" opacity=".6"/>
      <text class="dev-label" x="300" y="451">HAZİNE / UYGULAMA</text><text class="dev-score-svg" x="339" y="471">${score(t)}%</text><line x1="295" y1="459" x2="270" y2="459" stroke="${ct}" opacity=".6"/>
      <text class="dev-label-small" x="151" y="548">Sinir akışı = veri transfer sağlığı ${pct1(D.data?.overall?.data_exchange_health_pct)}</text>
    </svg>`;
  }

  function metricRows(c){
    if(!c) return '';
    const q=c.quality_pct==null?'—':pct1(c.quality_pct),e=pct1(c.evidence_pct),m=pct1(c.maturity_pct);
    return `<div class="dev-component"><div class="dev-comp-head"><div><div class="dev-comp-name">${esc(c.name)}</div><div class="dev-comp-anatomy">${esc(c.anatomy)} · ${Number(c.samples||0).toLocaleString('tr-TR')} kanıt</div></div><div class="dev-comp-score ${clsFor(c)}">${m}</div></div><div class="dev-triple"><div class="dev-tiny"><span>Kalite</span><b>${q}</b></div><div class="dev-tiny"><span>Kanıt</span><b>${e}</b></div><div class="dev-tiny"><span>Olgunluk</span><b>${m}</b></div></div><div class="dev-rationale">${esc(c.rationale)}</div></div>`;
  }

  function armsHtml(){
    const rows=Array.isArray(D.data?.arms)?D.data.arms:[];
    if(!rows.length)return '<div class="dev-empty">Son 24 saatte ana Brian kolu çalışma kanıtı yok.</div>';
    return `<div class="dev-arms">${rows.map(r=>{const p=Number(r.success_pct||0),c=p>=75?'#38efb2':p>=45?'#ffc857':'#ff6075';return `<div class="dev-arm"><div><div class="dev-arm-name">${esc(r.id)}</div><div class="dev-arm-meta">${r.runs} çalışma · ${r.success} başarılı · ${r.degraded} degraded · ${r.failed} hata${r.transfer_capture_pct==null?'':` · kayıt ${pct1(r.transfer_capture_pct)}`}</div></div><div class="dev-spark"><i style="width:${Math.max(0,Math.min(100,p))}%;background:${c}"></i></div><div class="dev-arm-score" style="color:${c}">${pct1(p)}</div></div>`}).join('')}</div>`;
  }

  function featureSignalsHtml(){
    const rows=Array.isArray(D.data?.feature_signals)?D.data.feature_signals:[];
    if(!rows.length)return '<div class="dev-empty">Promotion Council feature etkisi için daha fazla kanıt bekliyor.</div>';
    return `<div class="dev-list">${rows.map(x=>{const good=x.signal==='STRENGTH_CANDIDATE',bad=x.signal==='WEAK_REJECTED';const cls=good?'dev-good':bad?'dev-bad':'dev-mid';const label=good?'GÜÇLENDİRME ADAYI':bad?'RED / ZAYIF':'KANIT BEKLİYOR';return `<div class="dev-item"><strong class="${cls}">${label}</strong><b>${esc(x.experiment_id)}</b><br><span>${esc((x.reasons||[]).slice(0,2).join(' · ')||x.decision)}</span></div>`}).join('')}</div>`;
  }

  function gapsHtml(){
    const rows=Array.isArray(D.data?.capability_gaps)?D.data.capability_gaps:[];
    if(!rows.length)return '<div class="dev-empty">Aktif orta/yüksek kritik gelişim açığı görünmüyor.</div>';
    return `<div class="dev-list">${rows.map(g=>`<div class="dev-item"><strong class="${g.severity==='CRITICAL'||g.severity==='HIGH'?'dev-bad':'dev-mid'}">${esc(g.severity)}</strong><b>${esc(g.domain)} · ${esc(g.capability_id)}</b><br>${esc(g.reason)}<br><span style="color:#6e92a6">→ ${esc(g.suggested_action||'araştır')}</span></div>`).join('')}</div>`;
  }

  function render(){
    ensureUI();
    if(D.error){
      const b=document.getElementById('developmentBody');if(b)b.innerHTML=`<div class="dev-warning"><b>Analiz servisi okunamadı.</b><br>${esc(D.error)}</div>`;
      return;
    }
    if(!D.data)return;
    const o=D.data.overall||{},score=Number(o.brain_development_pct||0);
    const mini=document.getElementById('devMiniRing');if(mini)mini.style.setProperty('--p',`${Math.max(0,Math.min(100,score))}%`);
    if(document.getElementById('devMiniScore'))document.getElementById('devMiniScore').textContent=`${Math.round(score)}%`;
    if(document.getElementById('devMiniTier'))document.getElementById('devMiniTier').textContent=`SEVİYE: ${o.tier||'—'}`;
    if(document.getElementById('devMiniQuality'))document.getElementById('devMiniQuality').textContent=`Kalite ${pct1(o.measured_quality_pct)}`;
    if(document.getElementById('devMiniFlow'))document.getElementById('devMiniFlow').textContent=`Veri akışı ${pct1(o.data_exchange_health_pct)}`;

    const comps=Array.isArray(D.data.components)?D.data.components:[];
    const barHtml=comps.map(c=>`<div class="dev-bar-row"><div class="dev-bar-name">${esc(c.name)}</div><div class="dev-bar-track"><div class="dev-bar-fill" style="width:${Math.max(0,Math.min(100,Number(c.maturity_pct||0)))}%;background:${colorFor(c)};color:${colorFor(c)}"></div></div><div class="dev-bar-value ${clsFor(c)}">${c.evidence_pct<10?'KANIT?':pct1(c.maturity_pct)}</div></div>`).join('');
    const strength=(D.data.strengths||[]).map(x=>`<div class="dev-item"><strong class="dev-good">GÜÇLÜ</strong><b>${esc(x.name)}</b><br>${pct1(x.maturity_pct)} kanıtlı olgunluk</div>`).join('')||'<div class="dev-empty">Henüz güçlü olarak sınıflanacak yeterli kanıt yok.</div>';
    const weak=(D.data.weaknesses||[]).map(x=>`<div class="dev-item"><strong class="dev-bad">GELİŞTİR</strong><b>${esc(x.name)}</b><br>${pct1(x.maturity_pct)} kanıtlı olgunluk</div>`).join('');
    const unproven=(D.data.unproven||[]).map(x=>`<div class="dev-item"><strong class="dev-unproven">KANITSIZ</strong><b>${esc(x.name)}</b><br>${pct1(x.evidence_pct)} kanıt derinliği · puan uydurulmadı</div>`).join('');
    const body=document.getElementById('developmentBody');if(!body)return;
    body.innerHTML=`
      <div class="dev-hero"><div class="dev-anatomy">${anatomySvg()}</div><div class="dev-overview">
        <div class="dev-topline"><div class="dev-kpi"><span>Brian Beyin Gelişimi</span><b>${pct1(score)}</b><em>${esc(o.tier||'—')}</em></div><div class="dev-kpi"><span>Ölçülmüş Kalite</span><b>${pct1(o.measured_quality_pct)}</b><em>davranış sonucu</em></div><div class="dev-kpi"><span>Kanıt Güveni</span><b>${pct1(o.evidence_confidence_pct)}</b><em>örnek / kapsam</em></div></div>
        <div class="dev-formula"><b>Bu yüzde dekoratif değil.</b> Önce her organda gerçek davranış kalitesi ölçülür; sonra kanıt derinliği ile çarpılır. Genel beyin gelişimi kritik organların ağırlıklı geometrik ortalamasıdır. Bu yüzden tek bir güçlü sensör, kanıtsız hazineyi veya zayıf ALPHA'yı gizleyemez.</div>
        <div class="dev-bars">${barHtml}</div>
        <div class="dev-legend"><span><i class="dev-dot" style="background:#36efae"></i>Güçlü ≥70</span><span><i class="dev-dot" style="background:#31d7e8"></i>İyi ≥45</span><span><i class="dev-dot" style="background:#ffc857"></i>Gelişiyor ≥25</span><span><i class="dev-dot" style="background:#ff5f73"></i>Zayıf &lt;25</span><span><i class="dev-dot" style="background:#66717c"></i>Kanıt yetersiz</span></div>
      </div></div>
      <div class="dev-grid"><div class="dev-panel"><h4>🧠 Organ Bazlı Rapor</h4><div class="dev-panel-sub">Kalite ≠ kanıt. İkisi ayrı tutulur.</div>${comps.map(metricRows).join('')}</div><div class="dev-panel"><h4>⚡ Brian Kolları / Veri Alışverişi</h4><div class="dev-panel-sub">Son 24 saat MAIN Brian collector başarı yüzdeleri.</div>${armsHtml()}</div></div>
      <div class="dev-grid"><div class="dev-panel"><h4>✅ Güçlendiren / ⚠️ Geliştirilecek</h4><div class="dev-list">${strength}${weak}${unproven}</div></div><div class="dev-panel"><h4>🧪 Özellik Etki Kanıtı</h4><div class="dev-panel-sub">Promotion Council: bir özelliğin üretildiğini değil, deney sonucunu sınıflar.</div>${featureSignalsHtml()}</div></div></div>
      <div class="dev-grid"><div class="dev-panel"><h4>🧬 Açık Gelişim Boşlukları</h4>${gapsHtml()}</div><div class="dev-panel"><h4>📐 Ölçüm Sözleşmesi</h4><div class="dev-warning">Hazine gibi henüz tamamlanmış döngüsü olmayan organlara 100% servis sağlığı yazıp “başarılı” demiyoruz. Kanıt yoksa gri kalır. Dünya kaynak sayısı tek başına başarı değildir. ALPHA'da maliyet sonrası olumlu sonuç esas alınır. Laboratuvarda leakage ve veri kalitesi kontrol edilir.</div><div class="dev-rationale" style="margin-top:10px">Son analiz: ${esc(new Date(D.data.observed_at||Date.now()).toLocaleString('tr-TR'))} · Model: ${esc(D.data.model||'—')} · SHADOW ONLY · canlı emir yok.</div></div></div>`;
  }

  async function refresh(force=false){
    if(D.busy||!key())return;
    if(!force&&D.lastSync&&Date.now()-D.lastSync<20000)return;
    D.busy=true;
    try{D.data=await post(ENDPOINT,{});D.error=null;D.lastSync=Date.now()}catch(e){D.error=String(e?.message||e)}finally{D.busy=false;render()}
  }

  const boot=()=>{ensureUI();refresh(true);setInterval(()=>refresh(false),60000)};
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',()=>setTimeout(boot,300));else setTimeout(boot,300);
})();

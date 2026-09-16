'use strict';

/* Visual-only warning beacon. Mirrors fresh CRITICAL/HIGH meeting urgency onto the visible Operations room tile. */
(function installMeetingAlertBeacon(){
  if(document.getElementById('frontierMeetingAlertBeaconStyle')) return;
  const style=document.createElement('style');
  style.id='frontierMeetingAlertBeaconStyle';
  style.textContent=`
    [data-ops-room="meeting"]{position:relative;isolation:isolate;overflow:visible!important;transition:box-shadow .2s ease,border-color .2s ease,transform .2s ease}
    [data-ops-room="meeting"].meeting-alert-red{z-index:3;border-color:rgba(255,63,86,.96)!important;box-shadow:0 0 0 1px rgba(255,45,70,.42),0 0 16px rgba(255,34,60,.62),0 0 42px rgba(255,24,48,.38)!important;animation:meetingCriticalButtonAlarm .82s ease-in-out infinite}
    [data-ops-room="meeting"].meeting-alert-red:before,[data-ops-room="meeting"].meeting-alert-red:after{content:"";position:absolute;pointer-events:none;border-radius:inherit;inset:-4px;z-index:-1;border:2px solid rgba(255,54,78,.78);animation:meetingCriticalButtonHalo .82s ease-out infinite}
    [data-ops-room="meeting"].meeting-alert-red:after{inset:-11px;border-width:1px;animation-delay:.28s}
    [data-ops-room="meeting"].meeting-alert-orange{z-index:3;border-color:rgba(255,161,48,.92)!important;box-shadow:0 0 0 1px rgba(255,151,37,.34),0 0 14px rgba(255,139,28,.52),0 0 34px rgba(255,118,18,.28)!important;animation:meetingHighButtonAlarm 1.25s ease-in-out infinite}
    [data-ops-room="meeting"].meeting-alert-orange:before,[data-ops-room="meeting"].meeting-alert-orange:after{content:"";position:absolute;pointer-events:none;border-radius:inherit;inset:-4px;z-index:-1;border:2px solid rgba(255,164,53,.62);animation:meetingHighButtonHalo 1.25s ease-out infinite}
    [data-ops-room="meeting"].meeting-alert-orange:after{inset:-10px;border-width:1px;animation-delay:.38s}
    [data-ops-room="meeting"].meeting-alert-red i,[data-ops-room="meeting"].meeting-alert-red b{color:#ff8798!important;text-shadow:0 0 14px rgba(255,52,78,.46)}
    [data-ops-room="meeting"].meeting-alert-orange i,[data-ops-room="meeting"].meeting-alert-orange b{color:#ffc16f!important;text-shadow:0 0 12px rgba(255,151,45,.38)}
    @keyframes meetingCriticalButtonAlarm{0%,100%{box-shadow:0 0 0 1px rgba(255,45,70,.30),0 0 9px rgba(255,34,60,.40),0 0 22px rgba(255,24,48,.20);transform:translateZ(0) scale(1)}50%{box-shadow:0 0 0 2px rgba(255,73,94,.76),0 0 27px rgba(255,45,69,.96),0 0 64px rgba(255,25,52,.58);transform:translateZ(0) scale(1.018)}}
    @keyframes meetingCriticalButtonHalo{0%{transform:scale(.98);opacity:.94}100%{transform:scale(1.10);opacity:0}}
    @keyframes meetingHighButtonAlarm{0%,100%{box-shadow:0 0 0 1px rgba(255,151,37,.24),0 0 8px rgba(255,139,28,.32),0 0 18px rgba(255,118,18,.15);transform:translateZ(0) scale(1)}50%{box-shadow:0 0 0 2px rgba(255,180,75,.62),0 0 22px rgba(255,158,45,.84),0 0 50px rgba(255,126,22,.42);transform:translateZ(0) scale(1.012)}}
    @keyframes meetingHighButtonHalo{0%{transform:scale(.985);opacity:.82}100%{transform:scale(1.085);opacity:0}}
    @media (prefers-reduced-motion:reduce){[data-ops-room="meeting"].meeting-alert-red,[data-ops-room="meeting"].meeting-alert-orange,[data-ops-room="meeting"].meeting-alert-red:before,[data-ops-room="meeting"].meeting-alert-red:after,[data-ops-room="meeting"].meeting-alert-orange:before,[data-ops-room="meeting"].meeting-alert-orange:after{animation:none!important}}
  `;
  document.head.appendChild(style);

  function activeUrgency(){
    if(typeof window.meetingV6MajorEvents!=='function') return null;
    let events=[];
    try{events=window.meetingV6MajorEvents()||[]}catch{return null}
    const now=Date.now();
    const fresh=events.filter(e=>{
      const at=Date.parse(e?.time||'');
      if(!Number.isFinite(at)||at>now) return false;
      const urgency=String(e?.urgency||'').toUpperCase();
      const maxAge=urgency==='CRITICAL'?7200000:urgency==='HIGH'?3600000:0;
      return maxAge>0&&now-at<=maxAge;
    });
    if(fresh.some(e=>String(e?.urgency||'').toUpperCase()==='CRITICAL')) return 'red';
    if(fresh.some(e=>String(e?.urgency||'').toUpperCase()==='HIGH')) return 'orange';
    return null;
  }

  function sync(){
    const tile=document.querySelector('[data-ops-room="meeting"]');
    if(!tile)return;
    const urgency=activeUrgency();
    tile.classList.toggle('meeting-alert-red',urgency==='red');
    tile.classList.toggle('meeting-alert-orange',urgency==='orange');
    tile.dataset.meetingAlert=urgency||'calm';
  }
  const observer=new MutationObserver(sync);
  observer.observe(document.body,{childList:true,subtree:true});
  sync();
  setInterval(()=>{if(!document.hidden)sync()},3000);
  document.addEventListener('visibilitychange',()=>{if(!document.hidden)sync()});
})();

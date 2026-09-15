'use strict';

/* Presentation-only UI adapter for the legacy HUMAN_APPROVAL state.
   Backend approval authority is gpt-evidence-gate; DB/state-machine field names stay
   unchanged for compatibility. This file never performs an approval action. */
(function(){
  const replacements=[
    ['ÖMER ONAYI','GPT ONAYI'],
    ['Ömer Onayı','GPT Onayı'],
    ['Ömer Onay Masası','GPT Onay Masası'],
    ['Ömer onayı','GPT onayı'],
    ['İNSAN ONAYI ZORUNLU','GPT KANIT ONAYI'],
    ['insan onayı/complete','GPT kanıt onayı/complete'],
    ['Şu anda Ömer onayı bekleyen aday yok.','Şu anda GPT onayı bekleyen aday yok.'],
    ['gerçek onay kuyruğu yalnız HUMAN_APPROVAL + WAITING run’larından oluşur.','gerçek GPT karar kuyruğu yalnız HUMAN_APPROVAL + WAITING run’larından oluşur; bu legacy state adı DB uyumluluğu için korunur.']
  ];

  function relabel(root){
    if(!root)return;
    const walker=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
    const nodes=[];
    while(walker.nextNode())nodes.push(walker.currentNode);
    for(const node of nodes){
      let next=node.nodeValue||'';
      for(const [from,to] of replacements)next=next.split(from).join(to);
      if(next!==node.nodeValue)node.nodeValue=next;
    }
  }

  function compactMainEngineerCard(){
    const card=document.getElementById('developerBrian');
    if(!card)return;

    const title=card.querySelector('.section-title');
    const sub=card.querySelector('.section-sub');
    if(title)title.textContent='👨‍💻 Yazılımcı Brian';
    if(sub)sub.textContent='Ayrıntılı kod, test, review ve GPT karar akışı Mühendislik Odası’nda.';

    // Main dashboard stays executive-level. Detailed engineering truth remains in the room.
    for(const id of ['autonomySummary','autonomyGovernance','codeStream']){
      const el=document.getElementById(id);
      if(el)el.style.setProperty('display','none','important');
    }

    const launch=document.getElementById('berLaunch');
    if(launch){
      launch.textContent='👨‍💻 MÜHENDİSLİK ODASINI AÇ →';
      launch.style.removeProperty('display');
      // The room button is the boundary: hide lower engineering detail/source blocks on the main card.
      let sibling=launch.nextElementSibling;
      while(sibling){
        sibling.style.setProperty('display','none','important');
        sibling=sibling.nextElementSibling;
      }
    }

    relabel(card);
  }

  function apply(){
    relabel(document.getElementById('brianEngineeringRoomV2'));
    compactMainEngineerCard();
  }

  let queued=false;
  const scheduleApply=()=>{
    if(queued)return;
    queued=true;
    queueMicrotask(()=>{queued=false;apply();});
  };
  const observer=new MutationObserver(scheduleApply);
  observer.observe(document.documentElement,{subtree:true,childList:true,characterData:true});
  document.addEventListener('click',()=>setTimeout(apply,0),true);
  setInterval(apply,3000);
  apply();
})();

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

  function setTextIfChanged(el,value){
    if(el&&el.textContent!==value)el.textContent=value;
  }

  function compactMainEngineerCard(){
    const card=document.getElementById('developerBrian');
    if(!card)return;

    setTextIfChanged(card.querySelector('.section-title'),'👨‍💻 Yazılımcı Brian');
    setTextIfChanged(card.querySelector('.section-sub'),'Ayrıntılı kod, test, review ve GPT karar akışı Mühendislik Odası’nda.');

    for(const id of ['autonomySummary','autonomyGovernance','codeStream']){
      const el=document.getElementById(id);
      if(el&&el.style.display!=='none')el.style.setProperty('display','none','important');
    }

    const launch=document.getElementById('berLaunch');
    if(launch){
      setTextIfChanged(launch,'👨‍💻 MÜHENDİSLİK ODASINI AÇ →');
      if(launch.style.display==='none')launch.style.removeProperty('display');
      let sibling=launch.nextElementSibling;
      while(sibling){
        if(sibling.style.display!=='none')sibling.style.setProperty('display','none','important');
        sibling=sibling.nextElementSibling;
      }
    }

    relabel(card);
  }

  function apply(){
    relabel(document.getElementById('brianEngineeringRoomV2'));
    compactMainEngineerCard();
  }

  // Deliberately no MutationObserver here. The previous observer could observe its own
  // textContent writes and starve the browser event loop. A small bounded timer is enough
  // because the underlying engineering console refreshes on a 15-second cadence.
  document.addEventListener('click',()=>setTimeout(apply,0),true);
  setInterval(apply,2000);
  setTimeout(apply,0);
})();

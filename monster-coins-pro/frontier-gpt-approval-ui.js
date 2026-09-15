'use strict';

/* Presentation-only label adapter for the legacy HUMAN_APPROVAL state.
   Backend approval authority is gpt-evidence-gate; DB column names remain legacy
   for state-machine compatibility. No approval action is performed in the browser. */
(function(){
  const replacements=[
    ['ÖMER ONAYI','GPT ONAYI'],
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

  function apply(){relabel(document.getElementById('brianEngineeringRoomV2'));}
  const observer=new MutationObserver(()=>queueMicrotask(apply));
  observer.observe(document.documentElement,{subtree:true,childList:true,characterData:true});
  document.addEventListener('click',()=>setTimeout(apply,0),true);
  setInterval(apply,3000);
  apply();
})();

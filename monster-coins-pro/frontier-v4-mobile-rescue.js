'use strict';
(()=>{
  const $=(q,r=document)=>r.querySelector(q);

  function overlayOpen(){
    return Boolean($('.unlock.show') || $('.modal.show') || $('#brianAnatomyLive.bal-show'));
  }

  function setStyle(el,key,value){
    if(el && el.style[key]!==value) el.style[key]=value;
  }

  function repair(){
    const open=overlayOpen();
    if(document.body.classList.contains('frontier-overlay-open')!==open){
      document.body.classList.toggle('frontier-overlay-open',open);
    }

    if(!open){
      if(document.body.style.overflow==='hidden') document.body.style.overflow='';
      if(document.documentElement.style.overflow==='hidden') document.documentElement.style.overflow='';
      setStyle(document.body,'touchAction','pan-y pinch-zoom');
      setStyle(document.documentElement,'touchAction','pan-y pinch-zoom');
    }

    document.querySelectorAll('.hero .flow-map,.hero .brain-core,.hero .brain-orbit,.hero .brain-svg').forEach(el=>{
      setStyle(el,'pointerEvents','none');
    });

    const unlock=$('.unlock');
    if(unlock) setStyle(unlock,'pointerEvents',unlock.classList.contains('show')?'auto':'none');
    document.querySelectorAll('.modal').forEach(el=>{
      setStyle(el,'pointerEvents',el.classList.contains('show')?'auto':'none');
    });
    const anatomy=$('#brianAnatomyLive');
    if(anatomy) setStyle(anatomy,'pointerEvents',anatomy.classList.contains('bal-show')?'auto':'none');
  }

  function boot(){
    repair();
    requestAnimationFrame(repair);
    setTimeout(repair,350);
    setTimeout(repair,1400);

    window.addEventListener('pageshow',repair,{passive:true});
    window.addEventListener('orientationchange',()=>setTimeout(repair,220),{passive:true});
    document.addEventListener('visibilitychange',()=>{if(document.visibilityState==='visible')repair()});
    document.addEventListener('click',()=>setTimeout(repair,0),true);

    // Only watch DOM insertion/removal. Never observe class/style attributes here:
    // observing our own style writes caused an infinite MutationObserver loop on iOS Safari.
    const mo=new MutationObserver(()=>requestAnimationFrame(repair));
    mo.observe(document.body,{subtree:true,childList:true});
  }

  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',boot,{once:true});
  else boot();
})();

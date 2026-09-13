'use strict';
(()=>{
  const $=(q,r=document)=>r.querySelector(q);

  function overlayOpen(){
    return Boolean($('.unlock.show') || $('.modal.show') || $('#brianAnatomyLive.bal-show'));
  }

  function repair(){
    const open=overlayOpen();
    document.body.classList.toggle('frontier-overlay-open',open);

    if(!open){
      // Old anatomy/modal code can leave an inline overflow lock behind on iOS.
      if(document.body.style.overflow==='hidden') document.body.style.overflow='';
      if(document.documentElement.style.overflow==='hidden') document.documentElement.style.overflow='';
      document.body.style.touchAction='pan-y pinch-zoom';
      document.documentElement.style.touchAction='pan-y pinch-zoom';
    }

    // Decorative layers are never interaction targets.
    document.querySelectorAll('.hero .flow-map,.hero .brain-core,.hero .brain-orbit,.hero .brain-svg').forEach(el=>{
      el.style.pointerEvents='none';
    });

    // Hidden overlays must not survive as transparent tap shields.
    const unlock=$('.unlock');
    if(unlock&&!unlock.classList.contains('show')) unlock.style.pointerEvents='none';
    document.querySelectorAll('.modal').forEach(el=>{
      el.style.pointerEvents=el.classList.contains('show')?'auto':'none';
    });
    const anatomy=$('#brianAnatomyLive');
    if(anatomy) anatomy.style.pointerEvents=anatomy.classList.contains('bal-show')?'auto':'none';
  }

  function forceBottomClearance(){
    const app=$('.app');
    const nav=$('.bottom-nav');
    if(!app||!nav)return;
    const h=Math.ceil(nav.getBoundingClientRect().height||72);
    app.style.paddingBottom=`calc(${h+42}px + env(safe-area-inset-bottom))`;
  }

  function boot(){
    repair();forceBottomClearance();
    requestAnimationFrame(()=>{repair();forceBottomClearance()});
    setTimeout(()=>{repair();forceBottomClearance()},500);
    setTimeout(()=>{repair();forceBottomClearance()},1800);

    window.addEventListener('pageshow',()=>{repair();forceBottomClearance()});
    window.addEventListener('resize',forceBottomClearance,{passive:true});
    window.addEventListener('orientationchange',()=>setTimeout(forceBottomClearance,250),{passive:true});
    document.addEventListener('visibilitychange',()=>{if(document.visibilityState==='visible'){repair();forceBottomClearance()}});
    document.addEventListener('click',()=>setTimeout(repair,0),true);

    const mo=new MutationObserver(()=>repair());
    mo.observe(document.documentElement,{subtree:true,childList:true,attributes:true,attributeFilter:['class','style']});
  }

  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',boot,{once:true});
  else boot();
})();

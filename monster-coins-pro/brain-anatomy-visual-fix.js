'use strict';
(()=>{
  const $=(q,r=document)=>r.querySelector(q);

  function injectCss(){
    if($('#ba2visualfixcss')) return;
    const s=document.createElement('style');
    s.id='ba2visualfixcss';
    s.textContent=`
      #brianAnatomyLive.show{pointer-events:auto!important}
      #brianAnatomyLive .ba2h{pointer-events:auto!important}
      #brianAnatomyLive .ba2btn{pointer-events:auto!important;cursor:pointer!important;position:relative;z-index:60}
      .ba2hero{background:radial-gradient(circle at 50% 44%,rgba(6,72,104,.42),rgba(2,11,21,.92) 72%)!important;isolation:isolate}
      .ba2heroimg{position:absolute;z-index:0;left:50%;top:50%;width:205%;max-width:none;height:auto;transform:translate(-50%,-50%);object-fit:cover;object-position:center;pointer-events:none;user-select:none;-webkit-user-drag:none;filter:saturate(1.08) contrast(1.04) brightness(.92);opacity:.98}
      .ba2hero:before{content:'';position:absolute;z-index:1;inset:0;background:radial-gradient(circle at 50% 45%,transparent 32%,rgba(1,8,16,.22) 70%,rgba(1,7,14,.66) 100%);pointer-events:none}
      .ba2hero .ba2tag,.ba2hero:after{z-index:3}
      @media(max-width:850px){.ba2heroimg{width:230%;top:48%}}
      @media(max-width:520px){.ba2heroimg{width:260%;top:48%}}
    `;
    document.head.appendChild(s);
  }

  function installImage(){
    const hero=$('.ba2hero');
    if(!hero || hero.querySelector('.ba2heroimg')) return;
    const img=document.createElement('img');
    img.className='ba2heroimg';
    img.alt='Brian gelişim anatomisi';
    img.decoding='async';
    img.loading='eager';
    img.src='/brian-anatomy-dashboard-reference.webp?v=20260913-visual2';
    img.onerror=()=>{
      if(!img.dataset.fallback){
        img.dataset.fallback='1';
        img.src='/brian-anatomy-reference.webp?v=20260913-visual2';
      }else{
        img.style.display='none';
        hero.style.background='radial-gradient(circle at 50% 42%,#0d5b7a66,#020b15 55%,#010611 100%)';
      }
    };
    hero.prepend(img);
  }

  function closePanel(){
    const m=$('#brianAnatomyLive');
    if(!m) return;
    m.classList.remove('show','bal-show');
    m.style.pointerEvents='none';
    document.body.style.overflow='';
    document.documentElement.style.overflow='';
  }

  function wireButtons(){
    const m=$('#brianAnatomyLive');
    if(!m) return;
    const close=$('#ba2close');
    const refreshBtn=$('#ba2refresh');
    if(close && !close.dataset.visualFix){
      close.dataset.visualFix='1';
      close.addEventListener('click',(e)=>{
        e.preventDefault();e.stopPropagation();
        closePanel();
      },true);
    }
    if(refreshBtn && !refreshBtn.dataset.visualFix){
      refreshBtn.dataset.visualFix='1';
      refreshBtn.addEventListener('click',()=>{
        refreshBtn.animate?.([{transform:'rotate(0deg)'},{transform:'rotate(360deg)'}],{duration:650,easing:'ease-out'});
      },true);
    }
  }

  function repair(){
    injectCss();
    installImage();
    wireButtons();
    const m=$('#brianAnatomyLive');
    if(m?.classList.contains('show')) m.style.pointerEvents='auto';
  }

  function boot(){
    repair();
    requestAnimationFrame(repair);
    setTimeout(repair,250);
    setTimeout(repair,1200);
    document.addEventListener('click',()=>setTimeout(repair,0),true);
    const mo=new MutationObserver(()=>requestAnimationFrame(repair));
    mo.observe(document.body,{childList:true,subtree:true});
  }

  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',boot,{once:true});
  else boot();
})();

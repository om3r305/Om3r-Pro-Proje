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
      .ba2hero{background:radial-gradient(circle at 50% 46%,rgba(3,57,87,.48),rgba(2,11,21,.94) 68%)!important;isolation:isolate}
      .ba2heroimg{position:absolute;z-index:0;left:50%;top:50%;width:min(78%,560px);max-width:none;height:auto;transform:translate(-50%,-50%);object-fit:contain;object-position:center;pointer-events:none;user-select:none;-webkit-user-drag:none;filter:drop-shadow(0 0 20px rgba(29,213,255,.28)) saturate(1.08) contrast(1.05);opacity:1}
      .ba2hero:before{content:'';position:absolute;z-index:1;inset:0;background:radial-gradient(circle at 50% 48%,transparent 30%,rgba(1,8,16,.10) 62%,rgba(1,7,14,.42) 100%);pointer-events:none}
      .ba2hero .ba2tag,.ba2hero:after{z-index:3}
      @media(max-width:850px){.ba2heroimg{width:min(72%,480px);top:50%}}
      @media(max-width:520px){.ba2heroimg{width:72%;top:50%}}
    `;
    document.head.appendChild(s);
  }

  function installImage(){
    const hero=$('.ba2hero');
    if(!hero) return;
    let img=hero.querySelector('.ba2heroimg');
    if(!img){
      img=document.createElement('img');
      img.className='ba2heroimg';
      img.alt='Brian nöral gelişim anatomisi';
      img.decoding='async';
      img.loading='eager';
      hero.prepend(img);
    }
    if(!img.src.includes('/brian-neural-brain.webp')) img.src='/brian-neural-brain.webp?v=20260913-neural1';
    img.onerror=()=>{
      img.style.display='none';
      hero.style.background='radial-gradient(circle at 50% 42%,#0d5b7a66,#020b15 55%,#010611 100%)';
    };
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
      close.addEventListener('click',(e)=>{e.preventDefault();e.stopPropagation();closePanel()},true);
    }
    if(refreshBtn && !refreshBtn.dataset.visualFix){
      refreshBtn.dataset.visualFix='1';
      refreshBtn.addEventListener('click',()=>{
        refreshBtn.animate?.([{transform:'rotate(0deg)'},{transform:'rotate(360deg)'}],{duration:650,easing:'ease-out'});
      },true);
    }
  }

  function repair(){
    injectCss();installImage();wireButtons();
    const m=$('#brianAnatomyLive');
    if(m?.classList.contains('show')) m.style.pointerEvents='auto';
  }

  function boot(){
    repair();requestAnimationFrame(repair);setTimeout(repair,250);setTimeout(repair,1200);
    document.addEventListener('click',()=>setTimeout(repair,0),true);
    const mo=new MutationObserver(()=>requestAnimationFrame(repair));
    mo.observe(document.body,{childList:true,subtree:true});
  }

  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',boot,{once:true});
  else boot();
})();

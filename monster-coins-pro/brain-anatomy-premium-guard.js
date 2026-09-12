'use strict';
(()=>{
  const Native=window.MutationObserver;
  if(!Native||window.__bapObserverGuard) return;
  window.__bapObserverGuard=true;
  window.MutationObserver=class BrianPremiumObserver extends Native{
    constructor(callback){
      super((mutations,observer)=>{
        const meaningful=mutations.filter(m=>Array.from(m.addedNodes||[]).some(n=>{
          if(n.nodeType!==1) return true;
          const el=n;
          if(el.matches?.('.bap-mini,.bap-corner,.bap-extra,.bap-module,.bap-premium-human,.bap-radar-svg')) return false;
          if(el.closest?.('.bap-mini,.bap-extra,.bap-module,.bap-premium-human')) return false;
          return true;
        }));
        if(meaningful.length) callback(meaningful,observer);
      });
    }
  };
})();
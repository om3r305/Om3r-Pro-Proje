(function(root){
 'use strict';
 function createPacer(fps=60){const period=1000/fps;let deadline=null;return{due(now){if(!Number.isFinite(now))return false;if(deadline===null||now-deadline>250){deadline=now+period;return true;}if(now+.35<deadline)return false;deadline+=period;return true;},reset(){deadline=null;}};}
 const api={createPacer};if(typeof module==='object'&&module.exports)module.exports=api;else root.RenderPacing=api;
})(typeof window==='object'?window:globalThis);

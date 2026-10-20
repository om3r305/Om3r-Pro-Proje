'use strict';
(function(root){
  const number=value=>typeof value==='number'&&Number.isFinite(value)?value:typeof value==='string'&&value.trim()!==''&&Number.isFinite(Number(value))?Number(value):null;
  function radar(record,now=Date.now()){
    const raw=Array.isArray(record?.hot)&&record.hot.length?record.hot:Array.isArray(record?.candidates)?record.candidates:[];
    const at=record?.observed_at||record?.updated_at||record?.generated_at;
    const stamp=Date.parse(at||'');
    const current=record?.status==='ONLINE'&&record?.stale!==true&&Number.isFinite(stamp)&&now-stamp>=0&&now-stamp<=15*60000;
    const rows=raw.filter(row=>row&&typeof row==='object'&&!Array.isArray(row)).slice(0,8).map(row=>({symbol:String(row.symbol||row.base_asset||'Varlık bilinmiyor'),letter:String(row.base_asset||row.symbol||'?').slice(0,1),change:number(row.price_change_pct)}));
    return{current,at:Number.isFinite(stamp)?stamp:null,label:!record?'Veri bekleniyor':current?'24 saat · güncel kayıt':'24 saat · güncellik doğrulanmadı',rows};
  }
  const api={radar};if(typeof module==='object'&&module.exports)module.exports=api;else root.BrianTerminalModel=api;
})(typeof window==='object'?window:globalThis);

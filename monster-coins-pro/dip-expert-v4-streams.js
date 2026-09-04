function v4UpdateBar(sym,tf,k){
  v4Ensure(sym);const a=v4Bars[sym][tf],x={t:Number(k.t),o:Number(k.o),h:Number(k.h),l:Number(k.l),c:Number(k.c),v:Number(k.v),closed:Boolean(k.x)},i=a.findIndex(z=>z.t===x.t);if(i>=0)a[i]=x;else a.push(x);const max=tf==='1m'?760:tf==='5m'?200:tf==='15m'?140:90;if(a.length>max)a.splice(0,a.length-max);if(tf==='1m'){candles[sym]=a.slice(-180);live[sym]=x.c;}
}
function v4SpotStreams(symbols){
  return symbols.flatMap(s=>{const l=s.toLowerCase();return[`${l}@aggTrade`,`${l}@bookTicker`,`${l}@depth5@1000ms`,`${l}@kline_1m`,`${l}@kline_5m`,`${l}@kline_15m`,`${l}@kline_1h`]});
}
function v4PerpStreams(symbols){return symbols.filter(s=>v4FuturesSymbols.has(s)).flatMap(s=>{const l=s.toLowerCase();return[`${l}@aggTrade`,`${l}@bookTicker`,`${l}@depth5@1000ms`]});}
function v4HandleWsMessage(m,venue){
  const sym=String(m.s||m.data?.s||'');if(!sym)return;lastWs=v4Now();
  if(m.e==='aggTrade'){v4TapePush(venue==='USDM_PERP'?v4TapePerp:v4TapeSpot,sym,m);if(venue==='SPOT')tick(sym,Number(m.p));}
  else if(m.e==='kline'&&venue==='SPOT'){v4UpdateBar(sym,m.k.i,m.k);if(sym===selected)_v4PrevDraw();}
  else if(m.e==='bookTicker'||(m.b!=null&&m.a!=null&&!Array.isArray(m.b))){v4BookUpdate(venue==='USDM_PERP'?v4PerpBook:v4SpotBook,sym,m);if(venue==='SPOT'&&Number(m.b)>0&&Number(m.a)>0){live[sym]=(Number(m.b)+Number(m.a))/2;v4Evaluate(sym,live[sym]);}}
  else if(m.lastUpdateId!=null||m.e==='depthUpdate')v4BookUpdate(venue==='USDM_PERP'?v4PerpBook:v4SpotBook,sym,m);
  const now=v4Now();if(now-v4RenderAt>450){v4RenderAt=now;renderRadar();renderKpi();if(sym===selected)_v4PrevDraw();}
}
function v4ScheduleReconnect(){if(v4ReconnectTimer)return;v4ReconnectTimer=setTimeout(()=>{v4ReconnectTimer=null;connect()},1800);}
connect=function(){
  const gen=++v4WsGeneration;try{ws?.close()}catch{}try{v4SpotWs?.close()}catch{}try{v4PerpWs?.close()}catch{};
  const sy=[...new Set(['BTCUSDT',...v4AllSymbols()])].slice(0,15),spot=v4SpotStreams(sy),perp=v4PerpStreams(sy.filter(s=>s!=='BTCUSDT'));
  v4SpotWs=new WebSocket(`wss://stream.binance.com:9443/stream?streams=${spot.join('/')}`);ws=v4SpotWs;
  v4SpotWs.onopen=()=>{if(gen!==v4WsGeneration)return;if($('onlineText'))$('onlineText').textContent=`V4 MICRO LIVE · ${v4Universe.length}`;$('onlinePill')?.querySelector('.dot')?.classList.remove('off');$('marketDotSide')?.classList.remove('off');if($('marketTextSide'))$('marketTextSide').textContent=`Binance microstructure · ${v4Universe.length} trade market`;if($('feedState')){$('feedState').textContent='MICRO LIVE';$('feedState').className='pos'}};
  v4SpotWs.onmessage=e=>{if(gen!==v4WsGeneration)return;try{v4HandleWsMessage(JSON.parse(e.data).data||JSON.parse(e.data),'SPOT')}catch{}};
  v4SpotWs.onerror=()=>{try{v4SpotWs.close()}catch{}};v4SpotWs.onclose=()=>{if(gen===v4WsGeneration)v4ScheduleReconnect()};
  if(perp.length){v4PerpWs=new WebSocket(`wss://fstream.binance.com/stream?streams=${perp.join('/')}`);v4PerpWs.onmessage=e=>{if(gen!==v4WsGeneration)return;try{v4HandleWsMessage(JSON.parse(e.data).data||JSON.parse(e.data),'USDM_PERP')}catch{}};v4PerpWs.onerror=()=>{};v4PerpWs.onclose=()=>{};}
};

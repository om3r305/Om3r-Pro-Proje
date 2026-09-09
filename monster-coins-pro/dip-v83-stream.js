'use strict';
// Public market display only: no account, order or worker execution endpoints.
(function(){
  const URL='wss://fstream.binance.com/market/stream?streams=ethusdt@aggTrade/ethusdt@kline_1m/ethusdt@markPrice@1s';
  let socket=null,retry=null,retries=0,lastRest=0,lastPaint=0,lastTrade=null,lastKline=null,lastMark=null,connectedAt=0;
  const seen={};
  const restLoad=loadChart,draw=renderChart,restFresh=chartFresh;
  const fresh=x=>x&&Date.now()-x.at<5000&&Date.now()-x.eventAt<5000;
  function overlay(){
    if(!model.chart)return;
    const rows=model.chart.candles;
    if(fresh(lastKline)&&Array.isArray(rows)&&rows.length){
      const k=lastKline.bar,last=rows.at(-1);
      if(k.t===last.t)rows[rows.length-1]=k;
      else if(k.t===last.t+60000){rows.push(k);if(rows.length>180)rows.shift();}
      else if(k.t>last.t+60000)lastRest=0; // REST repairs missing history; never fabricate bars.
    }
    if(fresh(lastTrade))model.chart.last_price=lastTrade.price;
    else if(fresh(lastKline))model.chart.last_price=lastKline.bar.c;
  }
  function labels(){
    const live=socket?.readyState===1&&fresh(lastTrade),badge=$('chartSource'),bar=$('freshnessBar'),mark=$('markPriceText');
    if(badge)badge.textContent=live?'CANLI · USD-M SON FİYAT':'REST · CANLI AKIŞ BEKLENİYOR';
    if(mark)mark.textContent=fresh(lastMark)?`Mark fiyatı ${px(lastMark.price)} · Son işlem fiyatından farklı olabilir`:'Mark fiyatı bekleniyor · Grafik USD-M son işlemleri gösterir';
    const sr=runtime(),at=Date.parse(sr?.generated_at||''),age=Number.isFinite(at)?Math.max(0,Math.round((Date.now()-at)/1000)):'—';
    const lag=live?Math.max(0,Date.now()-lastTrade.eventAt):null;
    if(bar){bar.textContent=`${live?`Borsa mesajı ${lag} ms önce`:'Anlık bağlantı yok; REST görüntüsü'} · Brian kararı ${age} sn önce (60 sn döngü)`;bar.style.color=live?'#91a2b8':'#f0b90b';}
  }
  renderChart=function(){overlay();draw();labels();};
  chartFresh=function(){return !!(socket?.readyState===1&&fresh(lastTrade)&&fresh(lastKline))||restFresh();};
  loadChart=async function(){
    if(fresh(lastTrade)&&fresh(lastKline)&&model.chart?.candles?.length&&Date.now()-lastRest<60000)return;
    if(chartBusy)return;lastRest=Date.now();await restLoad();
  };
  function paint(){if(Date.now()-lastPaint<250)return;lastPaint=Date.now();renderChart();}
  function connect(){
    if(document.hidden||socket||retry)return;
    const ws=new WebSocket(URL);socket=ws;connectedAt=Date.now();
    ws.onmessage=function(ev){
      if(socket!==ws||document.hidden)return;
      let d;try{const msg=JSON.parse(ev.data);d=msg.data||msg;}catch{return;}
      if(d.s!=='ETHUSDT'||(d.st!=null&&d.st!==1))return;
      const eventAt=Number(d.E);if(!Number.isFinite(eventAt)||eventAt>Date.now()+2000||Date.now()-eventAt>10000||eventAt<(seen[d.e]||0))return;
      seen[d.e]=eventAt;const at=Date.now();
      if(d.e==='aggTrade'&&Number(d.p)>0&&Number.isFinite(Number(d.p))){lastTrade={price:Number(d.p),eventAt,at};}
      else if(d.e==='markPriceUpdate'&&Number(d.p)>0&&Number.isFinite(Number(d.p))){lastMark={price:Number(d.p),eventAt,at};}
      else if(d.e==='kline'&&d.k?.i==='1m'){
        const k=d.k,b={t:Number(k.t),ct:Number(k.T),o:Number(k.o),h:Number(k.h),l:Number(k.l),c:Number(k.c),v:Number(k.v)};
        if(!Object.values(b).every(Number.isFinite)||b.t%60000||b.ct!==b.t+59999||b.l<=0||b.h<Math.max(b.o,b.c)||b.l>Math.min(b.o,b.c)||b.v<0)return;
        if(lastKline&&b.t<lastKline.bar.t)return;
        lastKline={bar:b,eventAt,at};
      }else return;
      retries=0;paint();
    };
    ws.onclose=function(){if(socket!==ws)return;socket=null;labels();if(!document.hidden){retry=setTimeout(()=>{retry=null;connect();},Math.min(30000,1000*2**Math.min(retries++,5)));}};
    ws.onerror=function(){ws.close();};
  }
  function suspend(){clearTimeout(retry);retry=null;const ws=socket;socket=null;if(ws)ws.close();}
  document.addEventListener('visibilitychange',()=>{if(document.hidden)suspend();else{lastRest=0;loadChart();connect();}});
  window.addEventListener('pagehide',suspend);
  window.addEventListener('pageshow',connect);
  setInterval(()=>{if(document.hidden)return;labels();if(socket&&Date.now()-Math.max(connectedAt,lastTrade?.at||0)>20000)socket.close();},1000);
  connect();
})();

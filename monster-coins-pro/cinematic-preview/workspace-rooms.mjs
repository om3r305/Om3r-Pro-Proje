export const ROOMS=[
 {id:'meeting',label:'Toplantı'}, {id:'accounting',label:'Muhasebe'},
 {id:'engineering',label:'Yazılım'}, {id:'anatomy',label:'Gelişim'},
 {id:'research',label:'Araştırma'}, {id:'performance',label:'Performans'},
 {id:'news',label:'Haberler'}, {id:'system',label:'Sistem durumu'}
];
export function roomUrl(id){return ROOMS.some(r=>r.id===id)?'/frontier-v4?room='+encodeURIComponent(id):null;}
export function createWorkspaceRooms({onOpen=()=>{},onClose=()=>{}}={}){
 const skin=document.createElement('link');skin.rel='stylesheet';skin.href='/brian-rooms.css?v=20261022-rooms1';document.head.append(skin);
 const bar=document.createElement('div');bar.className='integrated-rooms';bar.setAttribute('role','navigation');bar.setAttribute('aria-label','BRIAN bağlı çalışma alanları');
 bar.innerHTML='<span>BRIAN ODALARI</span>'+ROOMS.map(r=>`<button type="button" data-room="${r.id}">${r.label} ↗</button>`).join('');document.querySelector('.masthead').after(bar);
 const dialog=document.createElement('dialog');dialog.id='roomDialog';dialog.setAttribute('aria-labelledby','roomTitle');
 dialog.innerHTML='<header><div><p class="room-eyebrow">BRIAN / ORGANIC INTELLIGENCE</p><h2 id="roomTitle"></h2><p id="roomStatus" role="status"></p></div><div class="room-actions"><button id="roomReload">Yenile</button><a id="roomExternal" target="_blank" rel="noopener">Ayrı sekmede aç ↗</a><button id="roomClose" aria-label="Çalışma alanını kapat">✕</button></div></header><nav class="room-switcher" aria-label="Çalışma odasını değiştir">'+ROOMS.map((r,i)=>`<button type="button" data-room="${r.id}"><small>0${i+1}</small>${r.label}</button>`).join('')+'</nav><iframe id="roomFrame" title="Mevcut BRIAN çalışma alanı"></iframe>';
 document.body.append(dialog);const frame=dialog.querySelector('iframe'),status=dialog.querySelector('#roomStatus');let selected=null,timer=null,opener=null;
 function load(){clearTimeout(timer);status.textContent='Mevcut BRIAN odası bağlanıyor…';frame.src=roomUrl(selected.id);timer=setTimeout(()=>{status.textContent='Odanın açılışı henüz doğrulanmadı. Kilidi veya bağlantıyı ayrı sekmeden kontrol edebilirsin.';},30000);}
 function open(id){const room=ROOMS.find(r=>r.id===id);if(!room)return;if(!dialog.open)opener=document.activeElement;selected=room;dialog.querySelector('#roomTitle').textContent=room.label;dialog.querySelector('#roomExternal').href=roomUrl(id);frame.title='BRIAN / '+room.label;dialog.querySelectorAll('.room-switcher button').forEach(b=>{if(b.dataset.room===id)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current');});if(!dialog.open){dialog.showModal();onOpen();}load();dialog.querySelector('#roomClose').focus();}
 document.addEventListener('click',e=>{const b=e.target.closest('[data-room]');if(b)open(b.dataset.room);});
 window.addEventListener('message',e=>{if(e.origin!==location.origin||e.source!==frame.contentWindow||e.data?.type!=='brian-room-status'||e.data.room!==selected?.id)return;clearTimeout(timer);status.textContent=e.data.state==='ready'?'Mevcut BRIAN odası açıldı · veri durumu odanın içinde gösterilir.':e.data.state==='locked'?'Dashboard kilitli · mevcut anahtarı ayrı sekmede açabilirsin.':'Oda yüklenemedi · yeniden dene veya ayrı sekmede aç.';});
 dialog.querySelector('#roomReload').onclick=load;dialog.querySelector('#roomClose').onclick=()=>dialog.close();
 dialog.addEventListener('close',()=>{clearTimeout(timer);frame.removeAttribute('src');onClose();opener?.focus?.({preventScroll:true});});return {open};
}

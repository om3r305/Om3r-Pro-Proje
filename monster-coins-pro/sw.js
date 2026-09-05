const CACHE='monster-coins-pro-shell-v5';
const ASSETS=[
  '/','/app.css','/app.js','/manifest.webmanifest','/brand.svg','/logo.svg',
  '/dip.html','/dip.css','/dip.js',
  '/dip-expert-v2.js','/dip-expert-v3.js',
  '/dip-expert-v4-core.js','/dip-expert-v4-market.js','/dip-expert-v4-risk.js',
  '/dip-expert-v4-engine.js','/dip-expert-v4-streams.js','/dip-expert-v4-session.js',
  '/dip-expert-v4-ui.js','/dip-expert-v4-hotfix.js','/dip-expert-v5-brain.js'
];
self.addEventListener('install',event=>event.waitUntil(caches.open(CACHE).then(cache=>cache.addAll(ASSETS)).then(()=>self.skipWaiting())));
self.addEventListener('activate',event=>event.waitUntil(Promise.all([
  self.clients.claim(),
  caches.keys().then(keys=>Promise.all(keys.filter(key=>key!==CACHE).map(key=>caches.delete(key))))
])));
self.addEventListener('fetch',event=>{
  if(event.request.method!=='GET') return;
  event.respondWith(fetch(event.request).then(response=>{
    const copy=response.clone();
    caches.open(CACHE).then(cache=>cache.put(event.request,copy));
    return response;
  }).catch(()=>caches.match(event.request).then(response=>response||caches.match('/'))));
});
self.addEventListener('message',event=>{
  if(event.data?.type!=='NOTIFY') return;
  const {title,body,tag}=event.data;
  event.waitUntil(self.registration.showNotification(title||'Monster Coins Pro',{
    body:body||'',
    icon:'/logo.svg',
    badge:'/logo.svg',
    tag:tag||'monster-coins-pro',
    renotify:true,
    data:{url:'/'}
  }));
});
self.addEventListener('notificationclick',event=>{
  event.notification.close();
  event.waitUntil(clients.matchAll({type:'window',includeUncontrolled:true}).then(list=>{
    const existing=list.find(client=>'focus' in client);
    return existing?existing.focus():clients.openWindow('/');
  }));
});

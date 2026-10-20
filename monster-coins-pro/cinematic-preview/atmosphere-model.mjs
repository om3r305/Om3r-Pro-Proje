// Presentation state only. Never authorizes execution or infers profitability.
export function atmosphereFor(raw, graph, replay=false) {
 const mood=(key,label,reason,path,color,power=.35)=>({key,label,reason,path,color,power});
 if(replay)return mood('replay','KAYIT İNCELEME','Atmosfer kaydedilmiş yanıta aittir.','/','#a6b3d4');
 if(!graph?.valid||!graph.fresh)return mood('unverified','VERİ DOĞRULANMADI','Güncel kaynak doğrulanmadan fırsat veya kritik olay çıkarılmaz.','/observed_at','#9ba9bd',.15);
 const current=graph.nodes.filter(n=>n.state==='current'||n.state==='failed'&&Number.isFinite(Date.parse(n.at||''))&&Date.now()-Date.parse(n.at)>=0&&Date.now()-Date.parse(n.at)<=1200000);
 const critical=current.find(n=>['CRITICAL','EMERGENCY'].includes(String(n.raw.severity||'').toUpperCase())&&n.state==='current');
 if(critical)return mood('critical','KRİTİK KAYNAK OLAYI',critical.label+' / '+critical.raw.severity,critical.path,'#ff354f',1);
 const failed=current.find(n=>n.state==='failed');
 if(failed)return mood('fault','MODÜL HATASI',failed.label+' / '+failed.raw.status+' — piyasa krizi anlamına gelmez.',failed.path,'#ff354f',.8);
 const alpha=current.find(n=>n.id==='alpha'&&n.state==='current');
 const warning=current.find(n=>n.state==='current'&&['WARNING','HIGH'].includes(String(n.raw.severity||'').toUpperCase()));
 if(warning)return mood('caution','KAYNAK UYARISI',warning.label+' / '+warning.raw.severity,warning.path,'#ffc342',.7);
 if(alpha&&['VETO','BLOCKED'].includes(raw.alpha.action))return mood('caution','KARAR ENGELİ / SHADOW',raw.alpha.action+' — fırsat doğrulanmış değil.','/alpha/action','#ffc342',.55);
 if(alpha&&['OPEN_LONG','OPEN_SHORT'].includes(raw.alpha.action)){
  const strong=raw.alpha.opportunity_strength==='STRONG';
  return mood(strong?'opportunity':'candidate',strong?'GÜÇLÜ ADAY / SHADOW':'SİNYAL ADAYI / SHADOW',strong?'Kaynak fırsat gücünü STRONG olarak bildirdi. Başarı garantisi değildir.':'Kaynak açılış adayı bildirdi; güçlü fırsat veya kazanç doğrulanmadı.','/alpha',strong?'#46ffb0':'#79dcc0',strong?1:.45);
 }
 return mood('neutral','NORMAL / GÖZLEM','Doğrulanmış kritik olay veya güçlü fırsat alanı yok.','/','#e6f1ff',.3);
}


export async function fetchJSONL(url, maxLines=10000){
  const res = await fetch(url, {cache:'no-store'});
  if(!res.ok) throw new Error('fetch '+url+' '+res.status);
  const text = await res.text();
  const lines = text.split(/\r?\n/).filter(Boolean);
  return lines.slice(-maxLines).map(l=>{ try{return JSON.parse(l)}catch(e){return null}}).filter(Boolean);
}

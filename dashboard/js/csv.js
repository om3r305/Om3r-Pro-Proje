
export function parseCSV(text){
  const lines = text.trim().split(/\r?\n/);
  const headers = lines.shift().split(',');
  return lines.map(l=>{
    const parts = l.split(',');
    const obj = {};
    headers.forEach((h,i)=>obj[h.trim()]=(parts[i]||'').trim());
    return obj;
  });
}

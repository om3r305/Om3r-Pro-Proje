// Summaries contain only scalar fields actually present in the selected record.
export function recordSummary(raw={},limit=4){
 const fields=Object.entries(raw).filter(([,v])=>v!==null&&v!==undefined&&['number','string','boolean'].includes(typeof v)&&v!==''&&(typeof v!=='number'||Number.isFinite(v)));
 const numeric=fields.filter(([k,v])=>typeof v==='number'&&k!=='direction');
 return (numeric.length?numeric:fields).slice(0,limit);
}

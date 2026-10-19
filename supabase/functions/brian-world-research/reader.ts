import { parseHTML } from 'npm:linkedom@0.18.12';
import { officialUri } from './logic.ts';
type ParsedNode={textContent:string|null;remove():void;getAttribute(name:string):string|null};
type ParsedDocument={querySelector(selector:string):ParsedNode|null;querySelectorAll(selector:string):Iterable<ParsedNode>;body:ParsedNode|null;documentElement:ParsedNode|null};
export function documentText(html:string, uri:string) {
  // Linkedom's Window declaration relies on browser DOM globals absent in Deno.
  const {document}=parseHTML(html) as unknown as {document:ParsedDocument};
  for(const n of document.querySelectorAll('script,style,noscript,iframe,nav,header,footer,form,svg,template'))n.remove();
  const title=(document.querySelector('h1')?.textContent||document.querySelector('title')?.textContent||'').trim();
  const content=document.querySelector('article')||document.querySelector('main')||document.querySelector('[role="main"]')||document.body;
  const text=(content?.textContent||document.documentElement?.textContent||'').replace(/\s+/g,' ').trim();
  if(text.length<120||/request rate threshold exceeded|undeclared automated tool|verify you are human|access denied/i.test(title+' '+text.slice(0,800)))throw Error('OFFICIAL_DOCUMENT_NOT_READABLE');
  const indexOnly=/-index\.htm(?:l)?$/i.test(new URL(uri).pathname);
  const links=[...document.querySelectorAll('table.tableFile a[href]')].map(a=>a.getAttribute('href')).filter((x):x is string=>!!x);
  const domain=new URL(uri).hostname.replace(/^www\./,'');
  const primaryDocument=indexOnly?links.flatMap(href=>{const target=officialUri(new URL(href,uri).toString(),domain);return target&&/\/Archives\/edgar\/data\//.test(new URL(target).pathname)&&!/(-index\.htm|\.xsd|\.zip|\.jpg|\.pdf)/i.test(target)&&/\.(htm|html|xml)$/i.test(new URL(target).pathname)?[target]:[];})[0]??null:null;
  return{title,text,indexOnly,primaryDocument};
}

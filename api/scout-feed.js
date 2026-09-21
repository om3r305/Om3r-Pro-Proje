const {createHash,timingSafeEqual}=require('node:crypto');
const QUERIES={"discovery:geopolitics": "(war OR missile OR drone OR sanctions OR ceasefire OR invasion OR escalation OR retaliation OR \"Red Sea\" OR Taiwan OR shipping OR blockade OR attack) when:1h", "discovery:energy": "(oil OR Brent OR WTI OR OPEC OR \"natural gas\" OR LNG OR refinery OR pipeline OR tanker OR terminal OR \"energy supply\" OR Hormuz OR \"Bab el-Mandeb\") when:1h", "discovery:middle-east-risk": "(Iran OR Houthi OR Saudi OR Riyadh OR Yanbu OR Hormuz OR \"Bab el-Mandeb\" OR \"Red Sea\") (attack OR missile OR drone OR escalation OR retaliation OR tanker OR port OR refinery OR shipping) when:1h", "discovery:reuters-market-wire": "site:reuters.com (Iran OR Houthi OR Saudi OR Riyadh OR Hormuz OR oil OR refinery OR missile OR drone OR sanctions OR \"Federal Reserve\" OR SEC OR NVIDIA) when:1h", "discovery:macro": "(\"Federal Reserve\" OR ECB OR inflation OR CPI OR PCE OR payrolls OR unemployment OR \"interest rate\" OR yields OR GDP) when:1h", "discovery:ai": "(NVIDIA OR OpenAI OR semiconductor OR GPU OR TSMC OR HBM OR \"memory chip\" OR \"AI datacenter\" OR \"export control\") when:1h"};
const INTERNAL_HASH='b0549b2b41a5b832b37455389583e1d166d210490a8c6fe43cda2748aca7c38a';
function createHandler(fetchFeed=fetch,expectedHash=INTERNAL_HASH){
 return async function handler(req,res){
  res.setHeader('Cache-Control','no-store');
  if(req.method!=='GET')return res.status(405).json({error:'METHOD_NOT_ALLOWED'});
  const key=req.headers['x-brian-internal-key'];
  if(typeof key!=='string'||!timingSafeEqual(createHash('sha256').update(key.trim()).digest(),Buffer.from(expectedHash,'hex')))return res.status(401).json({error:'UNAUTHORIZED'});
  const lane=req.query.lane;
  if(typeof lane!=='string'||!Object.hasOwn(QUERIES,lane))return res.status(400).json({error:'UNKNOWN_LANE'});
  const url='https://news.google.com/rss/search?'+new URLSearchParams({q:QUERIES[lane],hl:'en-US',gl:'US',ceid:'US:en'});
  try{
   const response=await fetchFeed(url,{headers:{accept:'application/rss+xml,application/xml'},signal:AbortSignal.timeout(12000),redirect:'error'});
   if(!response.ok)return res.status(502).json({error:'NEWS_PROVIDER_HTTP_'+response.status});
   const xml=await response.text();
   if(xml.length>2000000||!/<rss\b|<feed\b/i.test(xml))return res.status(502).json({error:'INVALID_NEWS_FEED'});
   res.setHeader('Content-Type','application/rss+xml; charset=utf-8');return res.status(200).send(xml);
  }catch{return res.status(502).json({error:'NEWS_PROVIDER_UNAVAILABLE'});}
 };
}
module.exports=createHandler();

const intersects=(a,b,gap=7)=>Math.abs(a.x-b.x)<(a.w+b.w)/2+gap&&Math.abs(a.y-b.y)<(a.h+b.h)/2+gap;
// Preserve projected anchors whenever free; densely projected nodes use the nearest free slot.
export function resolveDesktopLabels(positions,width,height,obstacles=[]){
 const result=positions.map(p=>({...p})),visible=result.filter(p=>p.visible),used=[];
 const maxW=Math.max(1,...visible.map(p=>p.w)),maxH=Math.max(1,...visible.map(p=>p.h));
 const rows=[],cols=[];for(let x=maxW/2+8;x<=width-maxW/2-8;x+=maxW+9)cols.push(x);
 for(let y=120;y<=height-185;y+=maxH+9)rows.push(y);
 const valid=p=>p.x>=p.w/2+8&&p.x<=width-p.w/2-8&&p.y>=120&&p.y<=height-185&&!used.some(q=>intersects(p,q))&&!obstacles.some(q=>intersects(p,q));
 for(const p of visible){
  if(!valid(p)){let best=null,score=Infinity;for(const y of rows)for(const x of cols){const candidate={...p,x,y},distance=(x-p.x)**2+(y-p.y)**2;if(distance<score&&valid(candidate)){score=distance;best=candidate;}}if(best){p.x=best.x;p.y=best.y;}else p.visible=false;}
  if(p.visible)used.push(p);
 }
 return result;
}

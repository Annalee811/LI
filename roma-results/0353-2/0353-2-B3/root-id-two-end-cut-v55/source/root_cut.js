'use strict';
types52.cut='手动断开点';signs52.cut='|';
let cutSelection55=null,cutPreview55=null,cutRoot55=null;
const cutOption55=document.createElement('option');cutOption55.value='cut_two55';cutOption55.textContent='两端断开：选择起点与终点';$('tool52').append(cutOption55);
const cutPanel55=document.createElement('div');cutPanel55.id='twoEndCutPanel55';cutPanel55.innerHTML=`<details open><summary>两端断开并自动登记编号</summary><button id="startCut55">选择断开的两端</button><button id="resetCut55">重选两端</button><div id="cutInfo55">先选择根，依次点击中间段的起点、终点；蓝线为预览。</div><label>断开后归入编号（可空）<input id="cutTarget55" type="text" placeholder="留空自动编号，也可填 0001-1／0001-N1"></label><label><input type="checkbox" id="syncCut55" checked>可靠匹配的时期同步断开并续号</label><button id="applyCut55" disabled>确认断开中间段</button><small>只断开结构连接，不删除原根像素。中间段自动注册，之后改号或设置母根关系即可纳入已有编号体系；不自动推断它一定是侧根。</small></details>`;
$('familyPanel54').after(cutPanel55);
const quickCut55=document.createElement('button');quickCut55.id='quickCut55';quickCut55.textContent='两端断开';$('connectionEdit53').after(quickCut55);quickCut55.onclick=()=>{$('startCut55').click();cutPanel55.scrollIntoView({block:'nearest'});};
const roleLabel55=document.createElement('label');roleLabel55.innerHTML='中间段身份<select id="cutRole55"><option value="main">母根本体／续用指定根编号</option><option value="existing">已有侧根（母根号 → -1、-2…）</option><option value="new">新生侧根（母根号 → -N1、-N2…）</option><option value="pending">暂不归属，自动登记临时编号</option></select>';$('cutTarget55').closest('label').before(roleLabel55);
$('cutTarget55').placeholder='输入母根 0001，或已有完整侧根号 0001-N1';
let enteredMother55=null;$('motherLabel54').addEventListener('input',()=>{enteredMother55={root_id:selectedId55(),value:$('motherLabel54').value};});
function edgeKey55(a,b){return a<b?a+'|'+b:b+'|'+a;}
function key55(p){return Math.round(p[0])+','+Math.round(p[1]);}
function graph55(frame,rid){
 const nodes=new Map(),edges=new Map(),blocked=new Set((frame.manual_disconnects||[]).flatMap(c=>c.blocked_edges||[]).map(([a,b])=>edgeKey55(a,b)));
 const cuts=(frame.manual_disconnects||[]).flatMap(c=>c.points||[]);const segments=frame.segments.filter(s=>s.root_id===rid),byid=new Map(segments.map(s=>[s.id,s]));
 function add(p){const k=key55(p);if(!nodes.has(k)){nodes.set(k,[Math.round(p[0]),Math.round(p[1])]);edges.set(k,new Map())}return k;}
 function connect(a,b){if(a===b||blocked.has(edgeKey55(a,b)))return;const p=nodes.get(a),q=nodes.get(b),d=Math.hypot(p[0]-q[0],p[1]-q[1]);if(!d||d>26)return;edges.get(a).set(b,d);edges.get(b).set(a,d);}
 function path(points){let previous=null;for(const p of points){const k=add(p);if(previous)connect(previous,k);previous=k;}}
 for(const s of segments)path(s.points);
 const originalPoints=[...nodes.entries()];
 for(const stroke of frame.manual_root_strokes||[]){const owner=cache52(frame).owner;const p=stroke.points.filter(p=>cache52(frame).byIndex.get(owner[Math.round(p[1])*frame.width+Math.round(p[0])])?.root_id===rid);if(p.length){path(p);for(const end of [p[0],p.at(-1)]){let best=null,distance=64;for(const [k,q] of originalPoints){const d=(q[0]-end[0])**2+(q[1]-end[1])**2;if(d<distance){distance=d;best=[k,q]}}if(best){const q=best[1],steps=Math.max(2,Math.ceil(Math.sqrt(distance))),line=Array.from({length:steps+1},(_,i)=>[Math.round(end[0]+(q[0]-end[0])*i/steps),Math.round(end[1]+(q[1]-end[1])*i/steps)]);if(line.filter(c=>cache52(frame).byIndex.get(owner[c[1]*frame.width+c[0]])?.root_id===rid).length/line.length>=.8)connect(key55(end),best[0]);}}}}
 function nearest(s,x,y){let best=null,d=Infinity;for(const p of s.points){const v=(p[0]-x)**2+(p[1]-y)**2;if(v<d){d=v;best=key55(p)}}return best;}
 for(const n of frame.nodes)for(const pair of n.connections||[]){const a=n.arms.find(a=>a.id===pair[0]),b=n.arms.find(a=>a.id===pair[1]);const sa=byid.get(a?.segment_id),sb=byid.get(b?.segment_id);if(sa&&sb&&sa.id!==sb.id&&!cuts.some(p=>Math.hypot(p[0]-n.x,p[1]-n.y)<4))connect(nearest(sa,n.x,n.y),nearest(sb,n.x,n.y));}
 // Only bridge small, aligned gaps inside the already selected root identity.
 const ends=[];for(const s of segments)if(s.points.length>4)for(const side of [0,1]){const p=s.points[side?s.points.length-1:0],q=s.points[side?Math.max(0,s.points.length-8):Math.min(7,s.points.length-1)];if(cuts.some(c=>Math.hypot(c[0]-p[0],c[1]-p[1])<4))continue;const d=[p[0]-q[0],p[1]-q[1]],l=Math.hypot(...d)||1;ends.push({id:s.id,key:key55(p),p,d:d.map(v=>v/l)});}
 for(let i=0;i<ends.length;i++)for(let j=i+1;j<ends.length;j++){const a=ends[i],b=ends[j],dx=b.p[0]-a.p[0],dy=b.p[1]-a.p[1],dist=Math.hypot(dx,dy);if(a.id===b.id||dist>9||dist<.1)continue;const alignment=-(a.d[0]*b.d[0]+a.d[1]*b.d[1]);if(alignment>.85&&(a.d[0]*dx+a.d[1]*dy)/dist>.5&&(b.d[0]*-dx+b.d[1]*-dy)/dist>.5)connect(a.key,b.key);}
 return {nodes,edges};
}
function nearestGraph55(graph,p,max=13){let key=null,d=max*max;for(const [k,q] of graph.nodes){const v=(q[0]-p[0])**2+(q[1]-p[1])**2;if(v<d){d=v;key=k}}if(!key)throw Error('点击位置离所选根太远，请点在根本体上');return key;}
function pathBetween55(frame,rid,a,b){
 const graph=graph55(frame,rid),start=nearestGraph55(graph,a),end=nearestGraph55(graph,b);if(start===end)throw Error('两端太近，请选择两个不同位置');
 const distance=new Map([[start,0]]),previous=new Map(),heap=[[0,start]];
 function push(item){heap.push(item);let i=heap.length-1;while(i>0){const p=(i-1)>>1;if(heap[p][0]<=item[0])break;heap[i]=heap[p];i=p;}heap[i]=item;}
 function pop(){const first=heap[0],last=heap.pop();if(heap.length){let i=0;while(2*i+1<heap.length){let c=2*i+1;if(c+1<heap.length&&heap[c+1][0]<heap[c][0])c++;if(heap[c][0]>=last[0])break;heap[i]=heap[c];i=c;}heap[i]=last;}return first;}
 while(heap.length){const [d,k]=pop();if(d!==distance.get(k))continue;if(k===end)break;for(const [n,w] of graph.edges.get(k)){const value=d+w;if(value<(distance.get(n)??Infinity)){distance.set(n,value);previous.set(n,k);push([value,n]);}}}
 if(!distance.has(end))throw Error('两端之间没有已确认的连续路径。请先连接相关根段或补画缺失处，再断开。');
 const keys=[end];while(keys.at(-1)!==start)keys.push(previous.get(keys.at(-1)));keys.reverse();if(keys.length<4)throw Error('中间段太短，请重新选择两端');
 const set=new Set(keys),blocked=[];for(const key of [start,end])for(const neighbor of graph.edges.get(key).keys())if(!set.has(neighbor))blocked.push([key,neighbor]);
 return {keys,points:keys.map(k=>graph.nodes.get(k)),length:distance.get(end),blocked_edges:blocked};
}
function distanceToPoints55(points,x,y){let best=Infinity;for(const p of points){const d=(p[0]-x)**2+(p[1]-y)**2;if(d<best)best=d;}return best;}
function applyPartition55(frame,oldRoot,newRoot,plan){
 const selected=new Set(plan.keys),cache=cache52(frame),oldGroups=new Map(),newSegments=[],stamp=Date.now();let next=Math.max(...frame.segments.map(s=>s.index))+1;
 for(const s of frame.segments){if(s.root_id!==oldRoot||!s.points.some(p=>selected.has(key55(p)))){newSegments.push(s);continue;}
  const runs=[];let current=null;for(const p of s.points){const middle=selected.has(key55(p));if(!current||current.middle!==middle){current={middle,points:[]};runs.push(current)}current.points.push(p);}
  const preserved=runs.findIndex(r=>!r.middle);const pieces=[];
  for(let i=0;i<runs.length;i++){const run=runs[i],keep=i===(preserved<0?0:preserved),index=keep?s.index:next++,piece=structuredClone(s);piece.index=index;piece.id=keep?s.id:frame.period+'-S'+String(index).padStart(4,'0');piece.root_id=run.middle?newRoot:oldRoot;piece.points=run.points;piece.length_px=run.points.slice(1).reduce((n,p,j)=>n+Math.hypot(p[0]-run.points[j][0],p[1]-run.points[j][1]),0);piece.endpoint_kinds=[i===0?s.endpoint_kinds[0]:'manual_cut',i===runs.length-1?s.endpoint_kinds[1]:'manual_cut'];piece.review_status='modified';piece.manual_partition_from=s.id;pieces.push(piece);newSegments.push(piece);}
  oldGroups.set(s.index,pieces);
 }
 if(!oldGroups.size)throw Error('没有找到可分离的原根段，未修改');
 const closest=(pieces,x,y)=>pieces.reduce((best,p)=>distanceToPoints55(p.points,x,y)<distanceToPoints55(best.points,x,y)?p:best,pieces[0]);
 for(let i=0;i<cache.owner.length;i++){const pieces=oldGroups.get(cache.owner[i]);if(pieces)cache.owner[i]=closest(pieces,i%frame.width,Math.floor(i/frame.width)).index;}
 for(const n of frame.nodes){const source=cache.byId.get(n.owner_segment_id),pieces=source&&oldGroups.get(source.index);if(pieces)n.owner_segment_id=closest(pieces,n.x,n.y).id;for(const a of n.arms){const s=cache.byId.get(a.segment_id),pp=s&&oldGroups.get(s.index);if(pp)a.segment_id=closest(pp,a.x,a.y).id;}}
 frame.segments=newSegments;cache.byId=new Map(newSegments.map(s=>[s.id,s]));cache.byIndex=new Map(newSegments.map(s=>[s.index,s]));
 for(const n of frame.nodes)n.connections=n.connections.filter(pair=>{const a=n.arms.find(a=>a.id===pair[0]),b=n.arms.find(a=>a.id===pair[1]);const ra=cache.byId.get(a?.segment_id)?.root_id,rb=cache.byId.get(b?.segment_id)?.root_id;return !([ra,rb].includes(newRoot)&&[ra,rb].includes(oldRoot));});
 const middle=newSegments.filter(s=>s.root_id===newRoot);for(const [i,p] of [plan.points[0],plan.points.at(-1)].entries()){const owner=closest(middle,...p);frame.nodes.push({id:frame.period+'-C'+stamp+'-'+i,type:'cut',x:p[0],y:p[1],owner_segment_id:owner.id,arms:[],connections:[],review_status:'modified',ownership_radius:0,proposed:false,note:'用户选择两端断开；原根像素未删除'});}
 (frame.manual_disconnects??=[]).push({id:'CUT-'+stamp,old_root_id:oldRoot,new_root_id:newRoot,points:[plan.points[0],plan.points.at(-1)],blocked_edges:plan.blocked_edges,root_pixels_removed:0,review_status:'modified'});
 return middle[0].id;
}
function resetCut55(){cutSelection55=cutPreview55=cutRoot55=null;$('applyCut55').disabled=true;$('cutInfo55').textContent='先选择根，再依次点击起点、终点。';render();}
const previousRenderCut55=render;render=function(){previousRenderCut55();if(!active||!cutSelection55||$('tool52').value!=='cut_two55')return;const z=Math.max(.1,Number($('zoom').value)||1);if(cutPreview55){ovctx.beginPath();cutPreview55.points.forEach((p,i)=>i?ovctx.lineTo(...p):ovctx.moveTo(...p));ovctx.strokeStyle='#00eaff';ovctx.lineWidth=4/z;ovctx.stroke();}for(const p of cutPreview55?[cutPreview55.points[0],cutPreview55.points.at(-1)]:[cutSelection55]){ovctx.beginPath();ovctx.arc(...p,7/z,0,Math.PI*2);ovctx.strokeStyle='#ff334f';ovctx.lineWidth=3/z;ovctx.stroke();}};
const previousPointerCut55=$('overlay').onpointerdown;$('overlay').onpointerdown=function(event){
 if($('tool52').value!=='cut_two55')return previousPointerCut55(event);if(!active||event.button!==0)return;event.preventDefault();const p=xy(event),f=frame52();
 try{let root=selected52();if(!root){root=findSegment52(f,...p);if(!root)throw Error('先点击要断开的根');choose52(root);}if(cutPreview55)resetCut55();if(!cutSelection55){cutRoot55=root.root_id;const graph=graph55(f,cutRoot55),k=nearestGraph55(graph,p);cutSelection55=graph.nodes.get(k);$('cutInfo55').textContent='起点已选，再点击终点。';}else{cutPreview55=pathBetween55(f,cutRoot55,cutSelection55,p);$('applyCut55').disabled=false;$('cutInfo55').textContent='中间段预览 '+Math.round(cutPreview55.length)+' px。确认后自动注册编号，原根像素保留。';}render();}catch(e){status(e.message);}
};
$('overlay').addEventListener('dblclick',event=>{if($('tool52').value==='cut_two55'){event.preventDefault();event.stopImmediatePropagation();}},true);
$('startCut55').onclick=()=>{resetCut55();setEdit53('cut_two55');status('请在同一条根上依次点两端，蓝线预览后确认断开。');};$('resetCut55').onclick=resetCut55;
function commitCut55(){
 if(!cutPreview55||!cutRoot55){status('请先选两端');return false;}const f=frame52(),oldRoot=cutRoot55,plan=cutPreview55,role=$('cutRole55').value,requested=$('cutTarget55').value.trim()||((role==='existing'||role==='new')?$('motherLabel54').value.trim():enteredMother55?.root_id===oldRoot?enteredMother55.value.trim():'');let desired=null,target=null;
 try{
  if(role!=='pending'&&requested){desired=parseLabel54(requested);target=[...rawHierarchy54().values()].find(h=>h.label===desired);
   if(role==='existing'||role==='new'){
    if(!target)throw Error('侧根归属需要已存在的母根编号，请先标记母根。');
    const mother=target,used=[...rawHierarchy54().values()].filter(h=>h.parent_root_id===mother.root_id&&h.branch_kind===role).map(h=>Number(h.label.split('-').at(-1).replace(/^N/,'')));desired=mother.label+'-'+(role==='new'?'N':'')+(Math.max(0,...used)+1);target=null;
   }else if(!target&&desired.includes('-'))resolve54(desired.slice(0,desired.lastIndexOf('-')));
  }else if(role==='existing'||role==='new')throw Error('请先输入母根编号；已有侧根会自动用 -1，新生侧根自动用 -N1。');
 }
 catch(e){status(e.message);return false;}
 const before=serialize52(),undoLength=undo52.length,newRoot=newRoot52(),label=nextIndependent54();const plans=[{frame:f,plan}],skipped=[];
 if($('syncCut55').checked)for(const peer of structure52.periods){if(peer===f)continue;try{const q=pathBetween55(peer,oldRoot,plan.points[0],plan.points.at(-1));const samples=q.points.filter((_,i)=>i%4===0),close=samples.filter(p=>distanceToPoints55(plan.points,...p)<=100).length/Math.max(1,samples.length),ratio=q.length/plan.length;if(close<.7||ratio<.5||ratio>2)throw Error('走向或长度不一致');plans.push({frame:peer,plan:q});}catch(e){skipped.push(peer.period);}}
 try{
  snapshot52();let sid=null;for(const item of plans){const id=applyPartition55(item.frame,oldRoot,newRoot,item.plan);if(item.frame===f)sid=id;}
  structure52.root_track_hierarchy.push({root_id:newRoot,label,parent_root_id:null,family_root_id:newRoot,branch_kind:'independent',review_status:'modified',parentage_confirmed:false,first_seen_period:plans.map(p=>p.frame.period).sort()[0],observed_periods:plans.map(p=>p.frame.period),creation_reason:'user_two_end_disconnect_pending_assignment'});hierarchyOwner54=null;sync54();
  if(target)mergeIds53(newRoot,target.root_id);else if(desired)applyLabel54(newRoot,desired);
  selectedSegment52=sid;focusRoot53=selected52()?.root_id;const project=projectObject();validateStructure52(project.root_structure_annotation,project);resetCut55();changed52();
  drafts55.delete(oldRoot);drafts55.delete(newRoot);syncInputs55();status('已断开两端，中间段编号 '+label54(selected52().root_id)+'；已处理 '+plans.map(p=>p.frame.period).join('、')+(skipped.length?'。'+skipped.join('、')+' 未可靠匹配，保持原状。':'。')+' 可直接改号或设母根关系。');return true;
 }catch(e){structure52=before;hierarchyOwner54=null;rebuild52();undo52.splice(undoLength);refresh52();render();status('断开未完成，原结构已恢复：'+e.message);return false;}
}
$('applyCut55').onclick=commitCut55;
for(const name of ['rootQuickId53','rootId52'])$(name).addEventListener('input',event=>{if(cutPreview55&&$('tool52').value==='cut_two55'){$('cutTarget55').value=event.target.value;$('cutInfo55').textContent='输入将用于断开的中间段：'+event.target.value+'。母根本体或侧根身份按下面选择登记。';}});
for(const name of ['applyRoot53','assign52','rename52']){const previous=$(name).onclick;$(name).onclick=function(event){if(cutPreview55&&$('tool52').value==='cut_two55'){$('cutTarget55').value=value55();return commitCut55();}return previous?.call(this,event);};}
const previousCommitCut55=commit55;commit55=function(mode){if(cutPreview55&&$('tool52').value==='cut_two55'){if(mode==='swap'){status('请先确认断开中间段，再交换编号。');return false;}$('cutTarget55').value=value55();return commitCut55();}return previousCommitCut55(mode);};
const previousStateCut55=state53;state53=function(){previousStateCut55();if($('tool52').value==='cut_two55'){$('editHelp53').textContent=cutPreview55?'蓝线是待断开的中间段；输入编号只用于该中间段，再点确认或应用。':'请在同一条根上依次选择起点和终点，先预览再断开。';if(cutPreview55)$('applyRoot53').textContent='应用到中间段并断开';}};
// Manual foreground additions inherit the selected root's owner; an existing
// crossing pixel keeps its owner. Pixel undo restores both labels and ownership.
let ownerStroke55=null,strokePath55=[],manualStroke55=null;
const previousBackup55=backup;backup=function(){previousBackup55();ownerStroke55=new Map();strokePath55=[];manualStroke55=null;};
const previousDot55=dot;dot=function(x,y){
 const f=frame52(),cache=f?cache52(f):null,selected=selected52(),radius=Number($('brush').value)/2,bounds=[Math.max(0,Math.floor(x-radius)),Math.min(W-1,Math.ceil(x+radius)),Math.max(0,Math.floor(y-radius)),Math.min(H-1,Math.ceil(y+radius))];
 if(cache&&ownerStroke55)for(let yy=bounds[2];yy<=bounds[3];yy++)for(let xx=bounds[0];xx<=bounds[1];xx++){const i=yy*W+xx;if(!ownerStroke55.has(i))ownerStroke55.set(i,cache.owner[i]);}
 previousDot55(x,y);if(!cache)return;let added=false;
 for(let yy=bounds[2];yy<=bounds[3];yy++)for(let xx=bounds[0];xx<=bounds[1];xx++){const i=yy*W+xx;if(active.mask[i]!==255)cache.owner[i]=0;else if(cache.owner[i]===0&&selected){cache.owner[i]=selected.index;added=true;}}
 const point=[Math.max(0,Math.min(W-1,Math.round(x))),Math.max(0,Math.min(H-1,Math.round(y)))];if(!strokePath55.length||Math.hypot(point[0]-strokePath55.at(-1)[0],point[1]-strokePath55.at(-1)[1])>=1)strokePath55.push(point);
 if(added&&selected){selected.geometry_needs_review=true;if(!manualStroke55){manualStroke55={id:'STROKE-'+Date.now(),root_id:selected.root_id,segment_id:selected.id,points:strokePath55.slice(),review_status:'modified'};(f.manual_root_strokes??=[]).push(manualStroke55);}}
 if(manualStroke55)manualStroke55.points=strokePath55.slice();
};
const previousFinish55=finishBackup;finishBackup=function(){const before=active?.undo.at(-1);previousFinish55();const f=frame52();if(active&&f&&active.undo.at(-1)&&active.undo.at(-1)!==before){const u=active.undo.at(-1);u.ownerValues55=Uint32Array.from(u.idx,i=>ownerStroke55?.get(i)??cache52(f).owner[i]);u.manualStrokeId55=manualStroke55?.id;}ownerStroke55=null;manualStroke55=null;};
const previousPixelUndo55=undoStroke;undoStroke=function(){const u=active?.undo.at(-1),f=frame52();previousPixelUndo55();if(u?.ownerValues55&&f){const owner=cache52(f).owner;for(let j=0;j<u.idx.length;j++)owner[u.idx[j]]=u.ownerValues55[j];if(u.manualStrokeId55)f.manual_root_strokes=(f.manual_root_strokes||[]).filter(s=>s.id!==u.manualStrokeId55);render();}};$('undo').onclick=undoStroke;
const previousSelectCut55=select;select=async function(i){const name=active?.name;await previousSelectCut55(i);if(active?.name!==name)resetCut55();};
const previousImportCut55=importProject;importProject=async function(file,options={}){const result=await previousImportCut55(file,options);enteredMother55=null;resetCut55();return result;};
const previousUndoCut55=$('undo52').onclick;$('undo52').onclick=()=>{previousUndoCut55();resetCut55();};
window.rootCutReady55=(async()=>{await window.idEditReady55;return {version:55,two_end_cut:true};})();

'use strict';
// Cut topology at user-chosen native vertices, then detach the chosen pieces.
// Pixel layers are never edited. No path between clicks is required.
document.getElementById('freeCutPanel62')?.remove();
document.getElementById('freeCutStyle62')?.remove();
document.getElementById('piecePicker62')?.remove();
document.body.classList.remove('freeCutting62');
const freePanel62=document.createElement('section');freePanel62.id='freeCutPanel62';
freePanel62.innerHTML=`<h4>① 任意位置切开 → ② 分出根段</h4>
<button id="startFreeCut62">① 点选切口（可点一处或多处）</button>
<div id="freeCutInfo62">先点上面的按钮，再在想断开的位置点根线。两个切口不必连通，也不必属于同一条根。</div>
<label id="freeCutChoiceLabel62" hidden>最近切口所在根段（交叉处请核对）<select id="freeCutChoice62"></select></label>
<button id="commitFreeCut62" disabled>确认切开，暂不分号</button>
<button id="cutAndDetach62" disabled>确认切开 → ② 选择分出部分</button>
<button id="undoFreePoint62" disabled>撤回最后一个切口</button>
<button id="cancelFreeCut62" disabled>取消本次切开</button>
<button id="detachPieces62">② 分出切开的部分</button>
<small>红色剪口是预览；再次点击同一处可取消。可先切两端，再逐段点选粉色部分并确认分出；分出后才赋予不同身份。根段端点也可断开已有连接。仅修改当前期，不删除根系像素，不自动修改编号或其他时期。</small>`;
$('detachPanel61').before(freePanel62);
const styleFree62=document.createElement('style');styleFree62.id='freeCutStyle62';styleFree62.textContent='#freeCutPanel62{border:2px solid #cd433d;background:#fff8f4;padding:8px;margin:8px 0}#freeCutPanel62 h4{margin:0 0 8px}#freeCutInfo62{white-space:pre-wrap;margin:8px 0}#freeCutChoice62{width:100%}body.freeCutting62 #readableLabels57{display:none!important}';document.head.append(styleFree62);
let freeCut62=null;
const piecePicker62=document.createElement('div');piecePicker62.id='piecePicker62';piecePicker62.innerHTML='<label>切开后的单独根段（很短或交叉处难点选时）<select id="pieceSelect62"></select></label><button id="togglePiece62" disabled>加入／移出粉色选择</button>';$('detachInfo61').after(piecePicker62);
function updatePiecePicker62(){
 const select=$('pieceSelect62'),old=select.value,f=frame52(),parts=detach61&&f?.period===detach61.period?f.segments.filter(s=>s.root_id===detach61.root):[];
 const signature=JSON.stringify([detach61?.root,parts.map(s=>[s.id,detach61.parts.has(s.id)])]);if(select.dataset.signature===signature)return;select.dataset.signature=signature;select.replaceChildren();
 for(const s of parts.sort((a,b)=>Number(!!b.manual_cut_source_v62)-Number(!!a.manual_cut_source_v62)||a.index-b.index)){const option=document.createElement('option');option.value=s.id;option.textContent=(detach61.parts.has(s.id)?'✓ ':'')+s.id+' · '+Math.round(s.length_px)+' px · ('+s.points[0].join(',')+') → ('+s.points.at(-1).join(',')+')';select.append(option);}
 if(parts.some(s=>s.id===old))select.value=old;else if(parts.some(s=>s.id===selectedSegment52))select.value=selectedSegment52;$('togglePiece62').disabled=!parts.length;select.disabled=!parts.length;
}
$('togglePiece62').onclick=()=>{if(!detach61)return;const id=$('pieceSelect62').value,s=cache52(frame52()).byId.get(id);if(!s||s.root_id!==detach61.root)return;if(detach61.parts.has(id))detach61.parts.delete(id);else detach61.parts.add(id);updateDetach61();render();};
function resetFreeCut62(message=''){
 freeCut62=null;document.body.classList.remove('freeCutting62');
 for(const id of ['commitFreeCut62','cutAndDetach62','undoFreePoint62','cancelFreeCut62'])$(id).disabled=true;
 $('freeCutChoiceLabel62').hidden=true;if(message)$('freeCutInfo62').textContent=message;
}
function cutCandidate62(s,x,y){const q=distancePath52(s.points,x,y);return {segment_id:s.id,at:q.at,point:s.points[q.at].slice(),distance:q.d};}
function updateFreeCut62(){
 if(!freeCut62)return;const count=freeCut62.cuts.length;
 $('freeCutInfo62').textContent=freeCut62.period+' · 已预选 '+count+' 个切口。\n'+(count?'核对红色剪口；确认切开后，使用“② 分出”选中需要独立的部分。':'点击根线任意位置。可跨不同根点选；点击背景不会切根。');
 for(const id of ['commitFreeCut62','cutAndDetach62','undoFreePoint62'])$(id).disabled=!count;
 $('cancelFreeCut62').disabled=false;
 const last=freeCut62.cuts.at(-1);$('freeCutChoiceLabel62').hidden=!last;
 $('freeCutChoice62').replaceChildren();
 if(last)for(const c of last.candidates){const s=cache52(frame52()).byId.get(c.segment_id),option=document.createElement('option');option.value=s.id;option.textContent=(assigned60(s.root_id)?label54(s.root_id):'未定义候选')+' · '+s.id+' · '+(c.at===0||c.at===s.points.length-1?'端点断开':'根段内部');option.selected=s.id===last.segment_id;$('freeCutChoice62').append(option);}
}
function startFreeCut62(){
 if(!frame52()){notify60('请先载入带逐根结构的原图。');return;}
 resetDetach61();resetCut55();editMode60=editRoot60=null;$('rootEdit53').click();
 freeCut62={period:frame52().period,cuts:[]};document.body.classList.add('freeCutting62');updateFreeCut62();render();
 notify60('点击想断开的位置；可点一处或多处，红色预览后确认。交叉处可用下拉列表更换目标根段。');
}
function toggleFreeCut62(x,y){
 const f=frame52();if(!freeCut62||f?.period!==freeCut62.period)return;
 const candidates=f.segments.map(s=>cutCandidate62(s,x,y)).filter(q=>q.distance<=100).sort((a,b)=>a.distance-b.distance);
 if(!candidates.length){notify60('未命中根线，未添加切口。请放大后点在要切开的根上。');return;}
 const owner=ownerAt52(f,x,y),pick=candidates.find(q=>q.segment_id===owner?.id)||candidates[0];
 const previous=freeCut62.cuts.findIndex(c=>c.segment_id===pick.segment_id&&Math.hypot(c.point[0]-pick.point[0],c.point[1]-pick.point[1])<=3);
 if(previous>=0)freeCut62.cuts.splice(previous,1);else freeCut62.cuts.push({...pick,candidates});
 choose52(cache52(f).byId.get(pick.segment_id));updateFreeCut62();render();
}
function endpointsBlocked62(frame,segment,point){return (frame.manual_free_cuts_v62||[]).some(c=>(c.endpoints||[]).some(e=>e.segment_id===segment.id&&Math.hypot(e.point[0]-point[0],e.point[1]-point[1])<.01));}
function commitFreeCut62(thenDetach=false){
 if(!freeCut62?.cuts.length){notify60('请先点选切口。');return false;}
 const f=frame52(),cuts=structuredClone(freeCut62.cuts);if(f?.period!==freeCut62.period)return false;
 const before=serialize52(),previousUndo=undo52.slice(),selectedBefore=selectedSegment52;
 try{
  snapshot52();const cache=cache52(f),groups=new Map(),replacements=new Map();let next=Math.max(0,...f.segments.map(s=>s.index))+1,changed=0;
  const cutId='FREE62-'+Date.now()+'-'+next,record={id:cutId,kind:'user_arbitrary_topology_cut_v62',period_scope:'current_period_only',points:[],blocked_edges:[],endpoints:[],removed_connections:[],root_pixels_removed:0,formal_numbers_changed:false};
  for(const c of cuts){if(!cache.byId.has(c.segment_id))throw Error('根段已经变化，请重新选择切口。');if(!groups.has(c.segment_id))groups.set(c.segment_id,[]);groups.get(c.segment_id).push(c);}
  const nearest=(pieces,x,y)=>pieces.reduce((a,b)=>distancePath52(b.points,x,y).d<distancePath52(a.points,x,y).d?b:a,pieces[0]);
  for(const [id,cc] of groups){
   const s=cache.byId.get(id),positions=[...new Set(cc.map(c=>c.at))].sort((a,b)=>a-b);
   const interior=positions.filter(at=>at>0&&at<s.points.length-1);
   const limits=[0,...interior,s.points.length],pieces=[];
   for(let i=0;i<limits.length-1;i++){
    const piece=structuredClone(s);if(i){piece.index=next++;piece.id=f.period+'-S'+String(piece.index).padStart(4,'0');piece.previous_candidates=[];}
    piece.points=s.points.slice(limits[i],limits[i+1]);if(!piece.points.length)throw Error('切口重合，未修改。');
    piece.length_px=piece.points.slice(1).reduce((n,p,j)=>n+Math.hypot(p[0]-piece.points[j][0],p[1]-piece.points[j][1]),0);
    piece.endpoint_kinds=[i?'manual_cut':(positions.includes(0)?'manual_cut':s.endpoint_kinds?.[0]||'unknown'),i<limits.length-2?'manual_cut':(positions.includes(s.points.length-1)?'manual_cut':s.endpoint_kinds?.[1]||'unknown')];
    piece.manual_partition_from=s.manual_partition_from||s.id;piece.manual_cut_source_v62=s.id;piece.review_status='modified';
    // Existing cuts on the old segment follow the piece which keeps that endpoint.
    for(const old of f.manual_free_cuts_v62||[])for(const e of old.endpoints||[])if(e.segment_id===s.id&&piece.points.some(p=>p[0]===e.point[0]&&p[1]===e.point[1]))e.segment_id=piece.id;
    pieces.push(piece);
   }
   replacements.set(s.index,pieces);
   for(let i=0;i<pieces.length-1;i++)record.endpoints.push({segment_id:pieces[i].id,point:pieces[i].points.at(-1).slice()},{segment_id:pieces[i+1].id,point:pieces[i+1].points[0].slice()});
   for(const at of positions.filter(at=>at===0||at===s.points.length-1)){const p=s.points[at],piece=at===0?pieces[0]:pieces.at(-1);record.endpoints.push({segment_id:piece.id,point:p.slice()});}
   for(const n of f.nodes){
    if(n.owner_segment_id===s.id)n.owner_segment_id=nearest(pieces,n.x,n.y).id;
    const cutArms=new Set();for(const a of n.arms)if(a.segment_id===s.id){if(cc.some(c=>Math.hypot(c.point[0]-a.x,c.point[1]-a.y)<10))cutArms.add(a.id);a.segment_id=nearest(pieces,a.x,a.y).id;}
    n.connections=n.connections.filter(pair=>{if(pair.some(a=>cutArms.has(a))){record.removed_connections.push({node_id:n.id,pair});return false;}return true;});
   }
   changed+=interior.length+positions.filter(at=>at===0||at===s.points.length-1).length;
  }
  // Reassign only the segment index of each original foreground pixel.
  // No new foreground, width change, erased pixel, or double ownership.
  for(let i=0;i<cache.owner.length;i++){const pieces=replacements.get(cache.owner[i]);if(pieces)cache.owner[i]=nearest(pieces,i%f.width,Math.floor(i/f.width)).index;}
  f.segments=f.segments.flatMap(s=>replacements.get(s.index)||[s]);cache.byId=new Map(f.segments.map(s=>[s.id,s]));cache.byIndex=new Map(f.segments.map(s=>[s.index,s]));
  (f.manual_free_cuts_v62??=[]).push(record);
  // Never claim the newly created pieces were observed in another period.
  for(const link of structure52.cross_period_links||[])if((link.previous_period===f.period&&groups.has(link.previous_segment))||(link.current_period===f.period&&groups.has(link.current_segment))){link.review_status='pending';link.kind='identity_link_needs_review';link.note_v62='用户在当前期切开根段，跨期对应需要复核';}
  const first=cuts[0],pieces=replacements.get(before.periods.find(p=>p.period===f.period).segments.find(s=>s.id===first.segment_id).index);const focus=pieces?.[Math.min(1,pieces.length-1)]||cache.byId.get(first.segment_id);
  resetFreeCut62('已确认 '+changed+' 个切口。根编号暂时不变；下一步点“② 分出切开的部分”，选择需要独立的段。');
  if(focus){choose52(focus);focusRoot53=focus.root_id;}changed52();const project=projectObject();validateStructure52(project.root_structure_annotation,project);
  notify60('切开完成；根像素和原编号保留。现在可逐段点选，分出不属于同一条根的部分。');
  if(thenDetach)beginPieceDetach62();return true;
 }catch(error){structure52=before;hierarchyOwner54=null;rebuild52();undo52=previousUndo;selectedSegment52=selectedBefore;resetFreeCut62('切开失败，已恢复原标注：'+error.message);refresh52();render();notify60($('freeCutInfo62').textContent);return false;}
}
function beginPieceDetach62(){resetFreeCut62();$('detachScope61').value='segment';startDetach61();$('detachInfo61').scrollIntoView({block:'nearest'});}
$('startFreeCut62').onclick=startFreeCut62;$('commitFreeCut62').onclick=()=>commitFreeCut62(false);$('cutAndDetach62').onclick=()=>commitFreeCut62(true);$('detachPieces62').onclick=beginPieceDetach62;
$('cancelFreeCut62').onclick=()=>{resetFreeCut62('已取消切口预览，未修改标注。');render();};
$('undoFreePoint62').onclick=()=>{freeCut62?.cuts.pop();updateFreeCut62();render();};
$('freeCutChoice62').onchange=()=>{const last=freeCut62?.cuts.at(-1),c=last?.candidates.find(c=>c.segment_id===$('freeCutChoice62').value);if(c){Object.assign(last,c);choose52(cache52(frame52()).byId.get(c.segment_id));updateFreeCut62();render();}};
$('b3ImageViewport').addEventListener('pointerdown',event=>{if(!freeCut62||event.button!==0||!$('overlay').contains(event.target))return;event.preventDefault();event.stopImmediatePropagation();try{toggleFreeCut62(...xy(event));}catch(error){notify60('未添加切口：'+error.message);}},true);
const renderBeforeFree62=render;render=function(){renderBeforeFree62();updatePiecePicker62();const f=frame52();if(!f)return;const z=Math.max(.1,Number($('zoom').value)||1);ovctx.strokeStyle='#ec283b';ovctx.lineWidth=2/z;
 const markers=[...(f.manual_free_cuts_v62||[]).flatMap(c=>c.endpoints||[]),...(freeCut62?.period===f.period?freeCut62.cuts:[])];
 for(const e of markers){const p=e.point;ovctx.beginPath();ovctx.moveTo(p[0]-5/z,p[1]-5/z);ovctx.lineTo(p[0]+5/z,p[1]+5/z);ovctx.moveTo(p[0]-5/z,p[1]+5/z);ovctx.lineTo(p[0]+5/z,p[1]-5/z);ovctx.stroke();}
};
// Respect explicit user cuts when proposing route continuation.
routedSegments61=function(frame,rid,start){
 const ss=frame.segments.filter(s=>s.root_id===rid),ends=[],choices=new Map(),neighbors=new Map(ss.map(s=>[s.id,new Set()]));
 for(const s of ss){if(s.points.length<4)continue;for(const end of [0,1]){const p=s.points[end?s.points.length-1:0];if(endpointsBlocked62(frame,s,p))continue;const q=s.points[end?Math.max(0,s.points.length-10):Math.min(9,s.points.length-1)];ends.push({sid:s.id,p,d:unit61([p[0]-q[0],p[1]-q[1]])});}}
 const candidates=[];
 for(let i=0;i<ends.length;i++)for(let j=0;j<i;j++){const a=ends[i],b=ends[j];if(a.sid===b.sid)continue;const d=Math.hypot(a.p[0]-b.p[0],a.p[1]-b.p[1]),align=-(a.d[0]*b.d[0]+a.d[1]*b.d[1]);if(d>16||align<.91)continue;const score=align-d*.006,e={i,j,score};candidates.push(e);for(const k of [i,j])if(!choices.has(k)||score>choices.get(k).score)choices.set(k,e);}
 for(const e of candidates)if(choices.get(e.i)===e&&choices.get(e.j)===e){const a=ends[e.i].sid,b=ends[e.j].sid;neighbors.get(a).add(b);neighbors.get(b).add(a);}
 const found=new Set([start.id]),todo=[start.id];while(todo.length)for(const q of neighbors.get(todo.pop())||[])if(!found.has(q)){found.add(q);todo.push(q);}return found;
};
// Remove inferred bridges at cut endpoints but retain each piece's native edges.
const graphBeforeFree62=graph55;graph55=function(frame,rid){const graph=graphBeforeFree62(frame,rid),native=new Set();for(const s of frame.segments.filter(s=>s.root_id===rid))for(let i=1;i<s.points.length;i++)native.add(edgeKey55(key55(s.points[i-1]),key55(s.points[i])));
 const by=cache52(frame).byId;for(const c of frame.manual_free_cuts_v62||[])for(const e of c.endpoints||[]){if(by.get(e.segment_id)?.root_id!==rid)continue;const key=key55(e.point);for(const other of [...(graph.edges.get(key)?.keys()||[])])if(!native.has(edgeKey55(key,other))){graph.edges.get(key).delete(other);graph.edges.get(other)?.delete(key);}}return graph;
};
const selectBeforeFree62=select;select=async function(i){resetFreeCut62('时期切换，未确认的切口已取消。');return selectBeforeFree62(i);};
const importBeforeFree62=importProject;importProject=async function(file,options={}){resetFreeCut62();return importBeforeFree62(file,options);};
for(const id of ['startDetach61','dropPath60','startReviewCut60','selectMode60','panView60','joinPath60','defineApply60','clearDefinition60','undoPath60','undo52']){const previous=$(id).onclick;$(id).onclick=function(event){resetFreeCut62();return previous?.call(this,event);};}
window.freeCutReady62=(async()=>{await window.detachReady61;document.title='任意切开并分出根段 · V62';$('detachPanel61').querySelector('small').textContent='先点①任意位置切开，再点②逐段分出；不要求两个切口连通。分出的部分未定义，不删除像素、不自动判新生；本次只修改当前期。';notify60('已保留上传15的人工定义。新增“①任意位置切开 → ②分出根段”；切两端后可单独选中中间段。');return {version:62};})();

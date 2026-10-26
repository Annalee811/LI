'use strict';
document.getElementById('detachPanel61')?.remove();document.getElementById('detachStyle61')?.remove();
document.body.classList.remove('detaching61');
const detachPanel61=document.createElement('section');detachPanel61.id='detachPanel61';detachPanel61.innerHTML=`<h4>分离误并部分（不要求两点连通）</h4><button id="startDetach61">点选要分出的部分</button><label>选择范围<select id="detachScope61"><option value="route">沿所点走向预选，可多次增减</option><option value="component">所点连接块（适合相互分开的两条根）</option><option value="segment">只选所点的这一段，可多选</option></select></label><div id="detachInfo61">先在原图选择要修改的根，再点上面的按钮。</div><button id="commitDetach61" disabled>确认分出粉色部分</button><button id="clearDetach61" disabled>清空选择</button><button id="cancelDetach61" disabled>取消分离</button><small>粉色只是分离预览。再次点选可取消；确认后分出的部分保持未定义，不删除像素、不自动判新生。本次只修改当前期，其他时期保留。若需截取一段内部，请使用下面“两点分开”。</small>`;
$('periods60').before(detachPanel61);const css61=document.createElement('style');css61.id='detachStyle61';css61.textContent='#detachPanel61{border:2px solid #bc6790;padding:7px;background:#fff8fb;margin:8px 0}#detachPanel61 h4{margin:0 0 7px}#detachScope61{width:100%}#detachInfo61{white-space:pre-wrap;margin:7px 0}body.detaching61 #readableLabels57{display:none!important}';document.head.append(css61);
let detach61=null;
function resetDetach61(message=''){detach61=null;document.body.classList.remove('detaching61');$('commitDetach61').disabled=true;$('clearDetach61').disabled=true;$('cancelDetach61').disabled=true;if(message)$('detachInfo61').textContent=message;}
function unit61(v){const n=Math.hypot(...v)||1;return v.map(x=>x/n);}
function routedSegments61(frame,rid,start){
 const ss=frame.segments.filter(s=>s.root_id===rid),ends=[],choices=new Map(),neighbors=new Map(ss.map(s=>[s.id,new Set()]));
 for(const s of ss){if(s.points.length<4)continue;for(const end of [0,1]){const p=s.points[end?s.points.length-1:0],q=s.points[end?Math.max(0,s.points.length-10):Math.min(9,s.points.length-1)];ends.push({sid:s.id,end,p,d:unit61([p[0]-q[0],p[1]-q[1]])});}}
 const candidates=[];
 for(let i=0;i<ends.length;i++)for(let j=0;j<i;j++){const a=ends[i],b=ends[j];if(a.sid===b.sid)continue;const d=Math.hypot(a.p[0]-b.p[0],a.p[1]-b.p[1]),align=-(a.d[0]*b.d[0]+a.d[1]*b.d[1]);if(d>16||align<.91)continue;const score=align-d*.006;candidates.push({i,j,score});for(const k of [i,j])if(!choices.has(k)||score>choices.get(k).score)choices.set(k,{i,j,score});}
 for(const e of candidates)if(choices.get(e.i)?.i===e.i&&choices.get(e.i)?.j===e.j&&choices.get(e.j)?.i===e.i&&choices.get(e.j)?.j===e.j){const a=ends[e.i].sid,b=ends[e.j].sid;neighbors.get(a).add(b);neighbors.get(b).add(a);}
 const found=new Set([start.id]),todo=[start.id];while(todo.length){const id=todo.pop();for(const q of neighbors.get(id)||[])if(!found.has(q)){found.add(q);todo.push(q);}}return found;
}
function componentSegments61(frame,rid,point){
 const graph=graph55(frame,rid),start=nearestGraph55(graph,point),keys=new Set([start]),todo=[start];while(todo.length){const key=todo.pop();for(const other of graph.edges.get(key).keys())if(!keys.has(other)){keys.add(other);todo.push(other);}}
 return new Set(frame.segments.filter(s=>s.root_id===rid&&s.points.some(p=>keys.has(key55(p)))).map(s=>s.id));
}
function updateDetach61(){
 if(!detach61)return;const f=frame52();if(f?.period!==detach61.period){resetDetach61('时期已切换，未确认的分离选择已取消。');return;}
 const ss=f.segments.filter(s=>detach61.parts.has(s.id)&&s.root_id===detach61.root),all=structure52.periods.flatMap(p=>p.segments.filter(s=>s.root_id===detach61.root));const remaining=all.length-ss.length;const label=assigned60(detach61.root)?label54(detach61.root):'未定义候选';
 $('detachInfo61').textContent=f.period+' · 原根 '+label+'\n粉色预选 '+ss.length+' 段，现有根线约 '+Math.round(ss.reduce((n,s)=>n+s.length_px,0))+' px。'+(remaining<=0?'\n选中了所有期的全部部分，无法再拆成两份。请缩小选择；若只想取消编号，用撤回正式编号。':'\n确认后只把粉色部分分到未定义候选，其余沿用原号。');
 $('commitDetach61').disabled=!ss.length||remaining<=0;$('clearDetach61').disabled=!ss.length;$('cancelDetach61').disabled=false;
}
function startDetach61(){
 const r=selectedRoot60(),f=frame52();if(!r||!f||!f.segments.some(s=>s.root_id===r)){notify60('先点击当前期需要分离的根。');return;}
 resetCut55();editMode60=null;editRoot60=null;$('rootEdit53').click();detach61={root:r,period:f.period,parts:new Set()};document.body.classList.add('detaching61');updateDetach61();notify60('正在点选分离：点击误并的部分，粉色预览后再确认；不需要找到连续路径。');render();
}
function toggleDetachAt61(x,y){
 if(!detach61)return;const f=frame52();if(f?.period!==detach61.period){resetDetach61('时期改变，已取消本次分离。');return;}
 let s=ownerAt52(f,x,y);if(s&&s.root_id!==detach61.root){notify60('这里属于另一条根。请点当前所选根的部分，或先取消、重新选根。');return;}
 if(!s){let best=169;for(const q of f.segments)if(q.root_id===detach61.root){const d=distancePath52(q.points,x,y).d;if(d<best){best=d;s=q;}}}
 if(!s||s.root_id!==detach61.root){notify60('这里未命中当前根，选择未改变。请放大后点根线。');return;}
 let ids;const scope=$('detachScope61').value;if(scope==='component')ids=componentSegments61(f,detach61.root,[x,y]);else if(scope==='route')ids=routedSegments61(f,detach61.root,s);else ids=new Set([s.id]);
 if([...ids].every(id=>detach61.parts.has(id)))for(const id of ids)detach61.parts.delete(id);else for(const id of ids)detach61.parts.add(id);updateDetach61();render();
}
function commitDetach61(){
 if(!detach61){notify60('请先点选需要分出的部分。');return false;}const f=frame52(),root=detach61.root,parts=f.segments.filter(s=>s.root_id===root&&detach61.parts.has(s.id));const remaining=structure52.periods.reduce((n,q)=>n+q.segments.filter(s=>s.root_id===root).length,0)-parts.length;
 if(!parts.length||remaining<=0){notify60(!parts.length?'还没有选择粉色部分。':'没有剩余部分可保留，未修改。请缩小选择。');return false;}
 const before=serialize52(),undoLength=undo52.length;
 try{
  snapshot52();const newr=newRoot52(),label=internalLabel60(),map=rawHierarchy54();map.set(newr,{root_id:newr,label,parent_root_id:null,family_root_id:newr,branch_kind:'independent',review_status:'pending',annotation_status_v60:'unassigned',candidate_path_only:true,hidden_technical_label:true,birth_role_v58:'unresolved_side_vs_independent',first_seen_period:f.period,observed_periods:[f.period],creation_reason_v61:'user_detached_existing_parts_without_continuous_path'});structure52.root_track_hierarchy=[...map.values()];hierarchyOwner54=null;
  const selectedIds=parts.map(s=>s.id);for(const s of parts){s.root_id=newr;s.review_status='modified';s.parent_root_id=null;s.family_root_id=newr;s.branch_kind='independent';s.display_label=label;}
  const by=new Map(f.segments.map(s=>[s.id,s])),removed=[];for(const n of f.nodes)n.connections=n.connections.filter(pair=>{const roots=pair.map(id=>by.get(n.arms.find(a=>a.id===id)?.segment_id)?.root_id);if(roots.includes(root)&&roots.includes(newr)){removed.push({node_id:n.id,pair});return false;}return true;});
  const points=parts.flatMap(s=>[s.points[0],s.points.at(-1)]);(f.manual_disconnects??=[]).push({id:'DETACH61-'+Date.now(),kind:'manual_identity_detach_v61',old_root_id:root,new_root_id:newr,segment_ids:selectedIds,points:[],selected_endpoints:points,blocked_edges:[],removed_connections:removed,root_pixels_removed:0,period_scope:'current_period_only',review_status:'modified'});
  (structure52.manual_detachments_v61??=[]).push({period:f.period,original_root_id:root,new_root_id:newr,segment_ids:selectedIds,method:'user_selected_existing_parts',no_continuous_path_required:true,other_periods_unchanged:true});
  sync54();resetDetach61('分离已完成，新部分保持未定义。你可以给它指定已有编号、N-编号或母根侧根编号。');focusRoot53=newr;choose52(parts[0]);changed52();const project=projectObject();validateStructure52(project.root_structure_annotation,project);notify60('已分出当前期粉色部分；原号及其他时期保留，新部分未定义。可继续定义或撤销。');previews60();return true;
 }catch(error){structure52=before;hierarchyOwner54=null;rebuild52();undo52.splice(undoLength);resetDetach61();refresh52();render();notify60('分离未完成，已恢复原标注：'+error.message);return false;}
}
$('startDetach61').onclick=startDetach61;$('dropPath60').textContent='分离误并部分（无需连续路径）';$('dropPath60').onclick=startDetach61;$('commitDetach61').onclick=commitDetach61;$('clearDetach61').onclick=()=>{if(detach61)detach61.parts.clear();updateDetach61();render();};$('cancelDetach61').onclick=()=>{resetDetach61('已取消分离，标注未改变。');render();};$('detachScope61').onchange=()=>{if(detach61)detach61.parts.clear();updateDetach61();render();};
$('b3ImageViewport').addEventListener('pointerdown',event=>{if(!detach61||event.button!==0||!$('overlay').contains(event.target))return;event.preventDefault();event.stopImmediatePropagation();try{toggleDetachAt61(...xy(event));}catch(error){notify60('选择未改变：'+error.message);}},true);
const renderBefore61=render;render=function(){renderBefore61();if(!detach61||frame52()?.period!==detach61.period)return;const f=frame52(),scale=Math.max(.1,Number($('zoom').value)||1);ovctx.strokeStyle='#ff4b9b';ovctx.lineWidth=3/scale;for(const s of f.segments)if(detach61.parts.has(s.id)){ovctx.beginPath();s.points.forEach((p,i)=>i?ovctx.lineTo(...p):ovctx.moveTo(...p));ovctx.stroke();}updateDetach61();};
const statusBefore61=status;status=function(message){let text=String(message);if(text.includes('两端之间没有已确认的连续路径'))text+=' 可改用“点选分离误并部分”，不要求路径连通。';statusBefore61(text);if($('tool52')?.value==='cut_two55'){$('cutInfo55').textContent=text;$('reviewStatus60').textContent=text;}return text;};
const selectBefore61=select;select=async function(i){const period=frame52()?.period;if(detach61)resetDetach61('时期已切换，未确认的选择已取消。');const result=await selectBefore61(i);if(frame52()?.period!==period)resetCut55();return result;};
const importBefore61=importProject;importProject=async function(file,options={}){resetDetach61();return importBefore61(file,options);};
const undoBefore61=$('undoPath60').onclick;$('undoPath60').onclick=()=>{resetDetach61();return undoBefore61();};
const startCutBefore61=$('startReviewCut60').onclick;$('startReviewCut60').onclick=()=>{resetDetach61();return startCutBefore61();};
for(const id of ['joinPath60','selectMode60','panView60','defineApply60','clearDefinition60']){const previous=$(id).onclick;$(id).onclick=function(event){resetDetach61();return previous?.call(this,event);};}
window.detachReady61=(async()=>{await window.confidenceReady60;document.title='上传15 · 误并根分离与四期编号 V61';notify60('已载入你的上传15，原有定义完整保留。误并或不连通部分请用“点选分离”；内部截取仍可选两点。分出的部分保持未定义，不自动标新生。');return {version:61};})();

'use strict';
// V53: make edit state visible and release pan capture when selecting an edit tool.
const toolbar53=document.createElement('div');toolbar53.id='editingToolbar53';
let mergeTarget53=null,focusRoot53=null,selectionAction53='root';
toolbar53.innerHTML=`<div class="actions53"><button id="rootEdit53">改整根编号</button><button id="mergeRoot53">合并另一段为同一根</button><button id="cancelMerge53" hidden>取消合并选择</button><button id="pixelEdit53">改时序颜色／画笔</button><button id="moveEdit53">移动点位</button><button id="connectionEdit53">修改交叉连接</button><button id="exitPan53">退出平移，恢复编辑</button><button id="saveQuick53">保存进度 JSON</button></div>
<div class="quickId53"><span id="selection53">先在图上点击一条根或一个点位</span><input id="rootQuickId53" type="text" placeholder="根编号，例如 0001"><button id="applyRoot53">应用编号（四期同步）</button></div>
<div id="trackPeriods53"></div><label class="inline53"><input id="wholeRoots53" type="checkbox" checked>列表按整根查看；取消可选择单个根段</label>
<div id="paintClasses53"><span>时序画笔：</span><button data-code53="3">持续段</button><button data-code53="2">根尖延伸段</button><button data-code53="1">新生根／侧根</button><button data-code53="4">T1 基础根</button></div>
<div id="editHelp53"></div><div id="visibleStatus53" role="status" aria-live="polite">正在载入原图和标注……</div>`;
$('name').after(toolbar53);
const originalStatus53=status,originalImportStatus53=importStatus;
status=function(message){originalStatus53(message);if($('visibleStatus53'))$('visibleStatus53').textContent=message;};
importStatus=function(message){originalImportStatus53(message);if($('visibleStatus53'))$('visibleStatus53').textContent=message;};
function stopPan53(){if(window.B3Zoom60Installed?.pan)$('b3Pan').click();$('pan52').classList.remove('active');}
function registry53(obj=structure52){
 const tracks=new Map();for(const f of obj?.periods||[])for(const s of f.segments){if(!tracks.has(s.root_id))tracks.set(s.root_id,{root_id:s.root_id,first_seen_period:f.period,observed_periods:[],identity_status:s.identity_assignment?.reason||'manual_identity_under_review',review_status:'pending',biological_identity_confirmed:false});const v=tracks.get(s.root_id);if(!v.observed_periods.includes(f.period))v.observed_periods.push(f.period);}
 return [...tracks.values()].sort((a,b)=>a.root_id.localeCompare(b.root_id));
}
function mergeIds53(from,to){
 if(from===to)return;for(const f of structure52.periods)for(const s of f.segments){if(s.root_id===from){s.root_id=to;s.review_status='modified'}if(s.parent_root_id===from)s.parent_root_id=to;if(s.parent_root_id===s.root_id)s.parent_root_id=null;for(const c of s.previous_candidates||[])if(c.previous_root_id===from)c.previous_root_id=to;}
 for(const l of structure52.cross_period_links)if(l.root_id===from){l.root_id=to;l.review_status='pending'}
 structure52.root_tracks=registry53();focusRoot53=to;
}
function state53(){
 const tool=$('tool52').value,pan=!!window.B3Zoom60Installed?.pan,s=selected52(),n=node52();
 const help={select:selectionAction53==='node'?'修改连接：点击 X/B/? 点，再在左侧选择两臂连接。':'改根编号：点击根本体，在上方输入编号并应用；四期同步。交叉处请点击离开交点的根段。',
  move:'移动点位：按住图上的 T、X、B 或 ? 点并拖动。整根的编号请用“改根编号”。',
  tip:'添加根尖：在原图对应根尖处单击。',branch:'添加分叉：在实际侧根长出的位置单击。',
  crossing:'添加交叉：在两根穿越的位置单击。',split:'拆分根段：在误连根段的中部单击，再分别核对编号。',
  pixel:'时序画笔：先点“持续段／根尖延伸段／新生根”，再按住鼠标沿根描画；左侧可调笔刷并撤销。'};
 $('editHelp53').textContent=pan?'当前为平移模式：拖动只移动视野。点击“退出平移，恢复编辑”后再修改。':mergeTarget53?'合并模式：现在点击属于同一条根的另一段，四期所有同号轨迹将合并到 '+mergeTarget53.replace(/^R/,'')+'。新侧根应保持独立编号。':help[tool]||help.select;
 $('exitPan53').hidden=!pan;
 $('paintClasses53').hidden=tool!=='pixel';
 $('selection53').textContent=s?`已选整根 ${s.root_id.replace(/^R/,'')} · 当前段 ${s.id}${n?' · '+n.id:''}`:focusRoot53?`整根 ${focusRoot53.replace(/^R/,'')}：本期未匹配到轨迹`:'先在图上点击一条根或一个点位';
 if(document.activeElement!==$('rootQuickId53'))$('rootQuickId53').value=s?.root_id.replace(/^R/,'')||'';
 $('applyRoot53').disabled=!s;
 $('mergeRoot53').disabled=!s;$('cancelMerge53').hidden=!mergeTarget53;
 const rid=s?.root_id||focusRoot53,track=registry53().find(t=>t.root_id===rid);
 $('trackPeriods53').textContent=track?'同一根 '+rid.replace(/^R/,'')+'：'+['T1','T2','T3','T4'].map(t=>t+' '+(track.observed_periods.includes(t)?'已匹配':'未观察到／待匹配')).join(' ｜ '):'根编号与时序类别分开：新生根下一期变为持续或延伸时，仍沿用原编号。';
 if(track?.identity_status==='unresolved_previous_identity')$('trackPeriods53').textContent+='。此为待匹配的临时编号，不表示已认定为新生根。';
 if(rid&&structure52?.periods.some(f=>f.segments.some(s=>s.root_id===rid&&s.identity_assignment?.requires_temporal_class_review)))$('trackPeriods53').textContent+='。本根的时序颜色与跨期几何对应存在冲突，需对照 RGB 复核；未自动改色。';
 for(const [id,value] of [['rootEdit53','select'],['pixelEdit53','pixel'],['moveEdit53','move']])$(id).classList.toggle('active',tool===value&&!pan);
 for(const b of $('paintClasses53').querySelectorAll('button'))b.classList.toggle('active',Number(b.dataset.code53)===Number($('temporal').value));
}
function setEdit53(tool){
 stopPan53();$('tool52').value=tool;$('tool52').dispatchEvent(new Event('change'));
 if(tool==='pixel'){$('displayLayer').value='temporal';$('view52').value='temporal';setMode(255,'fg');render();}
 state53();
}
const oldChange53=$('tool52').onchange;
$('tool52').onchange=function(){stopPan53();selectionAction53='auto';oldChange53();state53();};
const oldPan53=$('pan52').onclick;
$('pan52').onclick=function(){oldPan53();state53();};
$('exitPan53').onclick=()=>{stopPan53();state53();status('已退出平移，可以编辑。')};
$('rootEdit53').onclick=()=>{setEdit53('select');selectionAction53='root';state53()};$('pixelEdit53').onclick=()=>setEdit53('pixel');$('moveEdit53').onclick=()=>setEdit53('move');
$('connectionEdit53').onclick=()=>{setEdit53('select');selectionAction53='node';state53();$('nodeDetails52').open=true;$('nodeDetails52').scrollIntoView({block:'nearest'});status('点击图上的交叉或分叉点，然后选择左侧两条连接臂。')};
$('applyRoot53').onclick=()=>{try{const s=selected52();if(!s)throw Error('请先点击要改号的根');const id=readRoot52($('rootQuickId53').value);snapshot52();mergeIds53(s.root_id,id);changed52();state53();status('已将同一根的编号同步到四期：'+id.replace(/^R/,'')+'。输入已有编号时，两组候选根会合并。')}catch(e){status(e.message)}};
$('mergeRoot53').onclick=()=>{const s=selected52();if(!s)return;setEdit53('select');selectionAction53='root';mergeTarget53=s.root_id;state53();status('已选第一段；请点击同一条根的另一段。不是新侧根时才合并。')};
$('cancelMerge53').onclick=()=>{mergeTarget53=null;state53();status('已取消合并选择')};
const previousChoose53=choose52;
choose52=function(s,n=null,center=false){
 if(s&&mergeTarget53){const target=mergeTarget53;mergeTarget53=null;if(s.root_id!==target){snapshot52();mergeIds53(s.root_id,target);changed52();status('已将两组轨迹合并为同一根 '+target.replace(/^R/,'')+'，四期同步。')}else status('这两段已经属于同一编号');}
 if(s)focusRoot53=s.root_id;previousChoose53(s,n,center);state53();
};
const previousConnect53=$('connect52').onclick;
$('connect52').onclick=()=>{const n=node52(),f=frame52(),a=n?.arms.find(a=>a.id===selectedArms52[0]),b=n?.arms.find(a=>a.id===selectedArms52[1]),ra=a?root52(f,a.segment_id):null,rb=b?root52(f,b.segment_id):null;previousConnect53();if(ra&&rb&&ra!==rb&&node52()?.connections.some(p=>p.includes(a.id)&&p.includes(b.id))){mergeIds53(rb,ra);changed52(node52());status('两臂已连接为同一根，四期编号同步。原时序像素类别未改变。')}};
const previousPointerDown53=$('overlay').onpointerdown;
$('overlay').onpointerdown=function(event){
 if(active&&frame52()&&event.button===0&&$('tool52').value==='select'&&(selectionAction53==='root'||mergeTarget53)){
  event.preventDefault();$('overlay').focus({preventScroll:true});const [x,y]=xy(event),s=findSegment52(frame52(),x,y);choose52(s);compare52(x,y);return;
 }
 return previousPointerDown53(event);
};
for(const b of $('paintClasses53').querySelectorAll('button'))b.onclick=()=>{setEdit53('pixel');$('temporal').value=b.dataset.code53;$('displayLayer').value='temporal';setMode(255,'fg');updateBrushPreview();state53();status('已选择“'+b.textContent+'”画笔，请在根上按住鼠标描画。');};
$('saveQuick53').onclick=()=>saveProject();
const previousRefresh53=refresh52;
function wholeRootList53(){
 const f=frame52();if(!f||!$('wholeRoots53').checked)return;const query=$('search52').value.trim().toLowerCase(),groups=new Map();
 for(const s of f.segments){if(!groups.has(s.root_id))groups.set(s.root_id,[]);groups.get(s.root_id).push(s);}
 const list=[];for(const [rid,group] of groups){if(query&&!rid.toLowerCase().includes(query)&&!group.some(s=>s.id.toLowerCase().includes(query)))continue;const representative=group.find(s=>s.id===selectedSegment52)||[...group].sort((a,b)=>b.length_px-a.length_px)[0];list.push({rid,group,representative});}
 list.sort((a,b)=>a.rid.localeCompare(b.rid));const selected=selected52()?.root_id;
 $('segments52').innerHTML=list.map(v=>`<option value="${v.representative.id}" ${v.rid===selected?'selected':''}>${v.rid.replace(/^R/,'')} · ${v.group.length} 段 · ${Math.round(v.group.reduce((n,s)=>n+s.length_px,0))}px</option>`).join('');if(!selected)$('segments52').selectedIndex=-1;
}
refresh52=function(){previousRefresh53();wholeRootList53();state53();};
$('wholeRoots53').onchange=refresh52;
const previousSelect53=select;
select=async function(i){const root=selected52()?.root_id||focusRoot53;await previousSelect53(i);focusRoot53=root;const same=frame52()?.segments.filter(s=>s.root_id===root).sort((a,b)=>b.length_px-a.length_px)[0];if(same)selectedSegment52=same.id;refresh52();render();};
const previousSerialize53=serialize52;
serialize52=function(){const obj=previousSerialize53();if(!obj)return obj;obj.root_tracks=registry53(obj);const links=[];for(const track of obj.root_tracks){let previous=null;for(const f of obj.periods){const group=f.segments.filter(s=>s.root_id===track.root_id);if(!group.length)continue;if(previous){const old=obj.cross_period_links.find(l=>l.previous_period===previous.frame.period&&l.current_period===f.period&&l.root_id===track.root_id);links.push({...(old||{}),id:old?.id||'L-'+track.root_id+'-'+f.period,previous_period:previous.frame.period,current_period:f.period,root_id:track.root_id,previous_segment:previous.group[0].id,current_segment:group[0].id,previous_segments:previous.group.map(s=>s.id),current_segments:group.map(s=>s.id),kind:old?.review_status==='confirmed'?'same_root':'same_root_candidate',review_status:old?.review_status==='confirmed'?'confirmed':'pending'});}previous={frame:f,group};}}obj.cross_period_links=links;obj.human_reviewed=obj.human_reviewed&&links.every(l=>l.review_status==='confirmed');return obj;};
const oldSaveHtml53=$('saveHtml52').onclick;
$('saveHtml52').onclick=()=>{toolbar53.remove();try{oldSaveHtml53();}finally{$('name').after(toolbar53);state53();}};
window.interactionReady53=(async()=>{await window.structureReady52;refresh52();$('assign52').textContent='仅修改当前段编号';$('rename52').textContent='仅修改本期同号根';status('已载入四期连续编号初稿。点击根后可在上方改号（四期同步），或合并误拆的根段。');return {version:53};})();

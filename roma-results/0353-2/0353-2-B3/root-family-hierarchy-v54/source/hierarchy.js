'use strict';
// Immutable numeric root_id tracks identity. Editable hierarchical labels express
// mother/lateral relationships without conflating identity with temporal class.
let hierarchyOwner54=null,hierarchyList54=null,hierarchyCache54=new Map();
function parseLabel54(value){let s=String(value).trim().toUpperCase().replace(/^R/,'');if(/^\d+$/.test(s))s=s.padStart(4,'0');if(!/^\d{4,6}(?:-(?:N)?[1-9]\d*)*$/.test(s))throw Error('编号格式为 0001、0001-1 或 0001-N1；二级侧根可用 0001-1-N1。');return s;}
function rawHierarchy54(){
 if(!structure52)return new Map();
 if(hierarchyOwner54!==structure52||hierarchyList54!==structure52.root_track_hierarchy){
  const ids=new Set(structure52.periods.flatMap(f=>f.segments.map(s=>s.root_id)));
  const legacy=!Array.isArray(structure52.root_track_hierarchy);if(legacy)structure52.root_track_hierarchy=[];
  const entries=structure52.root_track_hierarchy;for(const rid of ids)if(!entries.some(h=>h.root_id===rid))entries.push({root_id:rid,label:rid.replace(/^R/,''),parent_root_id:null,family_root_id:rid,branch_kind:'independent',review_status:'pending',parentage_confirmed:false});
  hierarchyOwner54=structure52;hierarchyList54=entries;hierarchyCache54=new Map(entries.map(h=>[h.root_id,h]));
  if(legacy){for(const h of entries){const records=structure52.periods.flatMap(f=>f.segments.filter(s=>s.root_id===h.root_id&&s.parent_root_id).map(s=>({segment_id:s.id,parent_root_id:s.parent_root_id})));h.preserved_legacy_parent_assignments=records;const parents=[...new Set(records.map(x=>x.parent_root_id))].filter(id=>hierarchyCache54.has(id)&&id!==h.root_id);if(parents.length===1&&!cycle54(hierarchyCache54,h.root_id,parents[0])){h.parent_root_id=parents[0];h.branch_kind='existing';}else if(parents.length)h.legacy_parent_conflict=true;}relabelTree54(hierarchyCache54);}
 }
 return hierarchyCache54;
}
function label54(rid){return rawHierarchy54().get(rid)?.label||String(rid||'').replace(/^R/,'');}
function family54(rid){return rawHierarchy54().get(rid)?.family_root_id||rid;}
function resolve54(value){const text=String(value).trim().toUpperCase(),map=rawHierarchy54();if(/^R\d{4,6}$/.test(text)&&map.has(text))return text;const label=parseLabel54(text),entry=[...map.values()].find(h=>h.label===label);if(!entry)throw Error('没有找到编号 '+label+'；请先选择或标记母根。');return entry.root_id;}
function cycle54(map,child,parent){const seen=new Set([child]);let cur=parent;while(cur){if(seen.has(cur))return true;seen.add(cur);cur=map.get(cur)?.parent_root_id;}return false;}
function sync54(){
 const map=rawHierarchy54();for(const f of structure52?.periods||[])for(const s of f.segments){const h=map.get(s.root_id);if(h){s.display_label=h.label;s.parent_root_id=h.parent_root_id||null;s.family_root_id=h.family_root_id;s.branch_kind=h.branch_kind;}}
}
function compact54(){
 const map=rawHierarchy54(),ids=new Set(structure52.periods.flatMap(f=>f.segments.map(s=>s.root_id)));let changed=false;
 for(const [id,h] of [...map])if(!ids.has(id)){for(const child of map.values())if(child.parent_root_id===id){child.parent_root_id=h.parent_root_id||null;child.review_status='modified';}map.delete(id);changed=true;}
 if(changed){relabelTree54(map);structure52.root_track_hierarchy=[...map.values()];hierarchyOwner54=null;}sync54();
}
function nextIndependent54(map=rawHierarchy54()){const used=new Set([...map.values()].map(h=>h.label));let i=1;while(used.has(String(i).padStart(4,'0')))i++;return String(i).padStart(4,'0');}
function relabelTree54(map){
 const kids=new Map();for(const h of map.values()){if(h.parent_root_id&&!map.has(h.parent_root_id))h.parent_root_id=null;if(cycle54(map,h.root_id,h.parent_root_id))throw Error('母根关系不能形成循环');if(h.parent_root_id){if(!kids.has(h.parent_root_id))kids.set(h.parent_root_id,[]);kids.get(h.parent_root_id).push(h);}}
 const used=new Set();
 function visit(h,family){h.family_root_id=family;used.add(h.label);const children=(kids.get(h.root_id)||[]).sort((a,b)=>Number(b.label.startsWith(h.label+'-'))-Number(a.label.startsWith(h.label+'-'))||a.root_id.localeCompare(b.root_id));const counters={existing:new Set(),new:new Set()};
  for(const c of children){const kind=c.branch_kind==='new'?'new':'existing',match=c.label.match(/-(N?)(\d+)$/);let n=match&&Boolean(match[1])===(kind==='new')?Number(match[2]):1;while(counters[kind].has(n)||used.has(h.label+'-'+(kind==='new'?'N':'')+n))n++;counters[kind].add(n);c.branch_kind=kind;c.label=h.label+'-'+(kind==='new'?'N':'')+n;visit(c,family);}
 }
 for(const h of [...map.values()].filter(h=>!h.parent_root_id).sort((a,b)=>a.root_id.localeCompare(b.root_id))){if(h.label.includes('-')||used.has(h.label))h.label=nextIndependent54(new Map([...map].filter(([id])=>id!==h.root_id)));h.branch_kind='independent';visit(h,h.root_id);}
}
function applyLabel54(rid,value){
 const map=rawHierarchy54(),h=map.get(rid);if(!h)throw Error('请先选择根');const label=parseLabel54(value);
 if([...map.values()].some(e=>e.root_id!==rid&&e.label===label))throw Error('编号已被另一根使用。同一条根请点“合并另一段为同一根”；侧根请设母根关系。');
 const at=label.lastIndexOf('-');let parent=null,kind='independent';if(at>=0){parent=resolve54(label.slice(0,at));kind=label.slice(at+1).startsWith('N')?'new':'existing';if(cycle54(map,rid,parent))throw Error('不能把根设成自身或后代的侧根');}
 const old=h.label;const replacements=new Map();for(const e of map.values())if(e.root_id===rid||e.label.startsWith(old+'-'))replacements.set(e.root_id,label+e.label.slice(old.length));
 const labels=[...map.values()].map(e=>replacements.get(e.root_id)||e.label);if(new Set(labels).size!==labels.length)throw Error('改号会与已有侧根编号重复，请换一个编号');
 for(const [id,name] of replacements)map.get(id).label=name;h.parent_root_id=parent;h.branch_kind=kind;h.review_status='modified';h.parentage_confirmed=false;
 for(const e of map.values()){let head=e;const seen=new Set();while(head.parent_root_id){if(seen.has(head.root_id))throw Error('母根关系形成循环');seen.add(head.root_id);head=map.get(head.parent_root_id);}e.family_root_id=head.root_id;}
 sync54();
}
function assignSide54(kind){
 const s=selected52();if(!s)throw Error('请先点击要标成侧根的根');const p=resolve54($('motherLabel54').value),map=rawHierarchy54(),h=map.get(s.root_id);
 if(cycle54(map,s.root_id,p))throw Error('母根不能是根自身或它的后代');
 if(h.parent_root_id===p&&h.branch_kind===kind)return;
 const used=[...map.values()].filter(e=>e.parent_root_id===p&&e.branch_kind===kind).map(e=>Number(e.label.split('-').at(-1).replace(/^N/,'')));
 const number=Math.max(0,...used)+1,label=label54(p)+'-'+(kind==='new'?'N':'')+number;
 snapshot52();applyLabel54(s.root_id,label);changed52();status('已设为 '+label+'，四期同号且与母根同色；下一期持续或延伸时仍保留此编号。');
}
const familyPanel54=document.createElement('div');familyPanel54.id='familyPanel54';familyPanel54.innerHTML=`<details open><summary>母根与侧根编号（四期同步）</summary><div id="familyInfo54"></div><label>母根编号<input id="motherLabel54" type="text" list="familyLabels54" placeholder="例如 0001 或 0001-1"><datalist id="familyLabels54"></datalist></label><button id="existingSide54">设为已有侧根 -1、-2…</button><button id="newSide54">设为新生侧根 -N1、-N2…</button><button id="independent54">设为独立母根</button><small>先选侧根，再填母根编号。新生侧根的 N 编号以后保留，不随类别变成持续／延伸而重编号。相交不等于亲缘；不确定时保持独立待复核。</small><label><input id="familyOnly54" type="checkbox">只显示当前根家族</label><div id="familyMembers54"></div></details>`;
$('rootDetails52').before(familyPanel54);
const legend54=document.createElement('div');legend54.id='familyLegend54';legend54.textContent='家族视图：母根与各级侧根同色，独立编号。时序类别颜色另行切换，二者互不替代。';toolbar53.append(legend54);
const originalRgb54=rgbColor52,originalColor54=rootColor52;
rgbColor52=rid=>originalRgb54(family54(rid));rootColor52=rid=>originalColor54(family54(rid));
const originalRegistry54=registry53;
registry53=function(obj=structure52){const rows=originalRegistry54(obj),map=obj===structure52?rawHierarchy54():new Map((obj?.root_track_hierarchy||[]).map(h=>[h.root_id,h]));return rows.map(t=>{const h=map.get(t.root_id);return h?{...t,display_label:h.label,parent_root_id:h.parent_root_id,family_root_id:h.family_root_id,branch_kind:h.branch_kind}:t;});};
const originalMerge54=mergeIds53;
mergeIds53=function(from,to){
 const oldMap=rawHierarchy54(),fromEntry=structuredClone(oldMap.get(from)),toEntry=structuredClone(oldMap.get(to));originalMerge54(from,to);
 const ids=new Set(structure52.periods.flatMap(f=>f.segments.map(s=>s.root_id))),entries=structure52.root_track_hierarchy.filter(e=>ids.has(e.root_id));const map=new Map(entries.map(e=>[e.root_id,e]));
 for(const h of map.values())if(h.parent_root_id===from)h.parent_root_id=to;
 const target=map.get(to);if(target?.parent_root_id===to){target.parent_root_id=fromEntry?.parent_root_id||null;target.label=fromEntry?.label||toEntry.label;target.branch_kind=fromEntry?.branch_kind||'independent';}
 relabelTree54(map);structure52.root_track_hierarchy=[...map.values()];hierarchyOwner54=null;sync54();
};
const originalSerialize54=serialize52;
serialize52=function(){if(structure52)compact54();const obj=originalSerialize54();if(!obj)return obj;obj.hierarchy_version=1;obj.root_track_hierarchy=structuredClone([...rawHierarchy54().values()]);obj.root_tracks=registry53(obj);for(const l of obj.cross_period_links)l.display_label=obj.root_track_hierarchy.find(h=>h.root_id===l.root_id)?.label;return obj;};
const originalValidate54=validateStructure52;
validateStructure52=function(obj,project){
 if(obj?.root_track_hierarchy){const map=new Map(),labels=new Set();for(const h of obj.root_track_hierarchy){if(map.has(h.root_id)||labels.has(h.label)||parseLabel54(h.label)!==h.label)throw Error('母根／侧根编号重复或无效');map.set(h.root_id,h);labels.add(h.label);}
  const roots=new Set(obj.periods.flatMap(f=>f.segments.map(s=>s.root_id)));for(const id of roots)if(!map.has(id))throw Error('部分根缺少母根／侧根编号记录');
  for(const h of map.values()){if(!roots.has(h.root_id)||h.parent_root_id&&!map.has(h.parent_root_id)||cycle54(map,h.root_id,h.parent_root_id))throw Error('母根关系无效或形成循环');if(h.parent_root_id){const p=map.get(h.parent_root_id);if(!h.label.startsWith(p.label+'-')||h.label.slice(p.label.length+1).includes('-'))throw Error('侧根编号与直接母根不一致');}let head=h;while(head.parent_root_id)head=map.get(head.parent_root_id);if(h.family_root_id!==head.root_id)throw Error('家族颜色归属不一致');}
 }
 return originalValidate54(obj,project);
};
function update54(){
 const map=rawHierarchy54(),s=selected52(),h=s?map.get(s.root_id):null;
 $('rootQuickId53').placeholder='例如 0001、0001-1、0001-N1';$('applyRoot53').textContent='应用编号（四期及子根同步）';
 if(document.activeElement!==$('rootQuickId53'))$('rootQuickId53').value=h?.label||'';
 if(document.activeElement!==$('rootId52'))$('rootId52').value=h?.label||'';
 if(document.activeElement!==$('parent52'))$('parent52').value=h?.parent_root_id?label54(h.parent_root_id):'';
 if(document.activeElement!==$('motherLabel54'))$('motherLabel54').value=h?.parent_root_id?label54(h.parent_root_id):'';
 $('familyInfo54').textContent=h?`当前 ${h.label}｜${h.parent_root_id?'母根 '+label54(h.parent_root_id):'独立根／亲缘待复核'}｜家族 ${label54(h.family_root_id)}`:'先在图上选择根，再设置母根与侧根关系';
 $('selection53').textContent=s?`已选 ${h?.label||label54(s.root_id)} · 当前段 ${s.id}`:'先点击一条根或选择家族成员';
 const track=registry53().find(t=>t.root_id===s?.root_id||t.root_id===focusRoot53);
 if(track)$('trackPeriods53').textContent=`同一根 ${track.display_label}：`+['T1','T2','T3','T4'].map(t=>t+' '+(track.observed_periods.includes(t)?'已匹配':'未观察到／待匹配')).join(' ｜ ');
 const options=[...map.values()].sort((a,b)=>a.label.localeCompare(b.label,undefined,{numeric:true}));$('familyLabels54').innerHTML=options.map(v=>`<option value="${v.label}">`).join('');$('rootIds52').innerHTML=options.map(v=>`<option value="${v.label}">`).join('');
 for(const id of ['existingSide54','newSide54','independent54'])$(id).disabled=!s;
 const family=h?.family_root_id,relatives=options.filter(v=>v.family_root_id===family);
 $('familyMembers54').innerHTML=family?'<strong>同一家族成员（点选可跨期复核）</strong>'+relatives.map(v=>`<button data-root54="${v.root_id}" class="${v.root_id===s?.root_id?'active':''}">${v.label}</button>`).join(''):'';
 for(const b of $('familyMembers54').querySelectorAll('button'))b.onclick=()=>{const f=frame52(),segment=f?.segments.find(s=>s.root_id===b.dataset.root54);if(segment)choose52(segment,null,true);else status('本期未观察到 '+label54(b.dataset.root54)+'，请切换到它可见的时期。');};
 for(const o of $('nodeOwner52').options){const sid=o.value;o.textContent=label54(root52(frame52(),sid))+' · '+sid;}
 for(const b of $('arms52').querySelectorAll('button')){const n=node52(),a=n?.arms.find(a=>a.id===b.dataset.arm52);if(a)b.textContent=a.id+' '+label54(root52(frame52(),a.segment_id));}
 if($('wholeRoots53').checked&&frame52()){
  const f=frame52(),query=$('search52').value.toLowerCase(),gg=new Map();for(const q of f.segments){if(!gg.has(q.root_id))gg.set(q.root_id,[]);gg.get(q.root_id).push(q);}
  const values=[...gg].filter(([rid,g])=>!query||label54(rid).toLowerCase().includes(query)||g.some(s=>s.id.toLowerCase().includes(query))).sort((a,b)=>label54(a[0]).localeCompare(label54(b[0]),undefined,{numeric:true}));
  $('segments52').innerHTML=values.map(([rid,g])=>{const rep=g.find(q=>q.id===selectedSegment52)||g.reduce((a,b)=>a.length_px>b.length_px?a:b);return `<option value="${rep.id}" ${rid===s?.root_id?'selected':''}>${label54(rid)} · ${g.length} 段 · ${Math.round(g.reduce((n,q)=>n+q.length_px,0))}px</option>`}).join('');if(!s)$('segments52').selectedIndex=-1;
 }
}
const previousState54=state53;state53=function(){previousState54();update54();};
const previousRefresh54=refresh52;refresh52=function(){previousRefresh54();update54();};
$('applyRoot53').onclick=()=>{try{const s=selected52();if(!s)throw Error('先选择要改号的根');const value=$('rootQuickId53').value;snapshot52();applyLabel54(s.root_id,value);changed52();status('已同步四期编号和所属子根：'+label54(s.root_id));}catch(e){status(e.message)}};
$('existingSide54').onclick=()=>{try{assignSide54('existing')}catch(e){status(e.message)}};$('newSide54').onclick=()=>{try{assignSide54('new')}catch(e){status(e.message)}};
$('independent54').onclick=()=>{try{const s=selected52();if(!s)return;snapshot52();applyLabel54(s.root_id,nextIndependent54());changed52();status('已设为独立根；其后代编号和家族颜色同步。')}catch(e){status(e.message)}};
$('setParent52').onclick=()=>{try{const s=selected52();if(!s)return;if(!$('parent52').value.trim()){$('independent54').click();return;}$('motherLabel54').value=$('parent52').value;assignSide54(rawHierarchy54().get(s.root_id).branch_kind==='new'?'new':'existing');}catch(e){status(e.message)}};
function assignPart54(allPeriod=false){
 const s=selected52();if(!s)return;const value=$('rootId52').value,label=parseLabel54(value);let rid;
 if(label.includes('-'))resolve54(label.slice(0,label.lastIndexOf('-')));
 try{rid=resolve54(value);}catch(e){const map=rawHierarchy54();const literal=String(value).toUpperCase();rid=/^R\d{4,6}$/.test(literal)&&!map.has(literal)?literal:newRoot52();const entry={root_id:rid,label:rid.slice(1),parent_root_id:null,family_root_id:rid,branch_kind:'independent',review_status:'modified',parentage_confirmed:false};snapshot52();structure52.root_track_hierarchy.push(entry);hierarchyOwner54=null;const old=s.root_id;for(const q of frame52().segments)if(q.id===s.id||allPeriod&&q.root_id===old)q.root_id=rid;applyLabel54(rid,label);compact54();changed52();return;}
 snapshot52();const old=s.root_id;for(const q of frame52().segments)if(q.id===s.id||allPeriod&&q.root_id===old){q.root_id=rid;q.review_status='modified';}compact54();changed52();
}
$('assign52').onclick=()=>{try{assignPart54()}catch(e){status(e.message)}};$('rename52').onclick=()=>{try{assignPart54(true)}catch(e){status(e.message)}};
$('newRoot52').onclick=()=>{try{const s=selected52();if(!s)return;const value=nextIndependent54();$('rootId52').value=value;assignPart54();}catch(e){status(e.message)}};
const previousImport54=importProject;importProject=async function(file,options={}){const result=await previousImport54(file,options);hierarchyOwner54=null;rawHierarchy54();sync54();refresh52();render();return result;};
const previousSplit54=split52;split52=function(x,y){previousSplit54(x,y);hierarchyOwner54=null;rawHierarchy54();sync54();refresh52();render();};
$('familyOnly54').onchange=render;
window.hierarchyReady54=(async()=>{await window.interactionReady53;hierarchyOwner54=null;rawHierarchy54();sync54();refresh52();$('view52').options[0].textContent='母根家族同色＋侧根独立编号';$('rootEdit53').click();$('view52').value='identity';render();status('已加载四期母根／侧根编号初稿。母根与侧根同色，已有侧根用 -1，新生侧根用 -N1；之后均保持原编号。');return {version:54};})();

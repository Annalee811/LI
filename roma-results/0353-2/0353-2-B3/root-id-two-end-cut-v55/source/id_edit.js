'use strict';
// UI-only V55: shared draft for the two number inputs; all commits use it.
const drafts55=new Map();
const oldParse55=parseLabel54;
parseLabel54=function(value){
 let text=String(value).normalize('NFKC').trim().toUpperCase().replace(/[‐‑‒–—−]/g,'-').replace(/\s+/g,'').replace(/^R/,'');
 const parts=text.split('-');if(/^\d+$/.test(parts[0]))parts[0]=parts[0].padStart(4,'0');
 for(let i=1;i<parts.length;i++)if(/^N?\d+$/.test(parts[i]))parts[i]=(parts[i].startsWith('N')?'N':'')+Number(parts[i].replace(/^N/,''));
 return oldParse55(parts.join('-'));
};
const note55=document.createElement('div');note55.id='numberEditNotice55';note55.innerHTML='<strong>改根编号：</strong>输入后点“应用／归入编号”，或按 Enter。输入已有编号表示归为同一条根；不同根只想换号，请点“交换编号”。<div id="draftStatus55" role="status" aria-live="polite"></div>';
toolbar53.append(note55);
const swap55=document.createElement('button');swap55.id='swapNumber55';swap55.textContent='仅交换编号，不合并';$('applyRoot53').after(swap55);
const cancel55=document.createElement('button');cancel55.id='cancelNumber55';cancel55.textContent='取消未应用改号';swap55.after(cancel55);
function selectedId55(){return selected52()?.root_id||null;}
function value55(id=selectedId55()){return id?(drafts55.has(id)?drafts55.get(id):label54(id)):'';}
function syncInputs55(){
 const id=selectedId55(),value=value55(id);for(const name of ['rootQuickId53','rootId52'])if($(name).value!==value)$(name).value=value;
 $('applyRoot53').textContent='应用／归入编号（四期同步）';$('assign52').textContent='应用／归入编号（四期同步）';$('rename52').textContent='应用／归入编号（四期同步）';
 swap55.disabled=!id;cancel55.disabled=!id||!drafts55.has(id);
 let message=id?'当前已保存编号：'+label54(id):'先在图上或列表中选择根，再编辑编号。';
 if(id&&drafts55.has(id)){
  try{const label=parseLabel54(value),target=[...rawHierarchy54().values()].find(h=>h.label===label&&h.root_id!==id);message='待应用：'+label54(id)+' → '+label+(target?'。该编号已存在：应用会合并为同一根；不同根请点交换编号。':'。尚未写入标注，请点应用。');}
  catch(e){message=e.message;}
 }
 $('draftStatus55').textContent=message;
}
function input55(event){const id=selectedId55();if(!id)return;drafts55.set(id,event.target.value);syncInputs55();}
for(const name of ['rootQuickId53','rootId52']){
 $(name).addEventListener('input',input55);
 $(name).addEventListener('keydown',event=>{if(event.key==='Enter'&&!event.isComposing){event.preventDefault();commit55('apply');}});
 $(name).readOnly=false;$(name).disabled=false;
}
function swapLabels55(a,b){
 const map=rawHierarchy54(),ha=map.get(a),hb=map.get(b);
 if(ha.parent_root_id!==hb.parent_root_id||ha.branch_kind!==hb.branch_kind)throw Error('只可交换同级、同类型根的编号；母根与侧根请用左侧的亲缘关系设置。');
 const la=ha.label,lb=hb.label,changes=new Map();
 for(const h of map.values())if(h.root_id===a||h.label.startsWith(la+'-'))changes.set(h.root_id,lb+h.label.slice(la.length));else if(h.root_id===b||h.label.startsWith(lb+'-'))changes.set(h.root_id,la+h.label.slice(lb.length));
 const labels=[...map.values()].map(h=>changes.get(h.root_id)||h.label);if(new Set(labels).size!==labels.length)throw Error('交换会造成重复编号，未做修改');
 for(const [id,label] of changes){map.get(id).label=label;map.get(id).review_status='modified';}sync54();
}
function commit55(mode){
 const selected=selected52();if(!selected){status('先点击需要改号的根，再输入编号。');return false;}
 const id=selected.root_id,sid=selected.id;let label,target;
 try{label=parseLabel54(value55(id));target=[...rawHierarchy54().values()].find(h=>h.label===label);}
 catch(e){status(e.message);syncInputs55();return false;}
 if(target?.root_id===id){drafts55.delete(id);syncInputs55();status('当前已经是编号 '+label+'，没有改动。');return true;}
 if(mode==='swap'&&!target){status('目标编号尚未使用，直接点“应用”即可改号。');return false;}
 const before=serialize52(),undoLength=undo52.length;let newId=id;
 try{
  snapshot52();
  if(mode==='swap')swapLabels55(id,target.root_id);
  else if(target){newId=target.root_id;mergeIds53(id,newId);focusRoot53=newId;}
  else applyLabel54(id,label);
  // Validate the committed graph before announcing success or saving it.
  const project=projectObject();validateStructure52(project.root_structure_annotation,project);
  drafts55.delete(id);drafts55.delete(newId);selectedSegment52=sid;changed52();syncInputs55();
  status(mode==='swap'?'编号已交换，四期及子根同步；两条根保持独立。':target?'已归入根 '+label+'，四期同步；可撤销恢复。':'编号已改为 '+label+'，四期及子根已同步。');return true;
 }catch(e){
  structure52=before;hierarchyOwner54=null;rebuild52();undo52.splice(undoLength);selectedSegment52=sid;refresh52();render();status('改号未生效，原标注已保留：'+e.message);syncInputs55();return false;
 }
}
$('applyRoot53').onclick=()=>commit55('apply');$('assign52').onclick=()=>commit55('apply');$('rename52').onclick=()=>commit55('apply');swap55.onclick=()=>commit55('swap');
cancel55.onclick=()=>{drafts55.delete(selectedId55());syncInputs55();status('已取消未应用的编号输入，标注未变。');};
const previousState55=state53;state53=function(){previousState55();syncInputs55();};
const previousRefresh55=refresh52;refresh52=function(){previousRefresh55();syncInputs55();};
const previousUndo55=$('undo52').onclick;$('undo52').onclick=()=>{drafts55.clear();previousUndo55();syncInputs55();};
const previousImport55=importProject;importProject=async function(file,options={}){const result=await previousImport55(file,options);drafts55.clear();syncInputs55();return result;};
const previousMerge55=mergeIds53;mergeIds53=function(from,to){const value=previousMerge55(from,to);for(const frame of structure52.periods)for(const stroke of frame.manual_root_strokes||[])if(stroke.root_id===from)stroke.root_id=to;drafts55.delete(from);drafts55.delete(to);return value;};
for(const name of ['existingSide54','newSide54','independent54','setParent52','newRoot52']){const previous=$(name).onclick;$(name).onclick=function(event){const id=selectedId55(),label=id?label54(id):null;const result=previous?.call(this,event);if(id&&(selectedId55()!==id||label54(id)!==label)){drafts55.delete(id);drafts55.delete(selectedId55());}syncInputs55();return result;};}
$('overlay').addEventListener('dblclick',()=>{if(selected52()&&$('tool52').value!=='pixel'){$('rootQuickId53').focus();$('rootQuickId53').select();status('输入新编号，按 Enter 或点应用。');}});
window.idEditReady55=(async()=>{await window.hierarchyReady54;drafts55.clear();syncInputs55();status('根编号编辑修复版已载入。上下两个编号框已同步；输入后点应用或按 Enter，再保存 JSON。');return {version:55};})();

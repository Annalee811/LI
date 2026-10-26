'use strict';
document.getElementById('temporalReview59')?.remove();document.getElementById('temporalStyle59')?.remove();
const panel59=document.createElement('div');panel59.id='temporalReview59';panel59.innerHTML=`<details open><summary>上传14 · 新生、延续与远距离误归属复核 V59</summary><small>T1仅修正截图所指远距离误归属，其余保留。N-xxxx：独立新根候选；xxxx-Nx：新生侧根。编号表示身份，不能据N字样把后续各期都涂成新生。旧段持续、末端延长可同属一根；未匹配不等于重新出生。</small><button id="showTemporal59">查看时序类别颜色（保留根号）</button><button id="showIdentity59">查看母根家族颜色与根号</button><div id="temporalRootSummary59"></div><select id="reviewType59"><option value="new">新增N-候选：四期核对</option><option value="family">截图所指上下两簇：已拆开</option><option value="joins">纠正跨期错接与断号</option><option value="side">母根中分离出的新侧根</option><option value="classes">已有根重复标新生：本轮修正</option><option value="manual">手改根段的时序冲突：待核</option><option value="parent">可能是侧根：不直接标独立新根</option></select><button id="nextReview59">定位下一处本轮修订</button><div id="reviewInfo59"></div></details>`;
document.getElementById('parentagePanel58').before(panel59);
const style59=document.createElement('style');style59.id='temporalStyle59';style59.textContent='#temporalReview59{background:#f1f6fb;border:2px solid #a6bbd3;padding:8px;margin:6px 0;font-size:13px}#temporalRootSummary59,#reviewInfo59{white-space:pre-wrap;word-break:break-all;padding:6px 0}#reviewType59{width:100%}';document.head.append(style59);
function summary59(){
 const r=rootId58(),h=r?rawHierarchy54().get(r):null;
 if(!h){$('temporalRootSummary59').textContent='选择一条根，可核对四期同号下的时序类别。';return;}
 const live=lifetime58().find(t=>t.root_id===r);let text=h.label+' · 首次观察 '+(live?.states.find(t=>t.observed)?.period||'待定')+'\n';
 for(let fi=0;fi<structure52.periods.length;fi++){
  const f=structure52.periods[fi],ss=f.segments.filter(s=>s.root_id===r);if(!ss.length){text+=f.period+'：暂未可靠匹配（不表示消失或再次新生）\n';continue;}
  const im=files.find(x=>x.name===f.name),state=states.get(f.name+':'+f.image_sha256),values=state?.temporal||((active?.name===f.name)?active.temporal:null);const counts={};
  if(values)for(const s of ss)for(const [x,y] of s.points){const code=values[y*f.width+x];counts[code]=(counts[code]||0)+1;}
  else for(const s of ss)for(const [code,n] of Object.entries(s.semantic_code_counts||{}))counts[code]=(counts[code]||0)+n;
  const names={0:'待定',1:'新生',2:'根尖延长',3:'持续',4:'T1基础'};text+=f.period+'：'+Object.entries(counts).map(([k,v])=>(names[k]||k)+' '+v).join('，')+'\n';
 }
 text+='以上为中心线采样点数，辅助核对；根号与时序颜色分别编辑。';$('temporalRootSummary59').textContent=text;
}
const beforeRefresh59=refresh52;refresh52=function(){beforeRefresh59();summary59();};const beforeState59=state53;state53=function(){beforeState59();summary59();};
$('showTemporal59').onclick=()=>{$('rootEdit53').click();$('view52').value='temporal';$('displayLayer').value='temporal';render();drawLabels57();status('当前显示新生／持续／延长的时序颜色；N前缀和侧根编号保留，同一根可以有持续段及根尖延长段。');};
$('showIdentity59').onclick=()=>{$('rootEdit53').click();$('view52').value='identity';render();drawLabels57();status('当前按母根家族着色，各根保留自己的跨期编号。家族颜色与时序类别颜色含义不同。');};
const cursor59={};$('nextReview59').onclick=async()=>{
 const a=structure52?.temporal_revision_v59||{},kind=$('reviewType59').value;
 const family=a.T1_changes_explicitly_requested_in_screenshot;const list=kind==='family'&&family?[...family.reassigned_segments.filter(e=>e.period==='T1'&&e.root_id===family.lower_root_id),{period:'T1',root_id:family.upper_root_id,segment_id:'T1-S0005',point:[610,100]}]:kind==='new'?a.new_independent_numbering:kind==='joins'?[...(a.continuation_recoveries||[]),...(a.through_continuation_joins||[])]:kind==='side'?[...(a.new_lateral_identity_splits||[]),...(a.mixed_new_axes_separated_pending_parentage||[])]:kind==='classes'?a.temporal_pixel_corrections:kind==='manual'?a.manual_temporal_conflicts:a.parentage_reviews;
 if(!list?.length){$('reviewInfo59').textContent='本类无本轮记录。';return;}
 const at=(cursor59[kind]||0)%list.length;cursor59[kind]=at+1;const e=list[at],r=e.root_id,period=e.period||e.first_seen_period||'T2';await select(Number(period[1])-1);focusRoot53=r;const s=frame52().segments.find(s=>s.id===e.segment_id)||frame52().segments.find(s=>s.root_id===r);if(s)choose52(s,null,false);else selectedSegment52=null;
 const point=e.point||s?.points[Math.floor(s.points.length/2)];if(point){center52(...point);await compare52(...point);}summary59();
 $('reviewInfo59').textContent=`${at+1}/${list.length} · ${label54(r)} · ${period}\n`+(kind==='family'?'截图旧号0005对应上传14的0030位置。上方与下方缺少连接依据，已拆为独立的已有根家族；下方不是T2新根，不加N前缀。':kind==='new'?'当前N号为待复核候选；已排查明显前期同轴及母根附着，短碎片未逐一当独立新根编号。':kind==='side'?(e.candidate_mother_root_id?'不同新轴线从共享号中分离，母根与交叉关系尚待确认，暂不武断赋予母根。':'新侧根从母根共享号中分离，仍属于该母根家族。'):kind==='manual'?'已找到前期同轴，但保留你的手工根段；请对照RGB确认颜色。':kind==='parent'?'有可能的母根附着证据，保留待核，未直接改成独立新根。':kind==='classes'?'仅同轴已有像素由新生改为持续；根尖延长及根系轮廓保留。':'按前期轴线和交叉方向修正同一根的跨期身份。');
};
const beforeImport59=importProject;importProject=async function(file,options={}){const r=await beforeImport59(file,options);summary59();return r;};
window.temporalReady59=(async()=>{await window.parentageReady58;document.title='上传14四期根系：新生与延续细节复核 V59';summary59();status('已载入上传14修订V59。T1截图所指上下两簇已拆开；N-表示后期独立新根候选，母根-Nx表示新侧根。后期同根沿用原号，长度可变。归属不确定的仍待复核。');return {version:59};})();

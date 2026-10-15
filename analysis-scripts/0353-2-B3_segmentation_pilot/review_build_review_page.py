from pathlib import Path
import json,base64,subprocess,sys
ROOT=Path(__file__).resolve().parents[2];KIT=ROOT/'roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation';OUT=ROOT/'results/B3_AI_candidate_review';SOURCE=ROOT/'analysis-scripts/0353-2-B3_segmentation_pilot'
records=json.loads((OUT/'embedded_prefills.json').read_text())
def rle(s):
 b=base64.b64decode(s);out=[];prior=b[0];count=0
 for v in b:
  if v==prior:count+=1
  else:out.extend([prior,count]);prior=v;count=1
 out.extend([prior,count]);return out
for row in records:
 for k in ['mask','color','temporal','viability']:row[k]=rle(row[k])
project=dict(kind='B3_reviewable_annotations',version=1,encoding='rle_value_count',not_training_ground_truth=True,records=records)
(OUT/'review_project.json').write_text(json.dumps(project,separators=(',',':')))
core=(SOURCE/'annotate_core.html').read_text()
core=core.replace('0353-2-B3 人工根系标注 v4','0353-2-B3 根系标注 · 待复核版1').replace('内置60张原色图像，打开自动显示 · 细笔刷 · 未使用模型预标注','60张原色图与现有标注已内置 · 可直接修改 · 修改后保存项目')
core=core.replace('<label>标注者', '''<div class="tip"><strong>60张现有标注已载入，请逐张检查修改</strong><br>蓝色：基础或时序未定根；紫色：持续；绿色：新生候选；黄色：延伸。灰色表示待核对段。未能确认的褐变或存活属性保持未判定。</div>
<div id="periodNav"><button data-period="1">本窗T1</button><button data-period="2">本窗T2</button><button data-period="3">本窗T3</button><button data-period="4">本窗T4</button></div>
<button id="saveProject">保存全部修改（项目JSON）</button>
<label>载入上次修改的项目JSON<input id="projectFile" type="file" accept=".json,application/json"></label>
<div id="projectStatus" role="status" class="tip">正在载入现有标注…</div>
<a id="projectDownload" hidden></a>
<label>标注者''')
core=core.replace('先标出可辨认的完整根系边界，将无法区分的区域标为“不确定”。勾选整图复核后才能导出。','可直接修改现有标注。未完成整图检查时，用“保存全部修改”保存全部图片；本张已复核标签ZIP需要真实标注者与整图确认。')
core=core.replace('本页离线运行，不上传图片。关闭页面前请导出标注。','本页离线运行，不上传图片。关闭或刷新前请保存全部修改的项目JSON。')
core=core.replace('<script id="expected-images" type="application/json">{}</script>','<script id="expected-images" type="application/json">{}</script>\n<script id="embedded-prefills" type="application/json">'+json.dumps(project,separators=(',',':'))+'</script>')
core=core.replace("if(!annotator||!active.reviewed){status('请填写标注者，并确认已检查整张图。未完成的标注不能导出为人工真值。');return}", "if(!annotator||!active.reviewed){status('请填写标注者，并确认已检查整张图。未完成时点击“保存全部修改”保存项目，不必勾选复核。');return}")
core=core.replace("label_origin:'manual_brush_and_explicit_whole_image_review'", "prefill_origin:active.prefill_origin||null,label_origin:active.prefill_origin?'assisted_prefill_with_explicit_person_review':'manual_brush_and_explicit_whole_image_review'")
core=core.replace("editor:'0353-2-B3 offline human editor v4'", "editor:'0353-2-B3 reviewable annotation editor 1'")
core=core.replace("function remember(){if(!active)return;active.reviewed", "function remember(){if(!active)return;active.annotator=$('annotator').value.trim();active.reviewed")
core=core.replace("$('evidence').value=active.evidence;editEnabled(true)","$('evidence').value=active.evidence;if(active.annotator)$('annotator').value=active.annotator;editEnabled(true)")
module='''
function decodeRuns(a){if(!Array.isArray(a)||a.length%2||a.length>2*N)throw Error('项目编码不正确');const b=new Uint8Array(N);let pos=0;for(let i=0;i<a.length;i+=2){const value=a[i],count=a[i+1];if(!Number.isInteger(value)||value<0||value>255||!Number.isInteger(count)||count<1||pos+count>N)throw Error('项目像素不正确');b.fill(value,pos,pos+count);pos+=count}if(pos!==N)throw Error('项目尺寸不正确');return b}
function encodeRuns(b){const a=[];let prior=b[0],count=0;for(const value of b){if(value===prior)count++;else{a.push(prior,count);prior=value;count=1}}a.push(prior,count);return a}
function parseProject(data){
 if(data?.kind!=='B3_reviewable_annotations'||data.version!==1||data.encoding!=='rle_value_count'||!Array.isArray(data.records)||data.records.length>60||data.not_training_ground_truth!==true)throw Error('不是本实验标注项目');
 const parsed=new Map();for(const row of data.records){
  if(!Object.hasOwn(expectedImages,row.name)||row.image_sha256!==expectedImages[row.name]||parsed.has(row.name))throw Error('图像名称、SHA256或重复记录不正确');
  const s=fresh();for(const k of ['mask','color','temporal','viability'])s[k]=decodeRuns(row[k]);
  if(s.mask.some(v=>![0,64,128,255].includes(v))||s.color.some(v=>v>2)||s.temporal.some(v=>v>3)||s.viability.some(v=>v>3))throw Error('项目标签不正确');
  for(let i=0;i<N;i++)if(s.mask[i]!==255&&(s.color[i]||s.temporal[i]||s.viability[i]))throw Error('非根区域包含根属性');
  s.reviewed=row.reviewed===true;s.evidence=typeof row.evidence==='string'?row.evidence:'';s.annotator=typeof row.annotator==='string'?row.annotator:'';s.prefill_origin=typeof row.prefill_origin==='string'?row.prefill_origin:'assistant_visual_T3_corridor_seeds_plus_automated_temporal_flow_and_local_profile_fit';parsed.set(row.name,s);
 }return parsed;
}
function serializeProject(){remember();return {kind:'B3_reviewable_annotations',version:1,encoding:'rle_value_count',not_training_ground_truth:true,saved_at:new Date().toISOString(),records:[...states].map(([name,s])=>({name,image_sha256:expectedImages[name],mask:encodeRuns(s.mask),color:encodeRuns(s.color),temporal:encodeRuns(s.temporal),viability:encodeRuns(s.viability),reviewed:s.reviewed,evidence:s.evidence,annotator:s.annotator||'',prefill_origin:s.prefill_origin||null}))}}
$('saveProject').onclick=()=>{try{const data=serializeProject(),a=$('projectDownload');if(a.dataset.url)URL.revokeObjectURL(a.dataset.url);const url=URL.createObjectURL(new Blob([JSON.stringify(data)],{type:'application/json'}));a.href=url;a.dataset.url=url;a.download='0353-2-B3_review_project_'+Date.now()+'.json';a.hidden=false;a.textContent='点击这里下载项目JSON（'+data.records.length+'张）';a.click();$('projectStatus').textContent='已生成全部修改项目。请确认下载文件实际存在；下次用“载入上次修改的项目JSON”恢复。'}catch(e){$('projectStatus').textContent='保存项目失败：'+e.message}};
$('projectFile').onchange=async e=>{try{const f=e.target.files[0];if(!f)return;if(f.size>100*1024*1024)throw Error('项目文件过大');const parsed=parseProject(JSON.parse(await f.text()));remember();active=null;for(const [name,s] of parsed)states.set(name,s);await select(Math.max(0,index));$('projectStatus').textContent='已恢复'+parsed.size+'张项目标注。'}catch(err){$('projectStatus').textContent='项目恢复失败：'+err.message}finally{e.target.value=''}};
for(const b of document.querySelectorAll('[data-period]'))b.onclick=()=>{if(index<0)return;const name=files[index].name.replace(/_T[1-4]\\.png$/,'_T'+b.dataset.period+'.png'),i=files.findIndex(f=>f.name===name);if(i>=0)select(i)};
function loadPrefills(){const parsed=parseProject(JSON.parse($('embedded-prefills').textContent));for(const [name,s] of parsed)states.set(name,s);$('projectStatus').textContent='已载入'+parsed.size+'张现有标注，均待逐图复核；可直接修改并保存项目。'}
'''
core=core.replace("window.addEventListener('beforeunload'",module+"\nwindow.addEventListener('beforeunload'")
core=core.replace('window.datasetReady=loadEmbedded();','loadPrefills();\nwindow.datasetReady=loadEmbedded();')
(OUT/'annotate_review_core.html').write_text(core)
subprocess.run([sys.executable,str(SOURCE/'build_annotation_page.py'),'--kit',str(KIT),'--template',str(OUT/'annotate_review_core.html'),'--output',str(OUT/'annotate.html')],check=True)
print('REVIEWABLE_ANNOTATION_EDITOR_READY',len(records))

from pathlib import Path
import json,re,subprocess,sys
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/B3_reannotation_v2';KIT=ROOT/'roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation';SOURCE=ROOT/'analysis-scripts/0353-2-B3_segmentation_pilot'
project=json.loads((OUT/'review_project.json').read_text());core=(KIT/'reviewable-annotations/annotate.html').read_text()
for ident,empty in [('embedded-dataset','[]'),('expected-images','{}')]:
 core=re.sub(r'(<script id="'+ident+r'" type="application/json">).*?(</script>)',lambda m:m[1]+empty+m[2],core,flags=re.S)
core=re.sub(r'(<script id="embedded-prefills" type="application/json">).*?(</script>)',lambda m:m[1]+json.dumps(project,separators=(',',':'))+m[2],core,flags=re.S)
core=core.replace('根系标注 · 待复核版1','根系标注 · 基线与物体分层版2')
core=core.replace('let files=[],index=-1,mode=255,','let paintScene=1,files=[],index=-1,mode=255,')
core=core.replace("mask:new Uint8Array(N).fill(64),color:","cohort:new Uint8Array(N),scene:new Uint8Array(N),instance:new Uint8Array(N),mask:new Uint8Array(N).fill(64),color:")
core=core.replace('<label>覆盖层显示<select id="displayLayer">','''<label>根的首次观察批次（与持续状态独立）<select id="cohort"><option value="0">未确定</option><option value="1">T1 已有基线根</option><option value="2">首次观察于T2</option><option value="3">首次观察于T3</option><option value="4">首次观察于T4</option></select></label>
<label>本窗根编号（0＝未确定）<input id="rootId" type="number" min="0" max="255" value="0"></label>
<div class="tip">T1所有可见根均为已有基线根，不判新生、延伸或持续。后期“来源批次”和“变化状态”分别保存：T1已有根可以在T2持续，T2首次观察的根也可以在T3持续。</div>
<div id="sceneTools"><button data-scene="2">水滴</button><button data-scene="3">土壤</button><button data-scene="4">其他非根背景</button><button data-scene="0">清除标记</button></div>
<label>覆盖层显示<select id="displayLayer"><option value="cohort">根的来源批次</option><option value="scene">物体分类（根/水/土）</option>''')
core=core.replace('60张现有标注已载入，请逐张检查修改','示范窗四期已重描；其余窗口仍待详细复核').replace('蓝色：基础或时序未定根；紫色：持续；绿色：新生候选；黄色：延伸。灰色表示待核对段。未能确认的褐变或存活属性保持未判定。','T1统一显示已有根。示范窗按各期原图单独描画，未按亮纹自动吸附。其余56张只做了保守清理，不代表已完成详细补根。').replace('本实验标注图保持448×448原始像素坐标。','本实验标注图保持448×448原始像素坐标。<div id="pixelInfo" class="tip">将鼠标移到图中，可查看物体类别、来源批次和变化状态。</div>')
core=core.replace("if(layer!=='overview')return maps[layer]", "if(layer==='cohort'||layer==='scene')return 'root';if(layer!=='overview')return maps[layer]")
core=core.replace("const classes={", "const cohortNames=['未确定','T1已有基线根','首次观察于T2','首次观察于T3','首次观察于T4'],cohortRGB=[[100,116,139],[37,99,235],[22,163,74],[234,179,8],[192,38,211]],sceneNames=['未标注','根','水滴','土壤','其他非根背景','不确定/遮挡'],sceneRGB=[[0,0,0],[37,99,235],[6,182,212],[174,135,89],[180,180,180],[249,115,22]];\nconst classes={")
core=core.replace("const keys=$('displayLayer').value==='overview'?Object.keys(classes):[...maps[$('displayLayer').value],'ignore'];", "const layer=$('displayLayer').value;const keys=layer==='overview'?Object.keys(classes):(maps[layer]?[...maps[layer],'ignore']:[]);")
core=core.replace("$('legend').innerHTML=keys.map(k=>'<span>'+swatch(k)+'</span>').join('');", "$('legend').innerHTML=keys.map(k=>'<span>'+swatch(k)+'</span>').join('');if(layer==='cohort')$('legend').textContent=cohortNames.map((n,i)=>i+'：'+n).join('；');if(layer==='scene')$('legend').textContent=sceneNames.map((n,i)=>i+'：'+n).join('；');if(paintScene!==1)$('brushPreview').textContent='当前画笔：'+sceneNames[paintScene];else if(isT1())$('brushPreview').textContent='当前画笔：T1已有基线根';")
core=core.replace("function setMode(v,id){mode=v;", "function setMode(v,id){mode=v;paintScene=v===255?1:v===128?5:v===64?0:4;")
start=core.index('function render(){');end=core.index('updateBrushPreview();',start)
core=core[:start]+'''function render(){
 ovctx.clearRect(0,0,$('overlay').width,$('overlay').height);if(!active||!$('show').checked)return;const im=ovctx.createImageData(W,W),layer=$('displayLayer').value;
 for(let i=0;i<N;i++){const v=active.mask[i],j=i*4;let rgb=null;
  if(layer==='scene'){if(active.scene[i])rgb=sceneRGB[active.scene[i]]}
  else if(v===255){rgb=layer==='cohort'?cohortRGB[active.cohort[i]]:classes[category(active.color[i],active.temporal[i],active.viability[i],layer)].rgb}
  else if(v===128)rgb=sceneRGB[5];if(rgb){im.data.set(rgb,j);im.data[j+3]=layer==='scene'?120:175}
 }ovctx.putImageData(im,0,0)
}
''' +core[end:]
start=core.index('function dot(');end=core.index('function line(',start)
core=core[:start]+'''function isT1(){return index>=0&&/_T1\\.png$/i.test(files[index].name)}
function writePixel(i){
 const root=paintScene===1;active.mask[i]=root?255:paintScene===5?128:paintScene===0?64:0;active.scene[i]=paintScene;
 active.color[i]=root?Number($('color').value):0;active.temporal[i]=root&&!isT1()?Number($('temporal').value):0;active.viability[i]=root?Number($('viability').value):0;
 active.cohort[i]=root?(isT1()?1:Number($('cohort').value)):0;active.instance[i]=root?Math.max(0,Math.min(255,Number($('rootId').value)||0)):0;
}
function dot(x,y){if(Number($('brush').value)===1){const xx=Math.max(0,Math.min(W-1,Math.floor(x))),yy=Math.max(0,Math.min(W-1,Math.floor(y)));writePixel(yy*W+xx);return}const r=Number($('brush').value)/2;for(let yy=Math.max(0,Math.floor(y-r));yy<=Math.min(W-1,Math.ceil(y+r));yy++)for(let xx=Math.max(0,Math.floor(x-r));xx<=Math.min(W-1,Math.ceil(x+r));xx++)if((xx+.5-x)**2+(yy+.5-y)**2<=r*r)writePixel(yy*W+xx)}
''' +core[end:]
core=core.replace("mask:active.mask.slice(),color:","cohort:active.cohort.slice(),scene:active.scene.slice(),instance:active.instance.slice(),mask:active.mask.slice(),color:")
core=core.replace("const p=xy(e);line(last,p);", "const p=xy(e);line(last,p);")
core=core.replace("$('overlay').onpointermove=e=>{if(!painting||!active)return;", "$('overlay').onpointermove=e=>{if(active){const p=xy(e),x=Math.max(0,Math.min(W-1,Math.floor(p[0]))),y=Math.max(0,Math.min(W-1,Math.floor(p[1]))),i=y*W+x;$('pixelInfo').textContent='像素('+x+','+y+')：'+sceneNames[active.scene[i]]+'；根编号 '+active.instance[i]+'；来源 '+cohortNames[active.cohort[i]]+'；状态 '+['未判定','首次观察候选','延伸','持续可见'][active.temporal[i]]}if(!painting||!active)return;")
core=core.replace("if(verified){\n   active=", "if(verified){\n   active=")
core=core.replace("editEnabled(true);render();", "editEnabled(true);$('temporal').disabled=isT1();$('cohort').disabled=isT1();if(isT1()){$('temporal').value='0';$('cohort').value='1'}else{$('cohort').value='0'}$('rootId').value='0';for(const b of document.querySelectorAll('#palette button'))if(['new','growing','persistent'].includes(b.dataset.category))b.disabled=isT1();render();updateBrushPreview();")
core=core.replace("['fg','bg','ignore','undo','reviewed','export','color','temporal','viability','evidence']", "['fg','bg','ignore','undo','reviewed','export','color','temporal','viability','evidence','cohort','rootId']")
core=core.replace("b.disabled=!enabled)}", "b.disabled=!enabled);document.querySelectorAll('[data-scene]').forEach(b=>b.disabled=!enabled)}")
# Additional layers are exported without changing existing scientific root-mask codes.
core=core.replace("prefill_origin:active.prefill_origin||null,label_origin:", "root_first_observed_encoding:{unknown:0,T1_baseline:1,T2:2,T3:3,T4:4},scene_encoding:{unreviewed:0,root:1,water:2,soil:3,other_background:4,uncertain:5},prefill_origin:active.prefill_origin||null,label_origin:")
core=core.replace("[id+'_viability.png',await png(active.viability)]]", "[id+'_viability.png',await png(active.viability)],[id+'_cohort.png',await png(active.cohort)],[id+'_scene.png',await png(active.scene)],[id+'_instance.png',await png(active.instance)]]")
start=core.index('function parseProject(');end=core.index("$('saveProject').onclick",start)
core=core[:start]+'''function parseProject(data){
 if(data?.kind!=='B3_reviewable_annotations'||![1,2].includes(data.version)||data.encoding!=='rle_value_count'||!Array.isArray(data.records)||data.records.length>60||data.not_training_ground_truth!==true)throw Error('不是本实验项目');
 const parsed=new Map();for(const row of data.records){
  if(!Object.hasOwn(expectedImages,row.name)||row.image_sha256!==expectedImages[row.name]||parsed.has(row.name))throw Error('图像名称或SHA256不正确');
  const s=fresh();for(const k of ['mask','color','temporal','viability'])s[k]=decodeRuns(row[k]);
  if(data.version===2){for(const k of ['cohort','scene','instance'])s[k]=decodeRuns(row[k])}else{for(let i=0;i<N;i++){s.scene[i]=s.mask[i]===255?1:s.mask[i]===128?5:s.mask[i]===0?4:0;s.cohort[i]=s.mask[i]===255&&/_T1\\.png$/i.test(row.name)?1:0}}
  if(s.mask.some(v=>![0,64,128,255].includes(v))||s.color.some(v=>v>2)||s.temporal.some(v=>v>3)||s.viability.some(v=>v>3)||s.cohort.some(v=>v>4)||s.scene.some(v=>v>5))throw Error('项目标签代码不正确');
  for(let i=0;i<N;i++){if(s.mask[i]!==255&&(s.color[i]||s.temporal[i]||s.viability[i]||s.cohort[i]||s.instance[i]))throw Error('非根区域有根属性');if((s.scene[i]===1)!==(s.mask[i]===255))throw Error('根与物体分类矛盾')}
  s.reviewed=row.reviewed===true;s.evidence=typeof row.evidence==='string'?row.evidence:'';s.annotator=typeof row.annotator==='string'?row.annotator:'';s.prefill_origin=typeof row.prefill_origin==='string'?row.prefill_origin:'legacy_project_import';s.review_status=typeof row.review_status==='string'?row.review_status:'needs_review';
  if(/_T1\\.png$/i.test(row.name))for(let i=0;i<N;i++)if(s.mask[i]===255){if(s.cohort[i]!==1||s.temporal[i]!==0)s.reviewed=false;s.cohort[i]=1;s.temporal[i]=0}
  parsed.set(row.name,s);
 }return parsed;
}
function serializeProject(){remember();return {kind:'B3_reviewable_annotations',version:2,encoding:'rle_value_count',not_training_ground_truth:true,saved_at:new Date().toISOString(),records:[...states].map(([name,s])=>({name,image_sha256:expectedImages[name],...Object.fromEntries(['mask','color','temporal','viability','cohort','scene','instance'].map(k=>[k,encodeRuns(s[k])])),reviewed:s.reviewed,evidence:s.evidence,annotator:s.annotator||'',prefill_origin:s.prefill_origin||null,review_status:s.review_status||'needs_review'}))}}
for(const b of document.querySelectorAll('[data-scene]'))b.onclick=()=>{const v=Number(b.dataset.scene);setMode(v===0?64:0,'bg');paintScene=v;updateBrushPreview()};
''' +core[end:]
core=core.replace("'已载入'+parsed.size+'张现有标注，均待逐图复核；可直接修改并保存项目。'", "'已载入'+parsed.size+'张。T1全部已有根；示范窗四期重描，其余窗口仍需详细复核。'")
core=core.replace('<option value="overview">','<option value="overview" selected>')
core=core.replace("rgb=layer==='cohort'?cohortRGB[active.cohort[i]]:","rgb=layer==='cohort'?cohortRGB[active.cohort[i]]:(layer==='overview'&&isT1())?cohortRGB[1]:")
core=core.replace("const c=classes[key];$('color').value=","const c=classes[key];if(key==='new'&&!isT1())$('cohort').value=files[index].name.match(/_T([1-4])/)[1];$('color').value=")
(OUT/'annotate_core_v2.html').write_text(core)
subprocess.run([sys.executable,str(SOURCE/'build_annotation_page.py'),'--kit',str(KIT),'--template',str(OUT/'annotate_core_v2.html'),'--output',str(OUT/'annotate.html')],check=True)
print('LAYERED_EDITOR_V2_READY')

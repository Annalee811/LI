"""Editable Human temporal classes with U-Net root morphology; original human project retained."""
import argparse,base64,json,re,shutil
from pathlib import Path
import common as c
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--repository',type=Path,default=Path.cwd());a=p.parse_args();repo=a.repository.resolve();run=a.run.resolve();dest=run/'annotation_review';dest.mkdir(exist_ok=True)
html=(repo/'roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/custom-images-v1/annotate.html').read_text()
html=html.replace("root:{label:'未分类根',rgb:[37,99,235]","root:{label:'未分类根',rgb:[148,163,184]")
html=html.replace('根系标注 · 自选图片版','0353-2-B3 人工时序与根形态 v17')
html=html.replace('<option value="3">持续存在</option>','<option value="3">持续根</option><option value="4">T1 基础根</option>')
html=html.replace("const classes={root:","const classes={baseline:{label:'T1 基础根',rgb:[0,114,178],temporal:4},root:")
html=html.replace("new:{label:'新生候选',rgb:[22,163,74]","new:{label:'新生根',rgb:[0,158,115]")
html=html.replace("growing:{label:'延伸生长',rgb:[234,179,8]","growing:{label:'生长／延伸根',rgb:[230,171,2]")
html=html.replace("persistent:{label:'持续根',rgb:[147,51,234]","persistent:{label:'持续根',rgb:[136,86,167]")
html=html.replace("temporal:['root','new','growing','persistent']","temporal:['root','new','growing','persistent','baseline']")
html=html.replace("Object.keys(classes).filter(k=>k!=='ignore')","['baseline','new','growing','persistent','root']")
html=html.replace("active.temporal[i]=mode===255?Number($('temporal').value):0","active.temporal[i]=mode===255?currentTemporalCode():0")
html=html.replace("function remember(){", "function currentTemporalCode(){if(active&&/^T1_/.test(active.name))return 4;const t=Number($('temporal').value);return t===4?3:t}\nfunction remember(){")
html=html.replace("temporal:[0,1,2,3]","temporal:[0,1,2,3,4]")
html=html.replace("const pending=new Map(),codes=","window.referenceProvenance=obj.annotation_revision||obj.source_provenance||null;const pending=new Map(),codes=")
html=html.replace("return {kind:'custom_file_root_annotation',version:1,annotator:","return {annotation_revision:window.referenceProvenance||null,kind:'custom_file_root_annotation',version:1,annotator:")
html=html.replace("temporal_encoding:{unclassified:0,first_observed_candidate:1,elongating:2,persistent:3}","temporal_encoding:{unclassified:0,newly_observed:1,tip_growth:2,persistent:3,T1_baseline:4}")
html=html.replace("label_origin:'manual_brush_and_explicit_whole_image_review'","label_origin:window.referenceProvenance?'manual_temporal_U_Net_morphology_reference':'manual_brush_and_explicit_whole_image_review',source_provenance:window.referenceProvenance||null")
html=html.replace("id+'_human_annotation.zip'","id+'_reference_annotation.zip'").replace('导出本张人工标注 ZIP','导出本张标注 ZIP').replace('已导出本张人工标注。','已导出本张标注。')
html=html.replace('未完成的标注不能导出为人工真值。','可随时保存进度JSON；导出当前图前请完成复核。')
html=html.replace('蓝色：未分类根；绿色：新生候选；黄色：延伸生长；紫色：持续根；棕色：褐变；橙色：疑似死亡；红色：确认死亡；灰色：不确定区域。','蓝色：T1基础根；绿色：新生根；黄色：根尖生长／延伸；紫色：持续根；灰色：不确定区域。')
html+='\n<style>#b3QuickTools .tip{font-size:12px}#displayLayer option:not([value="temporal"]){display:none}</style>\n'
project=json.loads((run/'data/active_progress.json').read_text());images=[]
for im in project['images']:
 f=run/'data/upper'/im['name'];assert c.sha(f)==im['image_sha256'];images.append(dict(name=im['name'],data=base64.b64encode(f.read_bytes()).decode()))
html+='\n<script id="temporal-reference-project" type="application/json">'+json.dumps(project,ensure_ascii=False).replace('</','<\\/')+'</script>\n'
html+='<script id="temporal-reference-images" type="application/json">'+json.dumps(images)+'</script>\n'
html+='''<script>
window.temporalRevisionReady=(async()=>{
 await window.zoom60Ready;
 const project=JSON.parse(document.getElementById('temporal-reference-project').textContent);
 const data=JSON.parse(document.getElementById('temporal-reference-images').textContent);
 const originals=data.map(v=>{const s=atob(v.data),a=new Uint8Array(s.length);for(let i=0;i<s.length;i++)a[i]=s.charCodeAt(i);return new File([a],v.name,{type:'image/png'})});
 await loadFiles(originals);await importProject(new File([JSON.stringify(project)],'active_progress.json',{type:'application/json'}));
 window.referenceProvenance=project.annotation_revision;
 for(const id of ['color','viability'])document.getElementById(id).closest('label').style.display='none';
 const previousSelect=select;
 select=async function(i){await previousSelect(i);const baseline=active&&/^T1_/.test(active.name);const sel=document.getElementById('temporal');for(const option of sel.options)option.disabled=baseline?option.value!=='4':option.value==='4';sel.value=baseline?'4':(sel.value==='4'?'3':sel.value);document.querySelectorAll('#palette button').forEach(b=>{if(b.dataset.category==='baseline')b.disabled=!baseline;else if(['new','growing','persistent'].includes(b.dataset.category))b.disabled=baseline});document.getElementById('displayLayer').value='temporal';render()};
 await select(0);document.getElementById('displayLayer').value='temporal';render();
 const note=document.createElement('div');note.className='tip';note.textContent='本版保留原人工时序类别，根的粗细与范围按U-Net调整；新增而无可靠人工类别的根以灰色显示。T1蓝色为基础根；后期紫色持续、绿色新生、黄色根尖生长。空间以T1为基准，类别始终对照紧邻前一期。仅上半幅参与本轮训练，原始标注另存。';document.getElementById('b3QuickTools').prepend(note);
 status('已载入四期上半幅及U-Net时序修订参考。可修改、Ctrl+Z、保存全部进度；最高60倍，切换时期保持倍率。保存JSON与导出ZIP均保留来源记录。');
 window.loadedTemporalRevision={images:4,baselineCode:4,referenceType:project.annotation_revision.reference_type};
})();
</script>'''
(dest/'annotate.html').write_text(html)
for n in ['active_progress.json','original_human_progress.json','revision_provenance.json','revision_log.json']:shutil.copy2(run/'data'/n,dest/n)
(dest/'使用说明.txt').write_text('解压后用Chrome/Edge打开 annotate.html。四张上半幅RGB和修订参考已内置。支持细笔刷、Ctrl+Z、最高60倍、切图保持倍率、保存JSON与导出。\n蓝色T1基础根；紫色持续；绿色新生；黄色生长/延伸。后期每张均相对紧邻前一期分类。T1所有根固定为基础根。原始完整人工JSON另存，当前形态参考不改变保留像素的原人工时序类别。\n根形态按U-Net修订，时序类别以原人工标记为准；原有根像素如不在U-Net掩膜内不进入形态参考，但完整原始标记另存。新增而不能可靠继承类别的根保持未分类，请重点复核交叉、侧根与根尖。未加入死亡类别。\n')
print('EDITABLE_TEMPORAL_REVIEW_BUILT',len(html),flush=True)

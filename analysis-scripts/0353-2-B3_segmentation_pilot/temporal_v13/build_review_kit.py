from pathlib import Path
import json,base64,hashlib,shutil
import argparse
p=argparse.ArgumentParser();p.add_argument('--repository',type=Path,default=Path.cwd());p.add_argument('--run',type=Path,default=Path('results/B3_temporal_v13'));p.add_argument('--review',type=Path,default=Path('results/B3_annotation_revision'));a=p.parse_args();root=a.repository.resolve();run=a.run.resolve();review=run/'annotation_review';review.mkdir(exist_ok=True);src=a.review.resolve();project=json.loads((run/'data/active_progress.json').read_text());dataset=[]
for im in project['images']:
 f=root/'results/B3_full_canvas_v1/originals'/im['name'];assert hashlib.sha256(f.read_bytes()).hexdigest()==im['image_sha256'];dataset.append(dict(name=im['name'],mime='image/png',data=base64.b64encode(f.read_bytes()).decode()))
html=(root/'roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/custom-images-v1/annotate.html').read_text();html=html.replace('<title>根系标注 · 自选图片版</title>','<title>0353-2-B3 四期时序标注修订</title>').replace('<h1>根系标注 · 自选图片版</h1>','<h1>0353-2-B3 四期时序标注修订</h1>')
html+='\n<script id="revision-project" type="application/json">'+json.dumps(project,ensure_ascii=False).replace('</','<\\/')+'</script>\n'
html+='<script id="revision-images" type="application/json">'+json.dumps(dataset)+'</script>\n'
html+='<script id="revision-log" type="application/json">'+(src/'correction_log.json').read_text().replace('</','<\\/')+'</script>\n'
html+='''<script>
window.revisionReady=(async()=>{
 await window.zoom60Ready;
 const project=JSON.parse(document.getElementById('revision-project').textContent);
 const data=JSON.parse(document.getElementById('revision-images').textContent);
 const log=JSON.parse(document.getElementById('revision-log').textContent);
 const originals=data.map(v=>{const s=atob(v.data),a=new Uint8Array(s.length);for(let i=0;i<s.length;i++)a[i]=s.charCodeAt(i);return new File([a],v.name,{type:v.mime})});
 await loadFiles(originals);await importProject(new File([JSON.stringify(project)],'active_progress.json',{type:'application/json'}));
 document.getElementById('displayLayer').value='temporal';render();
 const label=document.createElement('label');label.textContent='跳转到修订根段';const jump=document.createElement('select');jump.id='revisionRegion';jump.innerHTML='<option value="">选择修订位置</option>'+log.records.map((r,i)=>'<option value="'+i+'">'+r.review_id+' '+r.current+' 新生→持续</option>').join('');label.append(jump);document.getElementById('b3QuickTools').prepend(label);
 jump.onchange=async()=>{if(jump.value==='')return;const r=log.records[Number(jump.value)];await select(['T1','T2','T3','T4'].indexOf(r.current));const b=r.bbox,z=Number(document.getElementById('zoom').value),v=document.getElementById('b3ImageViewport');v.scrollLeft=(b[0]+b[2])/2*z-v.clientWidth/2;v.scrollTop=(b[1]+b[3])/2*z-v.clientHeight/2;status(r.review_id+'：新生→持续；'+r.pixels+' 像素。'+r.reason)};
 status('已载入4张完整原图及修订标注，可直接修改、Ctrl+Z和保存进度。训练只取上半部分；下半部原有笔迹保留，不参与训练。修订4处328像素，仅修改时序类别。');
 window.loadedRevision={images:originals.length,changed_pixels:log.changed_pixels,regions:log.regions};
})();
</script>'''
(review/'annotate.html').write_text(html)
for name in ['accepted_decisions.json','candidate_components.json','correction_log.json']:shutil.copyfile(src/name,review/name)
for f in src.glob('R*'):
 if f.suffix in ['.png','.jpg']:shutil.copyfile(f,review/f.name)
for name in ['active_progress.json','original_human_progress.json']:shutil.copyfile(run/'data'/name,review/name)
(review/'使用说明.txt').write_text('解压后用Chrome或Edge打开 annotate.html。四张原始RGB与修订标注已内置，无需重新选文件。支持Ctrl+Z、保存JSON、最高60倍显示，切换图片保持倍率。可在工具栏跳转4处修订根段。原始标注和修订JSON均保留。\n仅上半部分参与本轮训练。已修订T2 203像素、T3 91像素、T4 34像素，新生改为持续；没有修改根背景边界、颜色、存活属性和下半部分。原图视觉核查由助手完成，未使用模型预测作为改标依据，尚不等同于独立专家仲裁。\n')
print('REVIEW_KIT_BUILT',len(html),review)

#!/usr/bin/env python3
"""Add zoom/pan and fixed side tools to the original v4; keep original image bytes."""
from pathlib import Path
import argparse,subprocess,sys,json,csv,hashlib,shutil
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--kit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--site',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
core=(HERE/'annotate_core.html').read_text();patch=(HERE/'legacy_zoom60_patch.js').read_text()
core=core.replace('</script></html>','</script><script>\n'+patch+'\n</script></html>')
assert patch in core
template=a.output/'annotate_core_zoom60.html';template.write_text(core)
subprocess.run([sys.executable,str(HERE/'build_annotation_page.py'),'--kit',str(a.kit),'--template',str(template),'--output',str(a.output/'annotate.html')],check=True)
(a.output/'legacy_zoom60_patch.js').write_text(patch)
if a.site:
    a.site.mkdir(parents=True,exist_ok=True);(a.site/'images').mkdir(exist_ok=True)
    rows=list(csv.DictReader((a.kit/'split_manifest.csv').open(encoding='utf-8-sig')))
    expected={};images=[]
    for r in rows:
        f=a.kit/r['image_path'];b=f.read_bytes();assert hashlib.sha256(b).hexdigest()==r['image_sha256']
        shutil.copyfile(f,a.site/'images'/f.name);expected[f.name]=r['image_sha256'];images.append(dict(name=f.name,url='./images/'+f.name))
    web=core.replace('<script id="embedded-dataset" type="application/json">[]</script>','<script id="embedded-dataset" type="application/json">'+json.dumps(images,separators=(',',':'))+'</script>')
    web=web.replace('<script id="expected-images" type="application/json">{}</script>','<script id="expected-images" type="application/json">'+json.dumps(expected,separators=(',',':'))+'</script>')
    web=web.replace("async function decodeImage(file){", "async function decodeImage(file){\n if(file.getBlob)file=await file.getBlob();")
    start=web.index('async function loadEmbedded(){');end=web.index("$('builtin').onclick",start)
    web=web[:start]+'''async function loadEmbedded(){
 try{const data=JSON.parse($('embedded-dataset').textContent);
  const built=data.map(v=>{let pending;const getBlob=()=>pending||(pending=fetch(v.url,{cache:'force-cache'}).then(r=>{if(!r.ok)throw Error('图片读取失败HTTP '+r.status);return r.blob()}));return {name:v.name,type:'image/png',getBlob,arrayBuffer:async()=>await(await getBlob()).arrayBuffer()}});
  await loadFiles(built);
 }catch(e){editEnabled(false);importStatus('图像读取失败：'+e.message)}
}
''' +web[end:]
    web=web.replace('内置60张原色图像，打开自动显示','60张原色图像，打开自动显示')
    (a.site/'index.html').write_text(web);(a.site/'.nojekyll').write_text('')
    (a.output/'site_manifest.json').write_text(json.dumps(expected,indent=2)+'\n')
    print('HOSTED_PAGE_READY',len(web.encode()),'IMAGES',len(images))

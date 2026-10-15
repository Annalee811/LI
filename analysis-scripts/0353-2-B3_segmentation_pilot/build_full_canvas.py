#!/usr/bin/env python3
"""Build four separate full-RGB annotation views with exact legacy patch coordinates."""
from pathlib import Path
import argparse,base64,csv,hashlib,html,json
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--kit',type=Path,default=ROOT/'roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
rows=list(csv.DictReader((a.kit/'split_manifest.csv').open(encoding='utf-8-sig')))
source_hashes=json.loads((a.source/'SHA256.json').read_text())
images=[];arrays={}
for t in range(1,5):
    name=f'T{t}_multiregion_RGB.png';f=a.source/name;b=f.read_bytes();digest=hashlib.sha256(b).hexdigest()
    assert digest==source_hashes[name]
    im=Image.open(f).convert('RGB');assert im.size==(2541,1867)
    arrays[t]=np.asarray(im)
    images.append(dict(name=name,period=t,width=im.width,height=im.height,sha256=digest,base64=base64.b64encode(b).decode()))
for r in rows:
    r['full_name']=f'T{r["period"]}_multiregion_RGB.png'
    r['full_x']=int(r['T1_x'])-7
    r['full_y']=int(r['crop_y'])+(1132 if r['band']=='B' else 0)
    old=np.asarray(Image.open(a.kit/r['image_path']).convert('RGB'))
    x,y=r['full_x'],r['full_y'];assert np.array_equal(old,arrays[int(r['period'])][y:y+448,x:x+448]),r['image_id']
page=(HERE/'full_canvas_core.html').read_text()
for ident,data in [('full-images',images),('patch-mapping',rows)]:
    placeholder=f'<script id="{ident}" type="application/json">[]</script>'
    assert placeholder in page
    page=page.replace(placeholder,f'<script id="{ident}" type="application/json">'+json.dumps(data,separators=(',',':'))+'</script>')
(a.output/'annotate.html').write_text(page)
(a.output/'patch_mapping.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
(a.output/'source_images.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='base64'} for r in images],indent=2)+'\n')
recovery=(HERE/'preserve_open_editor.js').read_text()
(a.output/'preserve_open_editor.js').write_text(recovery)
helper='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>保全旧页面已有标注</title>
<style>body{font:17px/1.7 system-ui;max-width:900px;margin:30px auto;padding:20px;color:#203c30}textarea{width:100%;height:400px;font:13px/1.4 monospace}button{padding:12px;margin:10px;font:inherit}.note{background:#fff0cc;padding:16px}</style>
<h1>先保全旧页面的8张标注</h1><div class="note">不要关闭或刷新仍显示你标注结果的旧 annotate.html。这个助手不能隔着另一个网页读取笔画。需要把下方代码放进旧页面自己的控制台。</div>
<ol><li>点击“复制备份代码”。</li><li>切回仍有标注的旧页面，按 F12 打开开发者工具，选择 Console／控制台。</li><li>粘贴下方代码并按 Enter；旧页面右上角会出现绿色“保存全部已有标注JSON”按钮。</li><li>点击绿色按钮并确认JSON文件已下载。未保存成功前不要关闭旧页面。</li><li>打开新包的 annotate.html，导入这个JSON。它会把原笔画按坐标迁回四张完整RGB。</li></ol>
<p>代码只读取本页现有标注并在本机生成备份；不请求网络、不刷新页面、不清除或改写原有像素和分类。</p>
<button id="copy">复制备份代码</button><span id="status"></span><textarea id="code" readonly>CODE</textarea>
<script>document.getElementById('copy').onclick=async()=>{const t=document.getElementById('code');try{await navigator.clipboard.writeText(t.value);document.getElementById('status').textContent='已复制，切回旧页面执行。'}catch(e){t.focus();t.select();document.getElementById('status').textContent='请按Ctrl+C复制所选代码。'}};</script></html>'''
(a.output/'先保全旧标注.html').write_text(helper.replace('CODE',html.escape(recovery)))
report=dict(full_images=4,source_RGB_unchanged=True,full_image_wh=[2541,1867],old_patches_verified_pixel_exact=len(rows),
            old_patch_size=[448,448],A_offset_xy=[0,0],B_offset_from_old_region_xy=[-2,1132],nondata_separator_y=[1108,1132],
            actual_user_annotations_received=False,legacy_file_address_not_remotely_accessible=True)
(a.output/'migration_geometry_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print('FOUR_FULL_IMAGES_READY',len(page.encode()),'OLD_PATCHES_EXACT',len(rows))

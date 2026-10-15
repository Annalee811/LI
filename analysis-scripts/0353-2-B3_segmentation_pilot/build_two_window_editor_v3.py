from pathlib import Path
import base64,hashlib,json,re

import argparse
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/B3_two_groups_review');args=p.parse_args()
OUT=args.output;OUT.mkdir(parents=True,exist_ok=True)
KIT=ROOT/'roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation'
page=(KIT/'baseline-review-v2/annotate.html').read_text()
project=json.loads((OUT/'review_project.json').read_text())
images=[];hashes={}
for rec in project['records']:
    b=(OUT/'originals'/rec['name']).read_bytes();sha=hashlib.sha256(b).hexdigest();assert sha==rec['image_sha256']
    images.append(dict(name=rec['name'],base64=base64.b64encode(b).decode()));hashes[rec['name']]=sha
for ident,value in [('embedded-dataset',images),('expected-images',hashes),('embedded-prefills',project)]:
    page=re.sub(r'(<script id="'+ident+r'" type="application/json">).*?(</script>)',lambda m:m[1]+json.dumps(value,separators=(',',':'))+m[2],page,flags=re.S)
page=page.replace('根系标注 · 基线与物体分层版2','根系标注 · 两组四期复核版3')
page=page.replace('60张原色图与现有标注已内置','两组四期，共8张原色图与标注已内置').replace('恢复内置60张标注图','恢复内置8张图像')
page=page.replace('示范窗四期已重描；其余窗口仍待详细复核','本页仅包含 B3_A0_C0、B3_A0_C1 两组四期')
page=page.replace('T1统一显示已有根。示范窗按各期原图单独描画，未按亮纹自动吸附。其余56张只做了保守清理，不代表已完成详细补根。','T1全部可见根为基线。可对应的后期根保留编号；来源无法确认的根单独编号。未画处尚未确认背景，仍需逐图复核。')
page=page.replace('T1全部已有根；示范窗四期重描，其余窗口仍需详细复核。','T1全部为已有根；可直接切换两组四期并保存修改。')
page=page.replace('新生候选','本期首次可见').replace('首次观察／本期首次可见','本期首次可见（不等于新生）')
page=page.replace('T1：把全部可见根标为蓝色“未分类根”','T1：把全部可见根标为蓝色“已有基线根”')
page=page.replace('<span id="brushLabel">3</span>','<span id="brushLabel">1</span>').replace('max="15" value="3"','max="15" value="1"')
page=page.replace('<option value="1.5" selected>','<option value="1.5">').replace('<option value="2">2 倍','<option value="2" selected>2 倍')
page=page.replace('<label><input id="show" type="checkbox" checked> 显示标注覆盖层</label>', '''<label><input id="show" type="checkbox" checked> 显示标注覆盖层</label>
<label>覆盖层透明度 <input id="opacity" type="range" min="20" max="255" value="175"></label>
<label><input id="showUncertain" type="checkbox"> 显示不确定段（默认隐藏）</label>
<label>只高亮本窗根编号（0＝全部）<input id="focusRoot" type="number" min="0" max="255" value="0"></label>
<button id="trackRoot">查看本编号的四期对照</button>
<small>同一编号表示待核对的跨期对应。无可靠对应的段使用各期独立编号；未标出不代表死亡。这里只控制显示，不改变标注。</small>''')
page=page.replace("else if(v===128)rgb=sceneRGB[5];if(rgb){im.data.set(rgb,j);im.data[j+3]=layer==='scene'?120:175}","else if(v===128&&$('showUncertain').checked)rgb=[148,163,184];if(rgb){im.data.set(rgb,j);const focus=Number($('focusRoot').value)||0;im.data[j+3]=focus&&active.instance[i]!==focus?25:(layer==='scene'?120:Number($('opacity').value))}")
page=page.replace("$('show').onchange=render;","$('show').onchange=render;$('opacity').oninput=render;$('focusRoot').oninput=render;$('showUncertain').onchange=render;")
page=page.replace('</section></main>','''</section></main>
<section id="trackingSection" hidden style="background:white;margin:20px;padding:16px;border-radius:8px">
<h2 style="font-size:18px">同窗四期根编号对照</h2><div id="trackingNote"></div>
<div id="trackingGrid" style="display:flex;gap:12px;flex-wrap:wrap"></div></section>''')
tracking=r'''
let trackingToken=0;
$('trackRoot').onclick=async()=>{
 if(index<0||!active)return;remember();const token=++trackingToken,win=files[index].name.replace(/_T[1-4]\.png$/,''),rid=Math.max(0,Math.min(255,Number($('focusRoot').value)||0));
 $('trackingSection').hidden=false;$('trackingGrid').replaceChildren();$('trackingNote').textContent=win+(rid?' · 根编号 '+rid+'：其他根淡显。':' · 全部根。')+' 各图保持自身原始坐标；仅在本窗内比较编号，未画处不等于消失或死亡。';
 for(let t=1;t<=4;t++){
  const name=win+'_T'+t+'.png',f=files.find(f=>f.name===name),s=states.get(name),box=document.createElement('div');
  if(!f||!s)continue;const title=document.createElement('div');title.textContent='T'+t+' · '+(rid?([...s.instance].some((v,i)=>v===rid&&s.mask[i]===255)?'此编号已标出':'此编号未标出'):'全部可见标注');box.append(title);
  const c=document.createElement('canvas');c.width=W;c.height=W;c.style.width='min(320px,80vw)';c.style.height='auto';box.append(c);$('trackingGrid').append(box);
  const im=await decodeImage(f);if(token!==trackingToken){if(im.close)im.close();return}const ctx=c.getContext('2d');ctx.drawImage(im,0,0);if(im.close)im.close();
  const pixels=ctx.getImageData(0,0,W,W);for(let i=0;i<N;i++){if(s.mask[i]!==255)continue;const focus=!rid||s.instance[i]===rid,alpha=focus?.8:.12;const col=t===1?[37,99,235]:s.temporal[i]===3?[147,51,234]:s.temporal[i]===2?[234,179,8]:[37,99,235];for(let j=0;j<3;j++)pixels.data[i*4+j]=Math.round(pixels.data[i*4+j]*(1-alpha)+col[j]*alpha)}ctx.putImageData(pixels,0,0);
 }
};
'''
assert "let trackingToken" not in page
page=page.replace("window.addEventListener('beforeunload'",tracking+"\nwindow.addEventListener('beforeunload'")
page=page.replace('本期首次可见，同一根新延伸','本期首次可见（遮挡解除也可能显露），同一根新延伸')
assert '其余56张' not in page and '60张' not in page
(OUT/'annotate.html').write_text(page)
print('SELF_CONTAINED_EIGHT_IMAGE_EDITOR',len(images),len(page.encode()))

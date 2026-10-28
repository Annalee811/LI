"""Offline native-resolution review of every corrected or unresolved root interval."""
import base64
import csv
import io
import json
import pickle
import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path('/workspace/LI')
OUT = ROOT / 'results/B3_branch_continuity_review_v85_20261028'
sys.path.insert(0, str(ROOT / 'analysis-scripts/0353-2-B3_segmentation_pilot/root_axis_junction_v83'))
from rendering import isolated
from prepare import csvout


def read(p):
    with p.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def main():
    with (OUT / 'root_axes.pkl').open('rb') as f:
        axes = pickle.load(f)
    cases = []
    for status, filename in [('已修正根段', 'Root_interval_proposals.csv'), ('归属待复核', 'Unresolved_root_intervals.csv')]:
        rows = read(OUT / filename)
        for i, r in enumerate(rows):
            t = r['period']; axis = int(r['axis']); p = axes[t][axis]
            if 'x_start' in r:
                x0 = min(int(r['x_start']), int(r['x_end'])); x1 = max(int(r['x_start']), int(r['x_end'])) + 1
                y0 = min(int(r['y_start']), int(r['y_end'])); y1 = max(int(r['y_start']), int(r['y_end'])) + 1
            else:
                x0, y0 = p.min(0); x1, y1 = p.max(0) + 1
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            w, h = max(180, x1 - x0 + 70), max(180, y1 - y0 + 70)
            roi = [max(0, int(cx - w / 2)), max(0, int(cy - h / 2)), min(2541, int(cx + w / 2)), min(1108, int(cy + h / 2))]
            cases.append(dict(status=status, period=t, axis=axis, ROI=roi, reason=r['reason']))
    csvout(OUT / 'Review_cases_with_coordinates.csv', [dict(status=r['status'], period=r['period'], predicted_axis=r['axis'], x0=r['ROI'][0], y0=r['ROI'][1], x1=r['ROI'][2], y1=r['ROI'][3], reason=r['reason']) for r in cases])
    images = {}
    for t in ['T1', 'T2', 'T3', 'T4']:
        images[t] = {}
        for label, p in [('RGB', ROOT / 'results/B3_temporal_upload18_noGAT_v66_20261028/data/upper' / (t + '_RGB.png')),
                         ('修订参考', ROOT / 'results/B3_reference_local_correction_v84_20261028/corrected_reference' / (t + '_classes.png')),
                         ('上一版 V83', ROOT / 'results/B3_root_axis_junction_v83_20261028/final/full' / (t + '_classes.png')),
                         ('本轮 V85', OUT / 'full' / (t + '_classes.png'))]:
            if label == 'RGB':
                data = p.read_bytes()
            else:
                buffer = io.BytesIO(); Image.fromarray(isolated(np.array(Image.open(p)))).save(buffer, format='PNG'); data = buffer.getvalue()
            images[t][label] = 'data:image/png;base64,' + base64.b64encode(data).decode()
    html = '''<!doctype html><meta charset="utf-8"><title>四期根系逐段复核</title>
<style>body{font-family:Arial,sans-serif;margin:24px;color:#182532;background:#f7f9fb}h1{font-size:24px}select,input{font-size:16px;padding:8px;margin:6px}button{padding:8px}#panels{display:grid;grid-template-columns:repeat(2,1fr);gap:20px}section{background:white;padding:12px;overflow:auto;border:1px solid #ddd}canvas{image-rendering:pixelated;max-width:none}h2{font-size:18px}#note{line-height:1.7}.controls{position:sticky;top:0;background:#f7f9fb;padding:6px;z-index:2}</style>
<h1>四期根系逐段复核：上一版与本轮修正</h1><p>紫：持续；绿：新生；黄：延伸；蓝：T1基础。轮廓和模型权重保持不变。</p>
<div class="controls"><button id="prev">上一处</button><select id="cases"></select><button id="next">下一处</button><select id="period"><option>T1</option><option>T2</option><option>T3</option><option>T4</option></select><label>放大 <input id="zoom" type="range" min="1" max="6" step="0.5" value="2"></label></div>
<p id="note"></p><div id="panels"></div>
<script>
const cases=CASES_DATA;const files=IMAGES_DATA;const cache={};let generation=0;
const list=document.querySelector('#cases'); cases.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${i+1}. ${r.status} / ${r.period} / 预测轴 ${r.axis}`;list.append(o)});
async function draw(){const token=++generation,r=cases[+list.value],t=document.querySelector('#period').value,z=+document.querySelector('#zoom').value;document.querySelector('#note').textContent=`${r.status}；当前显示 ${t}；原问题位于 ${r.period}；坐标 ${r.ROI.join(', ')}。${r.reason}。预测轴编号用于本次审计，不替代人工根编号。`;const holder=document.querySelector('#panels');holder.replaceChildren();for(const label of ['RGB','修订参考','上一版 V83','本轮 V85']){const section=document.createElement('section'),heading=document.createElement('h2'),canvas=document.createElement('canvas');heading.textContent=label;section.append(heading,canvas);holder.append(section);const key=t+label;if(!cache[key]){const img=new Image;img.src=files[t][label];await img.decode();cache[key]=img}if(token!==generation)return;const [x,y,x1,y1]=r.ROI;canvas.width=x1-x;canvas.height=y1-y;canvas.style.width=(canvas.width*z)+'px';canvas.style.height=(canvas.height*z)+'px';const ctx=canvas.getContext('2d');ctx.imageSmoothingEnabled=false;ctx.drawImage(cache[key],x,y,x1-x,y1-y,0,0,canvas.width,canvas.height)}}
function select(){document.querySelector('#period').value=cases[+list.value].period;draw()}list.onchange=select;document.querySelector('#period').onchange=draw;document.querySelector('#zoom').oninput=draw;document.querySelector('#prev').onclick=()=>{list.value=Math.max(0,+list.value-1);select()};document.querySelector('#next').onclick=()=>{list.value=Math.min(cases.length-1,+list.value+1);select()};select();
</script>'''
    html = html.replace('CASES_DATA', json.dumps(cases, ensure_ascii=False)).replace('IMAGES_DATA', json.dumps(images, ensure_ascii=False))
    (OUT / 'Root_by_root_review_offline.html').write_text(html)
    print('Offline review cases', len(cases), 'bytes', len(html.encode()), flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Audit actual exported data, split geometry, checkpoint and reported test counts."""
from pathlib import Path
import argparse,csv,json,hashlib,importlib.util
import numpy as np
from PIL import Image
import torch

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();root=a.run
checks=[]
def check(name,ok):
    checks.append(dict(name=name,pass_check=bool(ok)))
    if not ok:raise AssertionError(name)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
rows=list(csv.DictReader((root/'dataset/manifest.csv').open(encoding='utf-8-sig')))
for r in rows:
    for k in ['column','x','y','width','height']:r[k]=int(r[k])
check('100 tile images with counts 60/20/20',len(rows)==100 and {s:sum(r['split']==s for r in rows) for s in ['train','val','test']}==dict(train=60,val=20,test=20))
windows={r['window_id'] for r in rows}
check('50 paired spatial windows; T1/T2 always same split',len(windows)==50 and all(len([r for r in rows if r['window_id']==w])==2 and {r['period'] for r in rows if r['window_id']==w}=={'T1','T2'} and len({r['split'] for r in rows if r['window_id']==w})==1 for w in windows))
minimum=float('inf')
for i,r in enumerate(rows):
    for q in rows[i+1:]:
        if r['split']==q['split']:continue
        dx=max(r['x']-q['x']-q['width'],q['x']-r['x']-r['width'],0)
        dy=max(r['y']-q['y']-q['height'],q['y']-r['y']-r['height'],0)
        minimum=min(minimum,max(dx,dy))
check('no cross-split pixel overlap; at least 64px guard',minimum>=64)
for t in ['T1','T2']:
    rgb=np.array(Image.open(root/f'dataset/upper/{t}_RGB.png'))
    labels={k:np.array(Image.open(root/f'dataset/upper/{t}_{k}.png')) for k in ['mask','color','temporal','viability']}
    check(t+' upper crop and label dimensions exactly 2541x1108',rgb.shape==(1108,2541,3) and all(v.shape==(1108,2541) for v in labels.values()))
    count=0
    for r in rows:
        if r['period']!=t:continue
        x,y=r['x'],r['y']
        assert sha(root/r['images_path'])==r['image_sha256'] and sha(root/r['mask_path'])==r['mask_sha256']
        assert np.array_equal(np.array(Image.open(root/r['images_path'])),rgb[y:y+256,x:x+256])
        for k in labels:
            assert np.array_equal(np.array(Image.open(root/r[k+'_path'])),labels[k][y:y+256,x:x+256])
        count+=1
    check(t+' all 50 RGB tiles and four label layers exactly match upper sources',count==50)
    check(t+' temporal/color/viability layers stay inside the root mask',all(not ((v>0)&(labels['mask']!=255)).any() for k,v in labels.items() if k!='mask'))

metrics_path=root/'evaluation/metrics.json'
if metrics_path.exists():
    result=json.loads(metrics_path.read_text());training=json.loads((root/'training/training_run.json').read_text())
    ck=torch.load(root/'training/best.pt',map_location='cpu',weights_only=True)
    check('checkpoint, manifest and chosen epoch agree',sha(root/'training/best.pt')==training['checkpoint_sha256'] and ck['manifest_sha256']==sha(root/'dataset/manifest.csv') and ck['epoch']==training['best_epoch'])
    thresholds=list(csv.DictReader((root/'training/validation_thresholds.csv').open(encoding='utf-8-sig')))
    chosen=max(thresholds,key=lambda r:(float(r['Dice']),-abs(float(r['threshold'])-.5)))
    check('threshold frozen from validation only',float(chosen['threshold'])==training['frozen_threshold'] and not training['test_access_for_model_selection'])
    for split,key,col in [('val','validation',3),('test','spatial_holdout_test',4)]:
        total=np.zeros(4,np.int64)
        for t in ['T1','T2']:
            gt=np.array(Image.open(root/f'predictions/{split}/{t}_C{col}_human_mask.png'))
            pr=np.array(Image.open(root/f'predictions/{split}/{t}_C{col}_prediction.png'))==255
            valid=gt!=128;g=gt==255
            total+=np.array([(g&pr&valid).sum(),(~g&pr&valid).sum(),(g&~pr&valid).sum(),(~g&~pr&valid).sum()],np.int64)
        m=result[key]['micro_metrics'];tp,fp,fn,tn=map(int,total)
        check(split+' scores exactly reproduce from saved prediction masks',all(int(m[k])==int(v) for k,v in zip(['TP','FP','FN','TN'],total)) and abs(m['Dice']-2*tp/max(1,2*tp+fp+fn))<1e-12)
        check(split+' overlapping patches counted only once per source pixel',sum(total)==2*448*1108 and result[key]['n_unique_evaluated_pixels']==2*448*1108)
    check('no fabricated manual-length or three-model comparison',result['manual_root_length_agreement'] is None and 'not been trained' in result['model_comparison_status'])
report=dict(passed=len(checks),failed=sum(not c['pass_check'] for c in checks),checks=checks,scope='Software, pixel integrity, geometry and actual training outputs; not expert adjudication of labels.')
(root/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(passed=report['passed'],failed=report['failed'])))

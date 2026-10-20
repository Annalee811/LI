"""Compare V14 and the manual-continuity model on identical full-field crops.

Published V14 tilewise scores are retained separately; do not compare a new
reference directly with the older teacher reference as if it were the same task.
"""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import common as c
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);a=p.parse_args();r=a.run;rows=c.rows(r/'data/paired_manifest.csv');tables=r/'tables';tables.mkdir(exist_ok=True);comparison=[];continuity=[];junction=[];published=[];details={}
refs={ref:{t:np.array(Image.open(r/'data'/folder/f'{t}_classes.png')) for t in c.PERIODS[1:]} for ref,folder in [('original','original'),('manual_morphology','revised'),('v14_teacher','v14_reference')]}
for m in c.MODELS:
 old=json.loads((a.previous/m/'metrics.json').read_text())
 for ref in ['revised','original']:published.append(dict(model=m,published_V14_reference=ref,macro_Dice=old['test'][ref]['end_to_end']['macro_Dice'],evaluation='Original V14 tilewise probabilities and chain decoder, fixed published values'))
 axes={t:c.load_axes(r/'axis_cache'/m/(t+'.npz')) for t in c.PERIODS[1:]}
 for version,folder in [('V14',a.previous),('manual_continuity_v17',r)]:
  maps={t:np.array(Image.open(folder/m/'four_period_prediction'/f'{t}_temporal_classes.png')) for t in c.PERIODS[1:]};local_junction={}
  for t,axis in axes.items():
   jm=np.zeros_like(maps[t],bool)
   for dec in axis['decisions']:
    if dec.get('degree',0)>=3:
     y,x=np.rint(dec['point']).astype(int);jm[max(0,y-1):y+2,max(0,x-1):x+2]=True
   jm=ndi.binary_dilation(jm,iterations=12);local_junction[t]=jm
   mixed=0;direct=0;eligible=0;short=0;switches=0;edgecount=0
   for ai,path in enumerate(axis['paths'],1):
    xy=axis['xy'][path];owned=axis['axis_map'][tuple(xy.T)]==ai;values=maps[t][tuple(xy.T)][owned]
    if len(values)<20:continue
    eligible+=1;mixed+=int(2 in values and 3 in values);direct+=int(np.sum(((values[:-1]==2)&(values[1:]==3))|((values[:-1]==3)&(values[1:]==2))));bounds=np.r_[0,np.flatnonzero(values[1:]!=values[:-1])+1,len(values)];short+=int((np.diff(bounds)<5).sum());switches+=len(bounds)-2;edgecount+=len(values)-1
   continuity.append(dict(model=m,version=version,period=t,scope='Full upper field, includes training pixels; same learned routes for before/after',eligible_axes_20_owned_nodes=eligible,axes_with_new_and_growth=mixed,direct_new_growth_switches=direct,class_switches=switches,switches_per_1000_edges=1000*switches/max(edgecount,1),runs_shorter_than5px=short))
  for split in ['val','test']:
   for ref,gtmaps in refs.items():
    cm=np.zeros((4,4),np.int64);cmj=np.zeros_like(cm)
    for row in rows:
     if row['split']!=split:continue
     x,y=int(row['x']),int(row['y']);t=row['current'];sl=np.s_[y:y+256,x:x+256];gt=gtmaps[t][sl];pred=maps[t][sl];cm+=c.old.confusion(gt,pred);gj=gt.copy();gj[~local_junction[t][sl]]=255;cmj+=c.old.confusion(gj,pred)
    z=c.old.metrics(cm);details[f'{m}/{version}/{split}/{ref}']=z;cl=z['classes'];comparison.append(dict(model=m,version=version,split=split,reference=ref,Precision=float(np.mean([q['Precision'] or 0 for q in cl])),Recall=float(np.mean([q['Recall'] or 0 for q in cl])),Dice=z['macro_Dice'],IoU=float(np.mean([q['IoU'] or 0 for q in cl])),scope='Identical 256px crops from saved full-frame maps; NOT original V14 tilewise evaluation'))
    zj=c.old.metrics(cmj);junction.append(dict(model=m,version=version,split=split,reference=ref,junction_macro_Dice=zj['macro_Dice'],typed_reference_root_pixels=sum(z['support_pixels'] for z in zj['classes']),evaluation='Root class agreement within12px junction neighborhoods; no instance identity ground truth'))
c.csvout(tables/'V14_matched_comparison.csv',comparison);c.csvout(tables/'V14_published_metrics.csv',published);c.csvout(tables/'root_continuity_before_after.csv',continuity);c.csvout(tables/'junction_neighborhood_metrics.csv',junction);c.js(tables/'matched_comparison_details.json',details)
summary=['# V14 baseline and manual-continuity update','','The first table uses identical held-out crops from saved full-field maps and the new manual-class/U-Net-morphology reference. Original published V14 teacher scores use another reference and tilewise inference, and are preserved separately.','','|Model|Version|Precision|Recall|Dice|IoU|','|---|---|---:|---:|---:|---:|']
for z in comparison:
 if z['split']=='test' and z['reference']=='manual_morphology':summary.append('|'+c.MODEL_NAMES[z['model']]+'|'+z['version']+'|'+'|'.join(f'{z[k]:.4f}' for k in ['Precision','Recall','Dice','IoU'])+'|')
summary+=['','Root continuity is evaluated on common putative axes, excluding shared nodes owned by another crossing axis. Zero mixed NEW/GROWTH axes is an enforced grammar constraint, not independent biological identity accuracy. Junction-neighborhood Dice measures temporal class agreement, not instance routing correctness. Same-position development data only; external independent test images: 0.']
(tables/'ACCURACY.md').write_text('\n'.join(summary)+'\n');print('\n'.join(summary),flush=True)

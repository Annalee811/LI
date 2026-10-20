"""Freeze the teacher geometry and evidence before selective label correction."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
import common as c
from identity_decode import decode
from predict import products
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);p.add_argument('--v14',type=Path,default=Path('results/B3_temporal_reference_v14'));a=p.parse_args();r=a.run;r.mkdir(parents=True,exist_ok=True);folder=r/'teacher_axis_cache';assert not folder.exists(),'Use a new run, never overwrite existing teacher evidence';folder.mkdir();router_file=a.previous/'routing/unet/router.json';router=json.loads(router_file.read_text());axis={};rgb={}
for i,t in enumerate(c.PERIODS):
 rgb[t]=np.array(Image.open(a.previous/'data/upper'/f'{t}_RGB.png'));mask=np.array(Image.open(a.roots/'unet/upper'/f'{t}_root_mask.png'))>0;axis[t]=c.axes.build(mask,rgb[t],axis.get(c.PERIODS[i-1]) if i else None,router=router);link=c.axes.match(axis[c.PERIODS[i-1]],axis[t],None,None) if i else None;c.axes.save(folder/f'{t}.npz',axis[t],link)
 if link:
  np.savez_compressed(folder/f'{t}_matches.npz',node_score=link['node_score'],nearest_previous_axis=link['nearest_previous_axis'],nearest_distance=link['nearest_distance']);teacher_file=a.previous/'unet/four_period_prediction'/f'{t}_class_probability.npz';q=np.load(teacher_file)['probability'].astype('float32');classes,records=decode(q,axis[c.PERIODS[i-1]],axis[t],link,1.);products(r/'teacher_identity_preview',t,rgb[t],mask,classes,q);c.js(r/'teacher_identity_preview'/f'{t}_root_axes.json',records)
for m in c.MODELS:
 src=(a.previous if m=='unet' else a.v14)/m/'best_temporal_head.pt';dest=r/'warm_starts'/m;dest.mkdir(parents=True);shutil.copy2(src,dest/src.name)
c.js(r/'teacher_evidence_provenance.json',dict(teacher_model='Delivered V17 U-Net class scores, not V18 student predictions',router_SHA256=c.sha(router_file),teacher_head_SHA256=c.sha(a.previous/'unet/best_temporal_head.pt'),probabilities_SHA256={t:c.sha(a.previous/'unet/four_period_prediction'/f'{t}_class_probability.npz') for t in c.PERIODS[1:]},axes_source_SHA256=c.sha(Path(__file__).with_name('axes.py')),stitch_source_SHA256=c.sha(Path(__file__).with_name('stitch.py')),decoder_source_SHA256=c.sha(Path(__file__).with_name('identity_decode.py')),reference_not_independent_manual_truth=True))

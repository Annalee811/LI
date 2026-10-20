"""Learned junction tracing before crops; no human labels enter inference features."""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import common as c
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--model',choices=c.MODELS,required=True);a=p.parse_args();name=a.model;axis={};r=a.run;folder=r/'axis_cache'/name;folder.mkdir(parents=True,exist_ok=True);features=r/'feature_cache'/name;features.mkdir(parents=True,exist_ok=True);router=json.loads((r/'routing'/name/'router.json').read_text());rgb={t:np.array(Image.open(r/'data/upper'/f'{t}_RGB.png')) for t in c.PERIODS}
for i,t in enumerate(c.PERIODS):
 mask=np.array(Image.open(a.roots/name/'upper'/f'{t}_root_mask.png'))>0;axis[t]=c.axes.build(mask,rgb[t],axis.get(c.PERIODS[i-1]) if i else None,router=router);c.axes.save(folder/(t+'.npz'),axis[t]);print('ROUTED_AXES',name,t,len(axis[t]['paths']),flush=True)
for prev,t in zip(c.PERIODS,c.PERIODS[1:]):
 link=c.axes.match(axis[prev],axis[t],rgb[prev],rgb[t]);c.axes.save(folder/(t+'.npz'),axis[t],link);np.savez_compressed(folder/(t+'_matches.npz'),node_score=link['node_score'],nearest_previous_axis=link['nearest_previous_axis'],nearest_distance=link['nearest_distance']);base=a.previous/'feature_cache'/name/(t+'.npy')
 if base.exists():old=np.load(base,mmap_mode='r')[:,:,:25]
 else:
  pp,cp=[np.load(a.roots/name/'upper'/f'{s}_probability.npy') for s in [prev,t]];pg,cg=[np.array(Image.open(a.roots/name/'upper'/f'{s}_root_mask.png'))>0 for s in [prev,t]];old=c.old.features(rgb[prev],rgb[t],pp,cp,pg,cg)
 extra=np.concatenate([link['context'],c.axes.root_contrast(rgb[t])[...,None],c.axes.root_contrast(rgb[prev])[...,None]],2);value=np.concatenate([old,extra],2).astype('float16');assert value.shape==(1108,2541,31) and np.isfinite(value).all();np.save(features/(t+'.npy'),value);c.js(features/(t+'_provenance.json'),dict(manual_labels_read=False,router_SHA256=c.sha(r/'routing'/name/'router.json'),axes_source_SHA256=c.sha(Path(__file__).with_name('axes.py')),RGB_SHA256={s:c.sha(r/'data/upper'/f'{s}_RGB.png') for s in [prev,t]},feature_source='25 V14 prediction-only channels plus 4 whole-axis-context and 2 rawRGB local-contrast channels',dtype='float16'));del value,extra
 print('FEATURES_READY',name,t,flush=True)

"""Trace full RGB/root fields before crops. Store prediction-only axis context."""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import common as c
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--model',choices=c.MODELS);a=p.parse_args()
for model in [a.model] if a.model else c.MODELS:
 start=time.monotonic();rgb={t:np.array(Image.open(a.run/'data/upper'/f'{t}_RGB.png')) for t in c.PERIODS};cache=a.run/'axis_cache'/model;cache.mkdir(parents=True,exist_ok=True);features=a.run/'feature_cache'/model;features.mkdir(parents=True,exist_ok=True);axis={};contrasts={t:c.axes.root_contrast(rgb[t]) for t in c.PERIODS}
 for t in c.PERIODS:
  gate=np.array(Image.open(a.roots/model/'upper'/f'{t}_root_mask.png'))>0;path=cache/(t+'.npz')
  if path.exists():axis[t]=c.load_axes(path)
  else:axis[t]=c.axes.build(gate,rgb[t],axis.get(c.PERIODS[c.PERIODS.index(t)-1]) if t!='T1' else None);c.axes.save(path,axis[t])
  print('AXES',model,t,'axes',len(axis[t]['paths']),'nodes',len(axis[t]['xy']),'seconds',round(time.monotonic()-start,1),flush=True)
 for prev,t in zip(c.PERIODS,c.PERIODS[1:]):
  link=c.axes.match(axis[prev],axis[t],rgb[prev],rgb[t]);c.axes.save(cache/(t+'.npz'),axis[t],link);np.savez_compressed(cache/(t+'_matches.npz'),node_score=link['node_score'],nearest_previous_axis=link['nearest_previous_axis'],nearest_distance=link['nearest_distance'])
  baseline=a.previous/'feature_cache'/model/(t+'.npy')
  if baseline.exists():old=np.load(baseline,mmap_mode='r');base_pin=c.sha(baseline)
  else:
   prob={s:np.load(a.roots/model/'upper'/f'{s}_probability.npy') for s in [prev,t]};gates={s:np.array(Image.open(a.roots/model/'upper'/f'{s}_root_mask.png'))>0 for s in [prev,t]};old=c.old.features(rgb[prev],rgb[t],prob[prev],prob[t],gates[prev],gates[t]);base_pin='Recomputed from immutable RGB/probabilities with temporal_v13.common.features'
  assert old.shape[2]==25;extra=np.concatenate([link['context'],contrasts[t][...,None],contrasts[prev][...,None]],2).astype('float32');feat=np.concatenate([old,extra],2).astype('float16');assert feat.shape==(1108,2541,31) and np.isfinite(feat).all();np.save(features/(t+'.npy'),feat)
  c.js(features/(t+'_provenance.json'),dict(RGB_SHA256={s:c.sha(a.run/'data/upper'/f'{s}_RGB.png') for s in [prev,t]},root_mask_SHA256={s:c.sha(a.roots/model/'upper'/f'{s}_root_mask.png') for s in [prev,t]},base_25ch_features_SHA256=base_pin,axis_source_SHA256=c.sha(Path(__file__).with_name('axes.py')),manual_labels_read=False,whole_field_traced_before_cropping=True,channels31='25 inherited prediction-only RGB/geometry +4 whole-axis match/position maps +current/previous rawRGB local contrast',dtype='float16',root_stage_frozen=True));del feat,extra
  print('FEATURES',model,t,'seconds',round(time.monotonic()-start,1),flush=True)
 print('CACHE_DONE',model,flush=True)

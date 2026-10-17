#!/usr/bin/env python3
"""Validation-only component denoising and model/inference selection, never C4."""
import argparse,json
from pathlib import Path
import numpy as np
import protocol as p

def load(folder):return {t:np.load(folder/(t+'_probability.npy')) for t in p.PERIODS}
def evaluate(data,maps,cfg,col=3):
 cc=np.zeros(4,np.int64);per={};x=p.COLS[col];removed=0
 for t in p.PERIODS:
  mask,rm=p.clean(maps[t],cfg,True);removed+=int(rm.sum());r=p.counts(data.gt[t][:,x:x+448],mask);cc+=r;per[t]=p.metrics(r)
 return dict(root=p.metrics(cc),per_period=per,removed_pixels=removed)

def main():
 a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True);a.add_argument('--run',type=Path,required=True);a.add_argument('--baselines-only',action='store_true');args=a.parse_args()
 data=p.Data(args.data);registry=json.loads((args.run/'baseline_registry.json').read_text())
 if not args.baselines_only:
  trained=args.run/'unet_pretrained';run=json.loads((trained/'training_run.json').read_text())
  assert p.sha(trained/'best.pt')==run['checkpoint_sha256']
  for tta,folder in [(False,'val_identity'),(True,'val_tta')]:
   name='refined_unet'+('_tta' if tta else '');maps=load(trained/folder);p.save_maps(args.run/'candidates'/name/'val',maps)
   registry[name]=dict(kind='refined',tta=tta,checkpoint_sha256=run['checkpoint_sha256'],context=384)
  # Fixed equal-weight fusion of two U-Net architectures; weights never fitted to test.
  name='refined_legacy_mean';maps={t:(np.load(args.run/'candidates/refined_unet_tta/val'/(t+'_probability.npy'))+np.load(args.run/'candidates/unet_context384_tta/val'/(t+'_probability.npy')))*.5 for t in p.PERIODS}
  p.save_maps(args.run/'candidates'/name/'val',maps);registry[name]=dict(kind='mean',components=['refined_unet_tta','unet_context384_tta'],weights=[.5,.5])
 results={}
 for name,record in registry.items():
  cache=args.run/'candidates'/name/'validation_cleanup.json'
  if cache.exists():results[name]=json.loads(cache.read_text());continue
  maps=load(args.run/'candidates'/name/'val');raw,_=p.calibrate(data,maps)
  cfg=dict(threshold=raw['threshold'],hysteresis_delta=0.,compact_area=0)
  baseline=evaluate(data,maps,cfg);best=dict(configuration=cfg,metrics=baseline);trials=[]
  for high in sorted(set(round(float(np.clip(raw['threshold']+off,.05,.95)),3) for off in [-.025,0,.025])):
   for delta in [0.,.075,.15]:
    for area in [0,8,20]:
     config=dict(threshold=high,hysteresis_delta=delta,compact_area=area)
     score=evaluate(data,maps,config)
     admissible=score['root']['Recall']>=baseline['root']['Recall']-.005 and all(score['per_period'][t]['Recall']>=baseline['per_period'][t]['Recall']-.01 for t in p.PERIODS)
     trials.append(dict(**config,admissible=admissible,**score['root']))
     if admissible and score['root']['Dice']>best['metrics']['root']['Dice']+1e-8:best=dict(configuration=config,metrics=score)
  result=dict(candidate=record,raw=baseline,raw_configuration=cfg,selected=best,selection='C3 strict Dice with pooled Recall loss<=0.005 and every-period Recall loss<=0.01; no C4 scores',labels_changed=False)
  p.csvout(args.run/'candidates'/name/'validation_cleanup_grid.csv',trials);p.js(cache,result);results[name]=result
  print('VALIDATION_CLEANUP',name,'RAW',round(baseline['root']['Dice'],4),'CLEAN',round(best['metrics']['root']['Dice'],4),flush=True)
 if not args.baselines_only:
  winner=max(results,key=lambda n:(results[n]['selected']['metrics']['root']['Dice'],results[n]['selected']['metrics']['root']['Recall']))
  p.js(args.run/'selection.json',dict(winner=winner,candidates=results,registry=registry,
   selection_uses_C4=False,test_status='C4 reused from earlier rounds; repeated same-position benchmark, not fresh independent test',
   death_and_temporal_tasks=False))
  print('V6_FROZEN_WINNER',winner,results[winner]['selected']['metrics']['root'],flush=True)
if __name__=='__main__':main()

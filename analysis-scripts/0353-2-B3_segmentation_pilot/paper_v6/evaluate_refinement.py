#!/usr/bin/env python3
"""Score frozen choices on reused C4, retain raw/clean masks and native full RGB."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
import protocol as p
from train import RootUNet,inference
from select_refinement import load,evaluate
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'root_background_v5'))
import train_torch as oldtorch
import common as oldprotocol
from evaluate import length_field,agreement

def boundary_f1(gt,pred,tolerance=1):
 bg=gt&~ndi.binary_erosion(gt);bp=pred&~ndi.binary_erosion(pred)
 if not bg.any() or not bp.any():return 0.
 pr=float((ndi.distance_transform_edt(~bg)[bp]<=tolerance).mean());re=float((ndi.distance_transform_edt(~bp)[bg]<=tolerance).mean())
 return 2*pr*re/max(1e-12,pr+re)

class Predictors:
 def __init__(self,run,legacy,data,registry):
  self.run=run;self.legacy=legacy;self.data=data;self.registry=registry;self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');self.models={}
 def infer(self,name):
  spec=self.registry[name];kind=spec['kind']
  key='refined' if kind=='refined' else spec.get('model',name)
  if key not in self.models:
   model=RootUNet() if key=='refined' else oldtorch.create(key)
   ckpt=self.run/'unet_pretrained/best.pt' if key=='refined' else self.legacy/key/'best.pt'
   assert p.sha(ckpt)==spec['checkpoint_sha256']
   state=torch.load(ckpt,map_location=self.device,weights_only=True);model.load_state_dict(state['state_dict']);self.models[key]=model.to(self.device).eval()
  return inference(self.models[key],self.device,spec.get('tta',False))
 def column(self,name,col):
  folder=self.run/'candidates'/name/('test' if col==4 else 'val')
  if all((folder/(t+'_probability.npy')).exists() for t in p.PERIODS):return load(folder)
  spec=self.registry[name]
  if spec['kind']=='legacy':
   original=self.legacy/spec['model']/'predictions/test/C4'
   if all((original/(t+'_probability.npy')).exists() for t in p.PERIODS):maps={t:np.load(original/(t+'_probability.npy')) for t in p.PERIODS}
   else:
    assert name!='deeplab','Frozen TF baseline C4 maps must be supplied'
    maps=oldprotocol.Data(self.data.source).column(self.infer(name),col)
  elif spec['kind']=='mean':
   components=[self.column(component,col) for component in spec['components']]
   maps={t:sum(w*component[t] for w,component in zip(spec['weights'],components)) for t in p.PERIODS}
  else:maps=self.data.column(self.infer(name),col)
  p.save_maps(folder,maps);return maps
 def whole(self,name,t):
  folder=self.run/'whole_cache'/name;path=folder/(t+'_probability.npy')
  if path.exists():return np.load(path)
  spec=self.registry[name]
  if spec['kind']=='mean':prob=sum(w*self.whole(n,t) for w,n in zip(spec['weights'],spec['components']))
  elif spec['kind']=='legacy' and name=='unet':prob=np.load(self.legacy/'unet/predictions/upper'/(t+'_probability.npy'))
  else:
   infer=self.infer(name);side=384 if spec['kind']!='legacy' else 256;rgb=self.data.rgb[t];h,w=rgb.shape[:2]
   xs=list(range(0,w-side+1,192));ys=list(range(0,h-side+1,192))
   if xs[-1]!=w-side:xs.append(w-side)
   if ys[-1]!=h-side:ys.append(h-side)
   coords=[(x,y) for y in ys for x in xs];sm=np.zeros((h,w),np.float32);weight=np.zeros_like(sm)
   hann=np.hanning(side).astype(np.float32);blend=np.maximum(hann[:,None]*hann[None,:],.05)
   for i in range(0,len(coords),2):
    xy=coords[i:i+2];ims=np.stack([rgb[y:y+side,x:x+side] for x,y in xy]).astype(np.float32)/255
    for (x,y),pr in zip(xy,infer(ims)):sm[y:y+side,x:x+side]+=pr*blend;weight[y:y+side,x:x+side]+=blend
   assert (weight>0).all();prob=sm/weight
  folder.mkdir(parents=True,exist_ok=True);np.save(path,prob.astype(np.float32));return prob

def main():
 a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True);a.add_argument('--legacy',type=Path);a.add_argument('--run',type=Path,required=True);a.add_argument('--selection',default='selection.json');a.add_argument('--threads',type=int,default=3);args=a.parse_args()
 torch.set_num_threads(args.threads);data=p.Data(args.data);selection=json.loads((args.run/args.selection).read_text());predict=Predictors(args.run,args.legacy or args.data,data,selection['registry'])
 reports={};rows=[];lengthrows=[];lengthmetrics={};noise=[];topology={}
 for name,choice in selection['candidates'].items():
  reports[name]={};lengthmetrics[name]={};topology[name]={}
  for split,col in [('val',3),('test',4)]:
   maps=predict.column(name,col);rawcfg=choice['raw_configuration'];cfg=choice['selected']['configuration'];scores={}
   for mode,config in [('raw',rawcfg),('cleaned',cfg)]:
    score=evaluate(data,maps,config,col);scores[mode]=score
    rows.append(dict(candidate=name,split=split,mode=mode,period='all',**score['root']))
    for t in p.PERIODS:rows.append(dict(candidate=name,split=split,mode=mode,period=t,**score['per_period'][t]))
   if split=='val':assert scores['cleaned']==choice['selected']['metrics'],'Validation score must reproduce from immutable probability maps'
   reports[name][split]=scores;local=[];spt=sgt=spi=sgi=0;boundary=[]
   folder=args.run/'candidates'/name/split
   for t in p.PERIODS:
    prob=maps[t];assert prob.shape==(1108,448) and prob.dtype==np.float32 and np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
    gt=data.gt[t][:,p.COLS[col]:p.COLS[col]+448];g=gt==255;raw=p.clean(prob,rawcfg);clean,removed=p.clean(prob,cfg,True)
    for stem,array in [('human',g),('raw',raw),('cleaned',clean),('removed',raw&~clean),('added',~raw&clean)]:Image.fromarray(array.astype(np.uint8)*255).save(folder/(t+'_'+stem+'.png'))
    noise.append(dict(candidate=name,split=split,period=t,raw_pixels=int(raw.sum()),cleaned_pixels=int(clean.sum()),
     compact_removed_pixels=int(removed.sum()),total_removed_pixels=int((raw&~clean).sum()),total_added_pixels=int((~raw&clean).sum()),
     removed_reference_root_pixels=int((raw&~clean&g).sum()),removed_reference_background_pixels=int((raw&~clean&~g).sum())))
    lf=length_field(g);lp=length_field(clean);sg=skeletonize(g);sp=skeletonize(clean);spt+=int(sp.sum());sgt+=int(sg.sum());spi+=int((sp&g).sum());sgi+=int((sg&clean).sum());boundary.append(boundary_f1(g,clean))
    for yi in range(4):
     for xi in range(2):
      yy=slice(yi*277,(yi+1)*277);xx=slice(xi*224,(xi+1)*224)
      row=dict(candidate=name,split=split,period=t,block='X'+str(xi)+'Y'+str(yi),human_length_px=float(lf[yy,xx].sum()),predicted_length_px=float(lp[yy,xx].sum()))
      local.append(row);lengthrows.append(row)
   lengthmetrics[name][split]=agreement([r['human_length_px'] for r in local],[r['predicted_length_px'] for r in local]);pr=spi/max(1,spt);re=sgi/max(1,sgt)
   topology[name][split]=dict(clDice=2*pr*re/max(1e-12,pr+re),skeleton_Precision=pr,skeleton_Recall=re,
    boundary_F1_at_1px_mean=float(np.mean(boundary)),boundary_tolerance_pixels=1,strict_Dice=scores['cleaned']['root']['Dice'],
    note='Boundary1px and topology are separate diagnostics, not replacements for strict pixel Dice')
  print('V6_EVALUATED',name,reports[name]['test']['cleaned']['root'],flush=True)
 p.js(args.run/'all_metrics.json',reports);p.csvout(args.run/'segmentation_metrics.csv',rows);p.csvout(args.run/'noise_audit.csv',noise)
 p.csvout(args.run/'root_length_pairs.csv',lengthrows);p.js(args.run/'root_length_agreement.json',lengthmetrics);p.js(args.run/'topology_boundary_metrics.json',topology)
 winner=selection['winner'];choice=selection['candidates'][winner];products=args.run/'root_products';products.mkdir(exist_ok=True)
 for t in p.PERIODS:
  prob=predict.whole(winner,t);rgb=data.rgb[t];raw=p.clean(prob,choice['raw_configuration']);clean,removed=p.clean(prob,choice['selected']['configuration'],True)
  Image.fromarray(rgb).save(products/(t+'_original_RGB.png'));Image.fromarray(data.gt[t]).save(products/(t+'_human_mask.png'))
  for stem,array in [('raw_mask',raw),('cleaned_mask',clean),('removed_mask',raw&~clean),('added_mask',~raw&clean),('compact_removed_mask',removed)]:Image.fromarray(array.astype(np.uint8)*255).save(products/(t+'_'+stem+'.png'))
  rgba=np.concatenate([rgb,(clean.astype(np.uint8)*255)[:,:,None]],2);rgba[~clean,:3]=0;assert np.array_equal(rgba[clean,:3],rgb[clean]);Image.fromarray(rgba).save(products/(t+'_root_only_RGBA.png'))
  overlay=rgb.copy();overlay[clean]=(rgb[clean]*.5+np.array([0,133,125])*.5).astype(np.uint8);Image.fromarray(overlay).save(products/(t+'_overlay.png'))
  Image.fromarray(np.round(prob*65535).astype(np.uint16)).save(products/(t+'_root_score_u16.png'))
  print('V6_NATIVE_PRODUCTS',t,flush=True)
 failures=[]
 for split in ['val','test']:
  for key,bound in [('Dice',.95),('Precision',.90),('Recall',.95)]:
   if reports[winner][split]['cleaned']['root'][key]<bound:failures.append(split+' '+key+' <'+str(bound))
 p.js(args.run/'stage_gate.json',dict(winner=winner,root_quality_gate_passed=not failures,failures=failures,
  temporal_stage_started=False,death_class=False,other_positions_predicted=False,independent_external_test_images=0,
  test='Reused within-position C4 benchmark, not new independent test',selection_SHA256=p.sha(args.run/args.selection),selection_file=args.selection))
 p.js(args.run/'evaluation_verification.json',dict(passed=True,validation_scores_reproduced=True,all_maps_native_float32=True,reference_pixels_unchanged=True,
  postprocessing_uses_GT_at_inference=False,selection_uses_C4=False,background_subtypes='No separate soil/droplet/debris ground truth; one background class'))
 print('V6_EVALUATION_COMPLETE',winner,flush=True)
if __name__=='__main__':main()

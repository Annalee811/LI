"""C3-only pair-confusion calibration; all3 pairs guarded at junctions and tips."""
import argparse,itertools,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch
import common as c
from model import warm_start
from predict import load_head
from train import evaluate
from junctions import junction_region,topology
PAIRS=[(1,2),(1,3),(2,3)]

def scores(data,maps,col,scales=None,old=None):
 cms=[np.zeros((4,4),np.int64) for _ in range(3)]
 for prev,t in zip(c.PERIODS,c.PERIODS[1:]):
  gate=data.gates[t,col];gt=data.y[t][:,c.COLS[col]:c.COLS[col]+448]
  pred=old[t] if old is not None else np.where(gate,(maps[t]*np.array(scales,np.float32)[:,None,None]).argmax(0)+1,0).astype(np.uint8)
  _,tips,_=topology(data.gates[prev,col]);tr=ndi.distance_transform_edt(~tips)<=20 if tips.any() else np.zeros(gate.shape,bool)
  for i,region in enumerate([np.ones(gate.shape,bool),junction_region(gate),tr]):
   gg=gt.copy();gg[~region]=255;cms[i]+=c.confusion(gg,pred)
 counts=[int(cm[x,y]+cm[y,x]) for cm in cms[1:] for x,y in PAIRS]
 return c.metrics(cms[0]),c.metrics(cms[1]),counts

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);p.add_argument('--previous',type=Path);a=p.parse_args();previous=a.previous or a.run.with_name('B3_temporal_v10');torch.set_num_threads(3);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 for n in c.MODELS:
  out=a.run/n
  if (out/'pair_calibration.json').exists():print('PAIR_CALIBRATION_ALREADY_FROZEN',n,flush=True);continue
  data=c.Data(a.run/'data',a.roots,n,[3,4]);old={t:np.array(Image.open(previous/n/'evaluation/validation'/f'{t}_classes.png')) for t in c.PERIODS[1:]};baseline,bj,bc=scores(data,None,3,old=old)
  current_epoch=torch.load(out/'evaluation_head.pt',map_location='cpu',weights_only=True)['epoch']
  candidates=[('fine_tuned' if current_epoch else 'epoch0',load_head(out/'evaluation_head.pt',device))]
  if torch.load(out/'evaluation_head.pt',map_location='cpu',weights_only=True)['epoch']!=0:candidates.append(('epoch0',warm_start(previous/n/'evaluation_head.pt',device)))
  rows=[];best=None;best_model=None
  for label,m in candidates:
   maps={t:c.predict(m,data.cache[t,3],device) for t in c.PERIODS[1:]}
   for ns,gs in itertools.product([.75,.85,.9,.95,1.,1.05,1.1,1.15,1.25],[.5,.65,.8,.9,1.,1.1,1.25]):
    metric,junction,counts=scores(data,maps,3,[1.,ns,gs]);ok=metric['macro_Dice']>=baseline['macro_Dice']-.005 and all(r['Dice']>=b['Dice']-.005 for r,b in zip(metric['classes'],baseline['classes'])) and junction['macro_Dice']>=bj['macro_Dice']-.005 and all(v<=b for v,b in zip(counts,bc))
    r=dict(candidate=label,new_scale=ns,growth_scale=gs,global_macro_Dice=metric['macro_Dice'],junction_macro_Dice=junction['macro_Dice'],junction_persistent_new=counts[0],junction_persistent_growth=counts[1],junction_new_growth=counts[2],tip_persistent_new=counts[3],tip_persistent_growth=counts[4],tip_new_growth=counts[5],eligible=ok)
    rows.append(r);key=(-sum(counts),metric['macro_Dice'],-abs(np.log(ns))-abs(np.log(gs)))
    if ok and (best is None or key>best[0]):best=(key,r);best_model=m
  assert best is not None,'Epoch0 with1/1 calibration must remain a feasible baseline'
  selected=best[1];scales=[1.,selected['new_scale'],selected['growth_scale']]
  for name in ['evaluation_head.pt','final_head.pt']:shutil.copyfile(out/name,out/('pre_pair_'+name))
  for name in ['evaluation_head.pt','final_head.pt']:
   if selected['candidate']=='epoch0':
    m=warm_start(previous/n/name,device);value=dict(state_dict=m.state_dict(),epoch=0,class_multipliers=m.class_multipliers,config=json.loads((out/'config.json').read_text()),deployment='Epoch0 retained; later trained trial rejected by pair-confusion guards')
   else:value=torch.load(out/name,map_location='cpu',weights_only=True)
   value['class_adjustment']=scales;torch.save(value,out/name)
  model=load_head(out/'evaluation_head.pt',device);reports={s:evaluate(model,data,col,device,out/'evaluation'/s) for s,col in [('validation',3),('repeated_benchmark',4)]};c.js(out/'evaluation_metrics.json',reports)
  c.csvout(out/'pair_calibration_grid.csv',rows)
  record=dict(selected=selected,adjustment=scales,baseline_junction_and_tip_pair_counts=bc,baseline_macro_Dice=baseline['macro_Dice'],selection='C3 only; every class/global Dice decline<=0.005; all3 pair confusions at BOTH junctions and previous tips must not increase; minimize sum of these6 counts then maximize macro Dice',C4_used_for_selection=False,manual_annotation_changed=False,grid_per_candidate=63,old_epoch0_allowed=True)
  c.js(out/'pair_calibration.json',record);c.js(out/'calibration.json',record)
  run=json.loads((out/'training_run.json').read_text());run.update(pair_calibration=record,evaluation_head_sha256=c.sha(out/'evaluation_head.pt'),final_head_sha256=c.sha(out/'final_head.pt'));c.js(out/'training_run.json',run)
  if not (out/'final_history.csv').exists():(out/'final_history.csv').write_text('epoch,loss\n')
  print('PAIR_CALIBRATED',n,selected,flush=True)
if __name__=='__main__':main()

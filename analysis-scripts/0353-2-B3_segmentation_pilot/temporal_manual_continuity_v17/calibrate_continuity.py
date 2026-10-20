"""Validation-only V14/new-head blending, class ratios, and axis decoder selection."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import common as c
from predict import evaluate,products
from continuity import decode
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--model',choices=c.MODELS,required=True);a=p.parse_args();r=a.run;m=a.model;folder=r/m;full=folder/'four_period_prediction';data=c.Data(r,a.roots,m);torch.set_num_threads(3);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');warm=c.load_head(a.previous/m/'best_temporal_head.pt',device)
raw_metrics=folder/'raw_pixel_test_metrics.json'
if raw_metrics.exists():raw_metrics.rename(folder/'pre_calibration_raw_pixel_metrics.json')
backup=r/'development_before_continuity_calibration'/m;backup.mkdir(parents=True,exist_ok=True)
for n in ['metrics.json','decoder_selection.json']:shutil.copy2(folder/n,backup/n)
prob={t:np.load(full/f'{t}_raw_probability.npz')['probability'].astype('float32') for t in c.PERIODS[1:]};base={}
for t in c.PERIODS[1:]:
 file=full/f'{t}_V14_probability.npz'
 if file.exists():base[t]=np.load(file)['probability'].astype('float32')
 else:
  base[t]=c.predict_probability(warm,data.features[t],device).astype('float16').astype('float32');np.savez_compressed(file,probability=base[t].astype('float16'))
candidates=[];best=-1;selected=None;bestmaps=None;bestrecords=None;bestprob=None
for alpha in [0.,.5,1.]:
 for growth in [1.,2.,4.]:
  for new in [1.,.75]:
   calibrated={}
   for t in prob:
    q=alpha*prob[t]+(1-alpha)*base[t];q=q*np.array([1,new,growth],np.float32)[:,None,None];q=q/q.sum(0,keepdims=True);calibrated[t]=q
   for mode,strength in [('learned',0.),('learned',.5),('strict',0.),('strict',1.)]:
    maps={};records={}
    for prev,t in zip(c.PERIODS,c.PERIODS[1:]):maps[t],records[t]=decode(calibrated[t],data.axes[prev],data.axes[t],data.link[t],strength,mode)
    val=evaluate(data,maps,'val');score=val['revised']['end_to_end']['macro_Dice'];row=dict(new_head_weight=alpha,growth_factor=growth,new_factor=new,decoder=mode,strength=strength,validation_manual_morphology_macro_Dice=score);candidates.append(row)
    if score>best+1e-8:best=score;selected=row;bestmaps=maps;bestrecords=records;bestprob=calibrated
  print('CALIBRATION',m,alpha,growth,'best',round(best,4),flush=True)
selection=dict(**selected,selected_strength=selected['strength'],selected_on='15 validation date pairs, manual temporal classes / U-Net morphology only',test_used_for_selection=False,reference_semantics_replaced=False,candidates=candidates,V14_head_SHA256=c.sha(a.previous/m/'best_temporal_head.pt'),new_head_SHA256=c.sha(folder/'best_temporal_head.pt'),confidence='Validation-rescaled blended class score; not calibrated structured posterior',class_ratios_selected_on_validation=True)
c.js(folder/'decoder_selection.json',selection)
for prev,t in zip(c.PERIODS,c.PERIODS[1:]):
 products(full,t,data.rgb[t],data.gate[t],bestmaps[t],bestprob[t]);c.js(full/f'{t}_root_axes.json',bestrecords[t])
reports={s:evaluate(data,bestmaps,s,folder/'evaluation'/s) for s in ['val','test']};c.js(folder/'metrics.json',reports);meta=json.loads((full/'inference.json').read_text());meta.update(decoder_selection=selection,post_fit_calibration_on_validation_only=True,manual_labels_used_as_model_inputs=False);c.js(full/'inference.json',meta)
print('CONTINUITY_CALIBRATED',m,selected,'TEST',reports['test']['revised']['end_to_end']['macro_Dice'],flush=True)

"""Predict full-frame roots first; validate the continuity decoder without test labels."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import common as c
from identity_decode import decode

def evaluate(data,maps,split):
    cm=np.zeros((4,4),np.int64);per=[]
    for row in [r for r in data.pairs if r['split']==split]:
        x,y=int(row['x']),int(row['y']);t=row['current'];gt=data.targets['current'][t][y:y+256,x:x+256];pred=maps[t][y:y+256,x:x+256];cc=c.old.confusion(gt,pred);cm+=cc
        for v in c.old.metrics(cc)['classes']:per.append(dict(pair_id=row['pair_id'],previous=row['previous'],current=t,**v))
    return c.old.metrics(cm),per

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);p.add_argument('--model',choices=c.MODELS,required=True);a=p.parse_args();folder=a.run/a.model;full=folder/'four_period_prediction';full.mkdir(exist_ok=True)
    if (full/'inference.json').exists():print('REUSE_COMPLETED_PREDICTIONS',a.model,flush=True);return
    torch.set_num_threads(3);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');data=c.Data(a.run,a.roots,a.model);model=c.load_head(folder/'best_temporal_head.pt',device);probabilities={}
    for t in c.PERIODS[1:]:
        f=full/(t+'_raw_probability.npz')
        if not f.exists():np.savez_compressed(f,probability=c.predict_probability(model,data.features[t],device).astype('float16'))
        with np.load(f) as z:probabilities[t]=z['probability'].astype('float32')
    candidates=[];best=-1.;selected=None;maps=None;records=None
    for strength in [0.,.5,1.,2.]:
        current={};rr={}
        for prev,t in zip(c.PERIODS,c.PERIODS[1:]):current[t],rr[t]=decode(probabilities[t],data.axes[prev],data.axes[t],data.link[t],strength)
        val,_=evaluate(data,current,'val');score=val['macro_Dice'];candidates.append(dict(strength=strength,validation_macro_Dice=score));print('DECODER_VALIDATION',a.model,strength,score,flush=True)
        if score>best+1e-8:best=score;selected=strength;maps=current;records=rr
    c.js(folder/'decoder_selection.json',dict(selected_strength=selected,candidates=candidates,test_used_for_selection=False,reference='U-Net-derived proposal validation consistency; not independent manual accuracy'))
    maps['T1']=data.gate['T1'].astype('uint8')*4
    for t in c.PERIODS:
        cls=maps[t];gate=data.gate[t];rgb=data.rgb[t];Image.fromarray(cls).save(full/(t+'_temporal_classes.png'));Image.fromarray(gate.astype('uint8')*255).save(full/(t+'_root_mask.png'));Image.fromarray(c.PALETTE[cls]).save(full/(t+'_temporal_on_white.png'))
        overlay=rgb.copy();overlay[gate]=np.rint(.35*rgb[gate]+.65*c.PALETTE[cls[gate]]).astype('uint8');Image.fromarray(overlay).save(full/(t+'_temporal_overlay.png'))
        rgba=np.concatenate([c.PALETTE[cls],gate[...,None].astype('uint8')*255],2);Image.fromarray(rgba).save(full/(t+'_temporal_roots_RGBA.png'));Image.fromarray(data.axes[t]['axis_map'].astype('uint16')).save(full/(t+'_axis_ids_uint16.png'));Image.fromarray(data.axes[t]['uncertain'].astype('uint8')*255).save(full/(t+'_crossing_review_mask.png'))
        if t!='T1':c.js(full/(t+'_root_axes.json'),records[t])
    reports={};raw={t:np.where(data.gate[t],probabilities[t].argmax(0)+1,0).astype('uint8') for t in c.PERIODS[1:]}
    for split in ['val','test']:
        score,per=evaluate(data,maps,split);rawscore,_=evaluate(data,raw,split);reports[split]=dict(structured=score,raw_pixel=rawscore);c.csvout(folder/(split+'_per_pair_class_metrics.csv'),per)
    continuity=[]
    for t in c.PERIODS[1:]:
        bad=0
        for row in records[t]:
            values=np.unique(maps[t][data.axes[t]['axis_map']==row['axis_id']]);bad+=int(2 in values and 3 in values)
        assert bad==0
        continuity.append(dict(period=t,candidate_axes=len(records[t]),mixed_new_extension_axes=bad,unresolved_crossing_axes=sum(r['unresolved_crossing'] for r in records[t])))
    c.js(folder/'metrics.json',reports);c.csvout(folder/'continuity.csv',continuity);c.js(full/'inference.json',dict(version='unet_aligned_v21',head_SHA256=c.sha(folder/'best_temporal_head.pt'),reference_SHA256=c.sha(a.run/'data/selected_progress.json'),manual_labels_used_as_inference_features=False,manual_labels_used_for_training_and_validation_selection=True,root_networks_frozen=True,single_raster_label_per_pixel=True,continuity_grammar='Entire candidate root NEW, or PERSISTENT shaft with terminal EXTENSION; ambiguous geometry remains flagged',identity_not_independently_verified=True,historical_same_field_exposure=True,independent_external_test_images=0,selected_strength=selected,continuity=continuity));c.js(folder/'status.json',dict(stage='prediction_complete',selected_strength=selected,test_macro_Dice=reports['test']['structured']['macro_Dice']));print('PREDICTED',a.model,reports['test']['structured']['macro_Dice'],flush=True)

if __name__=='__main__':main()

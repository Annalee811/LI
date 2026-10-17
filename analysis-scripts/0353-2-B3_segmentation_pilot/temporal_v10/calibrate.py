"""Identical finite validation-only class calibration for all three temporal heads."""
import argparse,itertools,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import common as c
from model import TemporalHead

def main():
    a=argparse.ArgumentParser();a.add_argument('--run',type=Path,required=True);a.add_argument('--roots',type=Path,required=True);a.add_argument('--model',choices=c.MODELS);args=a.parse_args();run=args.run;torch.set_num_threads(3);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for n in [args.model] if args.model else c.MODELS:
        out=run/n
        if (out/'calibration.json').exists():print('CALIBRATION_ALREADY_FROZEN',n,flush=True);continue
        data=c.Data(run/'data',args.roots,n,[3,4]);ck=torch.load(out/'evaluation_head.pt',map_location='cpu',weights_only=True);m=TemporalHead().to(device,memory_format=torch.channels_last);m.load_state_dict(ck['state_dict']);m.eval()
        maps={t:c.predict(m,data.cache[t,3],device).astype(np.float16).astype(np.float32) for t in c.PERIODS[1:]};rows=[]
        for new,grow in itertools.product([.25,.5,1.,2.,4.],[.125,.25,.5,1.,2.,4.]):
            multipliers=np.array([1.,new,grow],np.float32)[:,None,None];cm=np.zeros((4,4),np.int64)
            for t,prob in maps.items():
                pred=np.where(data.gates[t,3],(prob*multipliers).argmax(0)+1,0).astype(np.uint8);gt=data.y[t][:,c.COLS[3]:c.COLS[3]+448];cm+=c.confusion(gt,pred)
            row=dict(persistent_scale=1.,new_scale=new,growth_scale=grow,validation_macro_Dice=c.metrics(cm)['macro_Dice']);rows.append(row)
        chosen=max(rows,key=lambda r:(r['validation_macro_Dice'],-abs(np.log2(r['new_scale']))-abs(np.log2(r['growth_scale']))));scales=[1.,chosen['new_scale'],chosen['growth_scale']]
        c.csvout(out/'class_calibration_grid.csv',rows)
        old=json.loads((out/'evaluation_metrics.json').read_text());c.js(out/'uncalibrated_evaluation_metrics.json',old);reports={}
        for split,col in [('validation',3),('repeated_benchmark',4)]:
            cm=np.zeros((4,4),np.int64);conditional=np.zeros_like(cm);per=[];folder=out/'evaluation'/split
            for t in c.PERIODS[1:]:
                raw=maps[t] if col==3 else c.predict(m,data.cache[t,col],device).astype(np.float16).astype(np.float32)
                prob=raw*np.array(scales,np.float32)[:,None,None];prob/=np.maximum(prob.sum(0,keepdims=True),1e-12)
                cls=prob.argmax(0)+1;gate=data.gates[t,col];pred=np.where(gate,cls,0).astype(np.uint8);gt=data.y[t][:,c.COLS[col]:c.COLS[col]+448]
                cc=c.confusion(gt,pred);cm+=cc;gg=gt.copy();gg[~gate]=255;conditional+=c.confusion(gg,cls,True);per.append(dict(period=t,**c.metrics(cc)))
                Image.fromarray(pred).save(folder/f'{t}_classes.png');np.savez_compressed(folder/f'{t}_conditional_probability.npz',probability=prob.astype(np.float16))
            reports[split]=dict(end_to_end=c.metrics(cm),classification_on_detected_human_roots=c.metrics(conditional),by_period=per)
        for filename in ['evaluation_head.pt','final_head.pt']:
            value=torch.load(out/filename,map_location='cpu',weights_only=True);value['class_multipliers']=scales;torch.save(value,out/filename)
        c.js(out/'evaluation_metrics.json',reports)
        record=dict(selected=chosen,class_multipliers=scales,selected_on='C3 only after uncalibrated best epoch is frozen',grid_size=len(rows),same_grid_all_models=True,C4_used_for_calibration=False,probability_storage_before_calibration='FP16 roundtrip, native blended probabilities',unmodified_annotations=True)
        c.js(out/'calibration.json',record);record2=json.loads((out/'training_run.json').read_text());record2.update(class_calibration=record,evaluation_head_sha256=c.sha(out/'evaluation_head.pt'),final_head_sha256=c.sha(out/'final_head.pt'));c.js(out/'training_run.json',record2)
        print('TEMPORAL_CALIBRATED',n,chosen,flush=True)
if __name__=='__main__':main()

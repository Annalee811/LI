"""C3-only checkpoint/chain penalty selection and split-safe final refit."""
import argparse,json,random,shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import common as c
from predict import load_head
from train import evaluate
from model import warm_start,loss
from coherence import decode,fragmentation

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(3);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');amp=device.type=='cpu' and torch.cpu._is_avx512_bf16_supported()
    for n in c.MODELS:
        out=a.run/n
        if (out/'chain_selection.json').exists():continue
        data=c.Data(a.run/'data',a.roots,n,range(5));old=json.loads((a.previous/n/'evaluation_metrics.json').read_text())['validation']['end_to_end'];base=np.array([r['Dice'] for r in old['classes']]);old_switches=0
        for t in c.PERIODS[1:]:
            y=np.array(Image.open(a.previous/n/'evaluation/validation'/f'{t}_classes.png'));old_switches+=fragmentation(y,data.gates[t,3])['class_switches']
        candidates=[(0,a.previous/n/'evaluation_head.pt')]+[(int(f.stem.replace('candidate_epoch','')),f) for f in sorted(out.glob('candidate_epoch*.pt'))]
        trials=[];best=None;winner=None
        for epoch,path in candidates:
            model=load_head(path,device);probs={t:c.predict(model,data.cache[t,3],device) for t in c.PERIODS[1:]}
            for penalty in [0.,.5,1.,2.,4.,8.,12.]:
                cm=np.zeros((4,4),np.int64);switches=0;short=0
                for t in c.PERIODS[1:]:
                    gate=data.gates[t,3];pred=decode(probs[t],gate,penalty);gt=data.y[t][:,c.COLS[3]:c.COLS[3]+448];cm+=c.confusion(gt,pred);f=fragmentation(pred,gate);switches+=f['class_switches'];short+=f['runs_shorter_than_5px']
                m=c.metrics(cm);vals=np.array([r['Dice'] for r in m['classes']]);eligible=bool(np.all(vals>=base-.005) and m['macro_Dice']>=old['macro_Dice']-.002 and switches<=old_switches)
                reduction=(old_switches-switches)/max(1,old_switches);score=m['macro_Dice']+.03*reduction
                row=dict(epoch=epoch,chain_penalty=penalty,macro_Dice=m['macro_Dice'],class_Dice=vals.tolist(),class_switches=switches,short_runs=short,eligible=eligible,selection_score=score);trials.append(row)
                if eligible and (best is None or score>best):best=score;winner=(path,row)
            print('CHAIN_CANDIDATE',n,epoch,'best',winner[1] if winner else None,flush=True)
        assert winner is not None,'Epoch0 unmodified must reproduce baseline and remain eligible'
        path,selected=winner;shutil.copyfile(out/'evaluation_head.pt',out/'pre_chain_evaluation_head.pt');shutil.copyfile(out/'final_head.pt',out/'pre_chain_final_head.pt')
        ck=torch.load(path,map_location='cpu',weights_only=True);ck['chain_penalty']=selected['chain_penalty'];ck['epoch']=selected['epoch'];ck['selection_protocol_version']='v12';torch.save(ck,out/'evaluation_head.pt');model=load_head(out/'evaluation_head.pt',device)
        reports={split:evaluate(model,data,col,device,out/'evaluation'/split) for split,col in [('validation',3),('repeated_benchmark',4)]};c.js(out/'evaluation_metrics.json',reports)
        # Evaluation always starts from a held-spatial evaluation head. Only final
        # deployment starts from the old all-upper fit; the two are never mixed.
        random.seed(c.SEED);np.random.seed(c.SEED);torch.manual_seed(c.SEED);data.draws=[];data.add_full();model=warm_start(a.previous/n/'final_head.pt',device)
        config=json.loads((out/'training_run.json').read_text());tw=torch.tensor(config['class_weights'],device=device);opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001)
        epochs=selected['epoch'];schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(1,epochs),eta_min=.00001);history=[]
        for epoch in range(1,epochs+1):
            model.train();values=[]
            for i in range(0,config['samples_per_epoch'],4):
                ss=[data.sample(i+j,[5]) for j in range(4)];x=torch.from_numpy(np.stack([s[0] for s in ss])).to(device).contiguous(memory_format=torch.channels_last);y=torch.from_numpy(np.stack([s[1] for s in ss])).to(device);opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp):logits,boundary=model.forward_with_boundary(x)
                calibrated=logits.float()+torch.tensor(np.log(np.asarray(model.class_multipliers)*np.asarray(model.class_adjustment)),device=device,dtype=torch.float32)[None,:,None,None]
                value=loss(calibrated,y,tw,boundary.float(),x[:,17]);assert torch.isfinite(value);value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();values.append(float(value.detach()))
            schedule.step();history.append(dict(epoch=epoch,loss=float(np.mean(values))))
            if epoch%10==0 or epoch==epochs:print('CHAIN_FINAL_REFIT',n,epoch,'of',epochs,flush=True)
        c.csvout(out/'final_training_draws.csv',data.draws);c.csvout(out/'final_history.csv',history)
        torch.save(dict(state_dict=model.state_dict(),epoch=epochs,class_multipliers=model.class_multipliers,class_adjustment=model.class_adjustment,chain_penalty=selected['chain_penalty'],config={**config,'purpose':'All-upper B3 deployment fit, not independent accuracy'}),out/'final_head.pt')
        config.update(selected_epoch=epochs,final_epochs_completed=epochs,evaluation_head_sha256=c.sha(out/'evaluation_head.pt'),final_head_sha256=c.sha(out/'final_head.pt'),checkpoint_selection='C3 only: macroDice+.03*fraction fewer along-chain switches; each class Dice loss<=.005, macro loss<=.002, switches<=v11',chain_penalty=selected['chain_penalty']);c.js(out/'training_run.json',config)
        c.js(out/'chain_selection.json',dict(selected=selected,previous_switches=old_switches,trials=trials,C4_used=False,manual_labels_unchanged=True,whole_connected_component_forced_one_class=False,raw_probabilities='CNN probabilities with inherited v11 adjustments; chain decoding does not modify them and is not probability calibration'))
        print('CHAIN_SELECTED',n,selected,flush=True)
if __name__=='__main__':main()

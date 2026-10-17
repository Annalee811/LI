"""Train paired temporal heads, score held spatial columns, refit all B3 labels."""
import argparse,json,random,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import common as c
from model import TemporalHead,loss

def evaluate(model,data,col,device,out=None):
    cm=np.zeros((4,4),np.int64);conditional=np.zeros_like(cm);per=[]
    for t in c.PERIODS[1:]:
        prob=c.predict(model,data.cache[t,col],device);classes=prob.argmax(0).astype(np.uint8)+1
        gate=data.gates[t,col];pred=np.where(gate,classes,0).astype(np.uint8);left=c.COLS[col];gt=data.y[t][:,left:left+448]
        row=c.confusion(gt,pred);cm+=row
        # Classification among correctly detected, explicitly typed reference roots.
        detected_gt=gt.copy();detected_gt[~gate]=255;conditional+=c.confusion(detected_gt,classes,root_only=True)
        per.append(dict(period=t,**c.metrics(row)))
        if out:
            out.mkdir(parents=True,exist_ok=True);Image.fromarray(pred).save(out/f'{t}_classes.png');np.savez_compressed(out/f'{t}_conditional_probability.npz',probability=prob.astype(np.float16));Image.fromarray(gt).save(out/f'{t}_human_target.png')
    return dict(end_to_end=c.metrics(cm),classification_on_detected_human_roots=c.metrics(conditional),by_period=per)

def main():
    a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True);a.add_argument('--roots',type=Path,required=True);a.add_argument('--model',choices=c.MODELS,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--epochs',type=int,default=100);a.add_argument('--samples',type=int,default=64);a.add_argument('--benchmark',action='store_true');args=a.parse_args()
    random.seed(c.SEED);np.random.seed(c.SEED);torch.manual_seed(c.SEED);torch.set_num_threads(3)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');out=args.output/args.model;out.mkdir(parents=True,exist_ok=True);data=c.Data(args.data,args.roots,args.model,range(5))
    freq=np.zeros(3,np.int64)
    for col in range(3):
        left=c.COLS[col]
        for t in c.PERIODS[1:]:
            yy=data.y[t][:,left:left+448]
            freq+=np.array([(yy==k).sum() for k in [1,2,3]])
    weights=np.clip(np.sqrt(freq.max()/freq),1,6).astype(np.float32);tw=torch.tensor(weights,device=device)
    root_weight=args.roots/args.model/('best_weights_fp16.npz' if args.model=='deeplab' else 'best.pt')
    config=dict(model=args.model,root_architecture=c.MODEL_NAMES[args.model],root_checkpoint_sha256=c.sha(root_weight),root_stage='Frozen v9 network and frozen postprocessing; not re-trained or substituted',head='Identical native256 U-shaped11-input/3-output temporal CNN for all three root backbones',inputs='Predicted-root-only previous and current RGB; masked previous/current root probabilities; previous predicted tip,branch,distance cues',background_removed_before_classification=True,manual_previous_mask_at_training=False,manual_previous_mask_at_inference=False,temporal_pairs=['T1->T2','T2->T3','T3->T4'],classes=c.NAMES,T1='baseline only',death=False,seed=c.SEED,CUDA=torch.cuda.is_available(),device=str(device),class_weights=weights.tolist(),max_epochs=args.epochs,samples_per_epoch=args.samples,batch_size=4,learning_rate=.0005,train_columns=[0,1,2],validation_column=3,repeated_benchmark_column=4,checkpoint_selection='Maximum C3 end-to-end supported-class macro Dice; every5 epochs; minimum40epochs; patience6 validation checks',C4_used_for_selection=False,external_independent_test_images=0,comparison='Frozen root backbone + learned temporal head; not three end-to-end temporally fine-tuned backbones',annotation_sha256=c.sha(args.data/'original_human_progress.json'),class_counts_training=freq.tolist(),untyped_root_pixels='ignored',source_images=4,unique_training_pairs=216,unique_validation_pairs=72,repeated_benchmark_pairs=72)
    amp=device.type=='cpu' and torch.cpu._is_avx512_bf16_supported()
    config['arithmetic']='CPU BF16 autocast,FP32 parameters/softmax/loss' if amp else 'FP32'
    c.js(out/'config.json',config);model=TemporalHead().to(device,memory_format=torch.channels_last);config['temporal_head_parameters']=sum(p.numel() for p in model.parameters());c.js(out/'config.json',config)
    opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.0001);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=.00003)
    best=-1;bad=0;history=[];start=time.monotonic();best_epoch=0
    for epoch in range(1,args.epochs+1):
        model.train();losses=[];tic=time.monotonic()
        for i in range(0,8 if args.benchmark else args.samples,4):
            ss=[data.sample(i+j,[0,1,2]) for j in range(4)];x=torch.from_numpy(np.stack([s[0] for s in ss])).to(device);y=torch.from_numpy(np.stack([s[1] for s in ss])).to(device)
            x=x.contiguous(memory_format=torch.channels_last);opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp):logits=model(x)
            value=loss(logits.float(),y,tw);assert torch.isfinite(value);value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(value.detach()))
        if args.benchmark:
            print('TEMPORAL_BENCHMARK',args.model,'2steps_seconds',time.monotonic()-tic,flush=True);return
        schedule.step();row=dict(epoch=epoch,loss=float(np.mean(losses)),validation_macro_Dice=None,validation_conditional_macro_Dice=None,seconds=time.monotonic()-tic)
        if epoch%5==0 or epoch==args.epochs:
            report=evaluate(model,data,3,device);score=report['end_to_end']['macro_Dice'];row.update(validation_macro_Dice=score,validation_conditional_macro_Dice=report['classification_on_detected_human_roots']['macro_Dice'])
            if score>best+1e-5:
                best=score;bad=0;best_epoch=epoch;torch.save(dict(state_dict=model.state_dict(),epoch=epoch,config=config),out/'evaluation_head.pt')
            else:bad+=1
            print('TEMPORAL_EPOCH',args.model,epoch,'VAL',round(score,4),'CONDITIONAL',round(row['validation_conditional_macro_Dice'],4),flush=True)
        history.append(row);c.csvout(out/'history.csv',history)
        if epoch>=40 and bad>=6:break
    model.load_state_dict(torch.load(out/'evaluation_head.pt',map_location=device,weights_only=True)['state_dict'])
    reports={split:evaluate(model,data,col,device,out/'evaluation'/split) for split,col in [('validation',3),('repeated_benchmark',4)]}
    c.js(out/'evaluation_metrics.json',reports)
    c.csvout(out/'evaluation_training_draws.csv',data.draws);data.draws=[];data.add_full()
    # Final deployment head uses every labelled upper pixel, including prior guards.
    random.seed(c.SEED);np.random.seed(c.SEED);torch.manual_seed(c.SEED);model=TemporalHead().to(device,memory_format=torch.channels_last);opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.0001)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=best_epoch,eta_min=.00003);final_history=[]
    for epoch in range(1,best_epoch+1):
        model.train();losses=[]
        for i in range(0,args.samples,4):
            ss=[data.sample(i+j,[5]) for j in range(4)];x=torch.from_numpy(np.stack([s[0] for s in ss])).to(device);y=torch.from_numpy(np.stack([s[1] for s in ss])).to(device)
            x=x.contiguous(memory_format=torch.channels_last);opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp):logits=model(x)
            value=loss(logits.float(),y,tw);assert torch.isfinite(value);value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(value.detach()))
        schedule.step();final_history.append(dict(epoch=epoch,loss=float(np.mean(losses))));c.csvout(out/'final_history.csv',final_history)
        if epoch%10==0 or epoch==best_epoch:print('TEMPORAL_FINAL_REFIT',args.model,epoch,'of',best_epoch,flush=True)
    c.csvout(out/'final_training_draws.csv',data.draws)
    torch.save(dict(state_dict=model.state_dict(),epoch=best_epoch,config={**config,'fitted_pixels':'All2541x1108 upper pixels, including former guards','purpose':'Deployment fit using all B3 upper labels; fitted-image scores are not independent accuracy'}),out/'final_head.pt')
    c.js(out/'training_run.json',dict(**config,evaluation_epochs_completed=len(history),selected_epoch=best_epoch,final_epochs_completed=best_epoch,evaluation_head_sha256=c.sha(out/'evaluation_head.pt'),final_head_sha256=c.sha(out/'final_head.pt'),seconds=time.monotonic()-start))
    print('TEMPORAL_MODEL_DONE',args.model,'C3',best,flush=True)
if __name__=='__main__':main()

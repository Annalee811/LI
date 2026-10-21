"""Measured historical replay (reported separately), then conservative continuation."""
import argparse,copy,json,os,random,time
from pathlib import Path
import numpy as np
import torch
import common as c
from models import create,objective

def save(obj,path):
    tmp=path.with_name(path.name+'.tmp');torch.save(obj,tmp);os.replace(str(tmp),str(path))
def py_state(value):return tuple(py_state(v) for v in value) if isinstance(value,list) else value
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--model',choices=['unet','segformer'],required=True);p.add_argument('--threads',type=int,default=3);p.add_argument('--max-total-epochs',type=int,default=60);p.add_argument('--min-extra',type=int,default=10);p.add_argument('--patience',type=int,default=8);a=p.parse_args()
    
    if (a.run/a.model/'training_run.json').exists():print('REUSE_COMPLETED_TRAINING',a.model,flush=True);return
    torch.set_num_threads(a.threads);torch.manual_seed(20261018);np.random.seed(20261018);random.seed(20261018);out=a.run/a.model;out.mkdir(exist_ok=True);data=c.Data(a.run);source=a.source/a.model;old=c.read(source/'training_run.json');original_hist=c.csvread(source/'history.csv');old_draws=c.csvread(source/'training_draws.csv');selected_old=int(c.read(source/'selection.json')['selected_epoch']);threshold=float(c.read(source/'selection.json')['threshold']);device=torch.device('cpu');model=create(a.model).to(device)
    assert not torch.cuda.is_available(),'This replay is pinned to the original CPU execution'
    start_ck=Path(old['source_model_dir'])/'best.pt';assert c.sha(start_ck)==old['starting_checkpoint_SHA256'];model.load_state_dict(torch.load(start_ck,map_location=device,weights_only=True)['state_dict'])
    tr=[r for r in data.rows if r['split']=='train'];vr=[r for r in data.rows if r['split']=='val'];queue=[]
    def next_rows():
        nonlocal queue
        if len(queue)<2:
            if not queue:queue=tr[:];random.shuffle(queue)
        result=[]
        for _ in range(2):
            if not queue:queue=tr[:];random.shuffle(queue)
            result.append(queue.pop())
        return result
    def evaluate():
        model.eval();losses=[];cc=np.zeros(4,np.int64);fixed=np.zeros(4,np.int64)
        with torch.inference_mode():
            for k in range(0,len(vr),2):
                rr=vr[k:k+2];xx,yy,vv=data.batch(rr,False);x=torch.from_numpy(xx.transpose(0,3,1,2).copy());y=torch.from_numpy(yy[:,None]);v=torch.from_numpy(vv[:,None]);z=model(x);losses.append(float(objective(z,y,v)));pp=z.sigmoid()[:,0].numpy()
                for r,pr in zip(rr,pp):
                    _,gt=data.load(r);cc+=c.counts(gt,pr>=.5);fixed+=c.counts(gt,pr>=threshold)
        report=c.metrics(cc);report['Accuracy']=float((cc[0]+cc[3])/cc.sum());return float(np.mean(losses)),report,c.metrics(fixed)
    def train_epoch(opt):
        model.train();losses=[];draws=[]
        for step in range(0,64,2):
            rr=next_rows();xx,yy,vv=data.batch(rr,True);x=torch.from_numpy(xx.transpose(0,3,1,2).copy());y=torch.from_numpy(yy[:,None]);v=torch.from_numpy(vv[:,None]);opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cpu',dtype=torch.bfloat16):z=model(x)
            loss=objective(z,y,v);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()));draws.extend(dict(draw=step+j,image_id=r['image_id'],tile_id=r['tile_id']) for j,r in enumerate(rr))
        return float(np.mean(losses)),draws
    def record(epoch,loss,vl,met,fixed,lr,phase):
        return dict(epoch=epoch,train_loss=loss,validation_loss=vl,validation_Dice=met['Dice'],validation_IoU=met['IoU'],validation_Accuracy=met['Accuracy'],validation_Precision=met['Precision'],validation_Recall=met['Recall'],validation_TP=met['TP'],validation_FP=met['FP'],validation_FN=met['FN'],validation_TN=met['TN'],selection_validation_Dice=fixed['Dice'],learning_rate=lr,phase=phase)
    history=[];drawlog=[];replay=out/'historical_replay_complete.json';resume=out/'replay_resume.pt';baseline_info=None
    if not replay.exists():
        opt=torch.optim.AdamW(model.parameters(),lr=8e-5,weight_decay=1e-4);baseline_info=evaluate();first=1;differences=[];weight_difference=None
        if resume.exists():
            ck=torch.load(resume,map_location='cpu',weights_only=False);model.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['torch_rng']);random.setstate(py_state(ck['python_rng']));np.random.set_state(('MT19937',np.array(ck['numpy_rng'][0],np.uint32),ck['numpy_rng'][1],ck['numpy_rng'][2],ck['numpy_rng'][3]));queue=[tr[i] for i in ck['queue']];history=ck['history'];drawlog=ck['drawlog'];differences=ck['differences'];weight_difference=ck['weight_difference'];first=ck['epoch']+1
        expected_state=torch.load(source/'best.pt',map_location='cpu',weights_only=True)['state_dict']
        for epoch in range(first,21):
            tic=time.monotonic();lr=8e-5*(.1+.9*.5*(1+np.cos(np.pi*(epoch-1)/20)))
            for g in opt.param_groups:g['lr']=lr
            loss,draws=train_epoch(opt);expected=original_hist[epoch-1];error=abs(loss-float(expected['train_loss']));expected_ids=[r['image_id'] for r in old_draws if int(r['epoch'])==epoch]
            if epoch==1:c.js(out/('first_epoch_replay_threads_'+str(a.threads)+'.json'),dict(threads=a.threads,loss=loss,original_loss=float(expected['train_loss']),absolute_error=error,draw_sequence_matches=[r['image_id'] for r in draws]==expected_ids,model_code_matches_archived_SHA256=c.sha(Path(__file__).with_name('models.py'))==old['model_code_SHA256']))
            assert np.isfinite(loss), 'Nonfinite replay loss'
            assert [r['image_id'] for r in draws]==expected_ids,'Historical draw sequence mismatch'
            vl,met,fixed=evaluate()
            if expected['validation_Dice']:
                for oldkey,key in [('validation_Dice','Dice'),('validation_Recall','Recall'),('validation_Precision','Precision')]:
                    pass  # Differences are measured and reported, never overwritten.
            if epoch==selected_old:
                weight_difference=max(float((z.cpu()-expected_state[name]).abs().max()) for name,z in model.state_dict().items())
            history.append(record(epoch,loss,vl,met,fixed,lr,'historical_replay_measurement'));drawlog.extend(dict(epoch=epoch,**d) for d in draws);differences.append(dict(epoch=epoch,training_loss_absolute_difference=error,training_loss_relative_difference=error/max(abs(float(expected['train_loss'])),1e-12),logged_validation_checked=bool(expected['validation_Dice']),validation_Dice_absolute_difference=abs(float(expected['validation_Dice'])-met['Dice']) if expected['validation_Dice'] else None))
            c.csvout(out/'history.csv',history);c.csvout(out/'training_draws.csv',drawlog);ns=np.random.get_state();save(dict(epoch=epoch,state_dict=model.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=[ns[1].tolist(),ns[2],ns[3],ns[4]],queue=[tr.index(r) for r in queue],history=history,drawlog=drawlog,differences=differences,weight_difference=weight_difference),resume)
            c.js(out/'status.json',dict(stage='historical_replay',epoch=epoch,state='running'));print('REPLAY',a.model,epoch,'loss',loss,'val_loss',vl,'difference',error,'seconds',round(time.monotonic()-tic,1),flush=True)
        assert weight_difference is not None;c.js(replay,dict(replay_complete=True,exact_original_history=False,near_original_training_loss=max(r['training_loss_relative_difference'] for r in differences)<=.01,near_original_logged_Dice=max(r['validation_Dice_absolute_difference'] for r in differences if r['validation_Dice_absolute_difference'] is not None)<=.005,source_history_SHA256=c.sha(source/'history.csv'),source_checkpoint_SHA256=c.sha(source/'best.pt'),max_weight_difference=weight_difference,epochs=differences,validation_loss_origin='Actual validation measurement on replayed models; not an original historical loss record'))
    else:
        history=c.csvread(out/'history.csv');history=[{k:(float(v) if k not in ['epoch','phase'] else (int(v) if k=='epoch' else v)) for k,v in r.items()} for r in history];history=[r for r in history if r['epoch']<=20];drawlog=c.csvread(out/'training_draws.csv');drawlog=[r for r in drawlog if int(r['epoch'])<=20]
    # Restore the existing final model. Extra training cannot replace the archived
    # model unless the fixed original validation checks improve conservatively.
    model.load_state_dict(torch.load(source/'best.pt',map_location='cpu',weights_only=True)['state_dict']);torch.manual_seed(20261021);random.seed(20261021);np.random.seed(20261021);queue=[];opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4);base_loss,base_met,base_fixed=evaluate();best_score=base_fixed['Dice'];best_epoch=selected_old;bad=0;loss_bad=0;min_loss=base_loss;lr=1e-5;first=21;extra_resume=out/'resume.pt'
    if not (out/'best.pt').exists():
        import shutil;shutil.copy2(source/'best.pt',out/'best.pt')
    if extra_resume.exists():
        ck=torch.load(extra_resume,map_location='cpu',weights_only=False);model.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['torch_rng']);random.setstate(py_state(ck['python_rng']));np.random.set_state(('MT19937',np.array(ck['numpy_rng'][0],np.uint32),ck['numpy_rng'][1],ck['numpy_rng'][2],ck['numpy_rng'][3]));queue=[tr[i] for i in ck['queue']];history=ck['history'];drawlog=ck['drawlog'];first=ck['epoch']+1;best_score=ck['best_score'];best_epoch=ck['best_epoch'];bad=ck['bad'];loss_bad=ck['loss_bad'];min_loss=ck['min_loss'];lr=ck['next_lr']
    c.js(out/'baseline_before_extension.json',dict(validation_loss=base_loss,validation_at_05=base_met,validation_at_original_threshold=base_fixed,threshold=threshold,original_final_checkpoint_SHA256=c.sha(source/'best.pt')))
    for epoch in range(first,a.max_total_epochs+1):
        tic=time.monotonic()
        for g in opt.param_groups:g['lr']=lr
        loss,draws=train_epoch(opt);vl,met,fixed=evaluate();history.append(record(epoch,loss,vl,met,fixed,lr,'additional_training_from_original_final_weights'));drawlog.extend(dict(epoch=epoch,**d) for d in draws)
        eligible=(fixed['Dice']>best_score+1e-4 and fixed['Precision']>=base_fixed['Precision']-.005 and fixed['Recall']>=base_fixed['Recall']-.005 and vl<=base_loss+.01)
        if eligible:
            best_score=fixed['Dice'];best_epoch=epoch;bad=0;save(dict(state_dict=model.state_dict(),epoch=epoch,validation_Dice_at_original_threshold=best_score,source_checkpoint_SHA256=c.sha(source/'best.pt')),out/'best.pt')
        else:bad+=1
        if vl<min_loss-1e-4:min_loss=vl;loss_bad=0
        else:loss_bad+=1
        next_lr=lr
        if loss_bad>=4:next_lr=max(1e-7,lr*.5);loss_bad=0
        c.csvout(out/'history.csv',history);c.csvout(out/'training_draws.csv',drawlog);ns=np.random.get_state();save(dict(epoch=epoch,state_dict=model.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=[ns[1].tolist(),ns[2],ns[3],ns[4]],queue=[tr.index(r) for r in queue],history=history,drawlog=drawlog,best_score=best_score,best_epoch=best_epoch,bad=bad,loss_bad=loss_bad,min_loss=min_loss,next_lr=next_lr),extra_resume)
        c.js(out/'status.json',dict(stage='additional_training',epoch=epoch,state='running',best_epoch=best_epoch,baseline_retained=best_epoch<=20));print('EXTEND',a.model,epoch,'train_loss',loss,'val_loss',vl,'selection_Dice',fixed['Dice'],'best',best_epoch,'seconds',round(time.monotonic()-tic,1),flush=True);lr=next_lr
        if epoch>=20+a.min_extra and bad>=a.patience:break
    c.js(out/'training_run.json',dict(model=c.NAMES[a.model],source_run=str(a.source.resolve()),starting_final_checkpoint_SHA256=c.sha(source/'best.pt'),best_epoch=best_epoch,epochs_completed=max(r['epoch'] for r in history),extra_epochs_completed=max(r['epoch'] for r in history)-20,threshold=threshold,validation_loss_available_all_epochs=True,historical_validation_loss_measured_on_replay=True,original_historical_validation_loss_exists=False,baseline_retained=best_epoch<=20,checkpoint_SHA256=c.sha(out/'best.pt'),test_used_for_selection=False,threads=a.threads,training_draws_per_epoch=64,not_full_dataset_pass=True));c.js(out/'status.json',dict(state='training_complete',best_epoch=best_epoch));print('TRAINING_COMPLETE',a.model,best_epoch,flush=True)

if __name__=='__main__':main()

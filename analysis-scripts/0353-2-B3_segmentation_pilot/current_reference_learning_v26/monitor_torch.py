"""Measured replay with the same current annotation reference for train/validation."""
import argparse, json, os, random, time
from pathlib import Path
import numpy as np
import torch
import common as c
from models import create, objective


def save(value, path):
    temporary = path.with_name(path.name+'.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    for key in ['run', 'source', 'trajectory']:
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--model', choices=['unet','segformer'], required=True)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    out = args.run/args.model
    if (out/'completion.json').exists():
        print('REUSE_COMPLETED_CURRENT_REFERENCE_MONITOR', args.model, flush=True)
        return
    torch.set_num_threads(args.threads)
    torch.manual_seed(20261018); random.seed(20261018); np.random.seed(20261018)
    source = args.source/args.model
    archived = c.read(source/'training_run.json')
    initial = Path(archived['source_model_dir'])/'best.pt'
    final = source/'best.pt'
    assert c.sha(initial) == archived['starting_checkpoint_SHA256']
    assert c.sha(Path(__file__).with_name('models.py')) == archived['model_code_SHA256']
    fixed_threshold = float(c.read(source/'selection.json')['threshold'])
    rates = {int(r['epoch']):float(r['learning_rate']) for r in c.csvread(args.trajectory/args.model/'history.csv')}
    previous = {int(r['epoch']):r for r in c.csvread(args.trajectory/args.model/'history.csv')}
    expected_draws = c.csvread(args.trajectory/args.model/'training_draws.csv')
    assert set(rates) == set(range(1,31))
    data, old_data = c.Data(args.run), c.Data(args.source)
    training = [r for r in data.rows if r['split']=='train']
    validation = [r for r in data.rows if r['split']=='val']
    old_lookup = {r['image_id']:r for r in old_data.rows}
    model = create(args.model)
    model.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True)['state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(),lr=rates[1],weight_decay=1e-4)
    queue, history, draw_log = [], [], []
    snapshots = out/'epoch_checkpoints'; snapshots.mkdir(exist_ok=True)
    checkpoint = out/'resume.pt'

    def evaluate():
        model.eval(); losses=[]; current=np.zeros(4,np.int64); original=np.zeros(4,np.int64); at05=np.zeros(4,np.int64)
        with torch.inference_mode():
            for k in range(0,len(validation),2):
                rows = validation[k:k+2]; xx,yy,vv=data.batch(rows,False)
                logit=model(torch.from_numpy(xx.transpose(0,3,1,2).copy()))
                losses.append(float(objective(logit,torch.from_numpy(yy[:,None]),torch.from_numpy(vv[:,None]))))
                pp=logit.sigmoid()[:,0].numpy()
                for row,p in zip(rows,pp):
                    _,target=data.load(row); _,old_target=old_data.load(old_lookup[row['image_id']])
                    current += c.counts(target,p>=fixed_threshold)
                    at05 += c.counts(target,p>=.5)
                    original += c.counts(old_target,p>=.5)
        met=c.metrics(current);met['Accuracy']=float((current[0]+current[3])/current.sum())
        met['Balanced_accuracy']=float(.5*(current[0]/(current[0]+current[2])+current[3]/(current[3]+current[1])))
        return float(np.mean(losses)),met,c.metrics(at05),c.metrics(original)

    def next_rows():
        nonlocal queue
        result=[]
        for _ in range(2):
            if not queue:queue=training[:];random.shuffle(queue)
            result.append(queue.pop())
        return result

    first=1
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(state['state_dict']);optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng']);random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
        queue=[training[i] for i in state['queue']];history=state['history'];draw_log=state['draw_log'];first=state['epoch']+1
    else:
        loss,met,at05,old=evaluate()
        c.js(out/'initial_reference_evaluation.json',dict(loss=loss,current_reference=met,threshold=fixed_threshold,checkpoint_SHA256=c.sha(initial)))
    config=dict(model=c.NAMES[args.model],reference_manifest_SHA256=c.sha(args.run/'data/manifest.csv'),training_reference='current',validation_reference='current',training_views=200,validation_views=20,epoch_draws=64,batch_size=2,epochs=30,record_type='Actual newly measured replay; not recovered original validation logs',optimizer='AdamW',learning_rates='Exactly the previously used per-epoch rates',starting_checkpoint_SHA256=c.sha(initial),reported_final_checkpoint_SHA256=c.sha(final),reported_final_model_unchanged=True,old_reference_curves_plotted=False,threads=args.threads,torch_version=str(torch.__version__),model_code_SHA256=c.sha(Path(__file__).with_name('models.py')))
    c.js(out/'config.json',config)
    for epoch in range(first,31):
        tic=time.monotonic()
        if epoch==21:
            model.load_state_dict(torch.load(final,map_location='cpu',weights_only=True)['state_dict'])
            optimizer=torch.optim.AdamW(model.parameters(),lr=rates[epoch],weight_decay=1e-4)
            torch.manual_seed(20261021);random.seed(20261021);np.random.seed(20261021);queue=[]
            lo,met,_,_=evaluate()
            c.js(out/'phase2_start_reference_evaluation.json',dict(loss=lo,current_reference=met,threshold=fixed_threshold,reported_final_checkpoint_SHA256=c.sha(final)))
        for group in optimizer.param_groups:group['lr']=rates[epoch]
        model.train();losses=[];draws=[]
        for step in range(0,64,2):
            rows=next_rows();xx,yy,vv=data.batch(rows,True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cpu',dtype=torch.bfloat16):
                logit=model(torch.from_numpy(xx.transpose(0,3,1,2).copy()))
            loss=objective(logit,torch.from_numpy(yy[:,None]),torch.from_numpy(vv[:,None]))
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            losses.append(float(loss.detach()))
            draws.extend(dict(epoch=epoch,draw=step+j,image_id=r['image_id'],tile_id=r['tile_id']) for j,r in enumerate(rows))
        expected=[r['image_id'] for r in expected_draws if int(r['epoch'])==epoch]
        assert [r['image_id'] for r in draws]==expected,'Training draw sequence changed'
        vl,met,at05,old=evaluate();train_loss=float(np.mean(losses))
        record=dict(epoch=epoch,train_loss=train_loss,validation_loss=vl,validation_Dice=met['Dice'],validation_IoU=met['IoU'],validation_Accuracy=met['Accuracy'],validation_Balanced_accuracy=met['Balanced_accuracy'],validation_Precision=met['Precision'],validation_Recall=met['Recall'],validation_TP=met['TP'],validation_FP=met['FP'],validation_FN=met['FN'],validation_TN=met['TN'],validation_Dice_at_05=at05['Dice'],learning_rate=rates[epoch],threshold=fixed_threshold,phase='revised_target_training_replay' if epoch<=20 else 'reported_checkpoint_continuation_replay',train_loss_difference_from_V25=train_loss-float(previous[epoch]['train_loss']),audit_old_reference_Dice_at_05=old['Dice'],audit_old_Dice_difference=old['Dice']-float(previous[epoch]['validation_Dice']),seconds=time.monotonic()-tic)
        history.append(record);draw_log.extend(draws)
        snapshot=snapshots/('epoch_%03d.pt'%epoch)
        save(dict(epoch=epoch,state_dict=model.state_dict(),current_reference_manifest_SHA256=config['reference_manifest_SHA256'],validation=met,threshold=fixed_threshold),snapshot)
        c.csvout(out/'history.csv',history);c.csvout(out/'training_draws.csv',draw_log)
        save(dict(epoch=epoch,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),queue=[training.index(r) for r in queue],history=history,draw_log=draw_log),checkpoint)
        c.js(out/'status.json',dict(state='running',epoch=epoch,reference='current',checkpoint=str(snapshot)))
        print('CURRENT_REFERENCE_EPOCH',args.model,epoch,'train',train_loss,'val',vl,'Dice',met['Dice'],'seconds',round(record['seconds'],1),flush=True)
    c.js(out/'completion.json',dict(state='complete',epochs=30,per_epoch_snapshots=30,current_reference=True,reported_final_model_unchanged=True,reported_final_checkpoint_SHA256=c.sha(final)))
    c.js(out/'status.json',dict(state='complete',epochs=30));print('CURRENT_REFERENCE_COMPLETE',args.model,flush=True)


if __name__=='__main__':main()

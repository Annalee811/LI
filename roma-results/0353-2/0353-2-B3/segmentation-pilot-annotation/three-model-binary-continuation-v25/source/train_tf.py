"""Original TF1.12 DeepLab: checked replay and conservative continuation."""
from __future__ import print_function,division
import argparse,json,os,random,sys,time
from pathlib import Path
import numpy as np
import tensorflow as tf
import common as c

def erode(x):return tf.minimum(-tf.nn.max_pool(-x,[1,3,1,1],[1,1,1,1],'SAME'),-tf.nn.max_pool(-x,[1,1,3,1],[1,1,1,1],'SAME'))
def dilate(x):return tf.nn.max_pool(x,[1,3,3,1],[1,1,1,1],'SAME')
def skeleton(x):
    s=tf.nn.relu(x-dilate(erode(x)))
    for _ in range(5):
        x=erode(x);d=tf.nn.relu(x-dilate(erode(x)));s=s+tf.nn.relu(d-s*d)
    return s
def objective(logit,y,v):
    b=tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=logit);positive=tf.boolean_mask(b,(y>.5)&(v>.5));negative=tf.boolean_mask(b,(y<.5)&(v>.5));k=tf.minimum(tf.size(negative),tf.maximum(1,4*tf.size(positive)+256*tf.shape(y)[0]));hard=tf.nn.top_k(negative,k=k).values
    bce=(tf.reduce_sum(positive)+tf.reduce_sum(hard))/tf.maximum(1.,tf.cast(tf.size(positive)+tf.size(hard),tf.float32));pp=tf.sigmoid(logit)*v;g=y*v;dice=1-(2*tf.reduce_sum(pp*g)+1)/(tf.reduce_sum(pp)+tf.reduce_sum(g)+1);sp=skeleton(pp);sg=skeleton(g);pr=(tf.reduce_sum(sp*g)+1)/(tf.reduce_sum(sp)+1);re=(tf.reduce_sum(sg*pp)+1)/(tf.reduce_sum(sg)+1)
    return bce+dice+.15*(1-2*pr*re/(pr+re+1e-7))
def tuples(value):return tuple(tuples(x) for x in value) if isinstance(value,list) else value
def main():
    p=argparse.ArgumentParser()
    for key in ['run','source','vendor']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--predict-only',action='store_true');p.add_argument('--threads',type=int,default=2);p.add_argument('--max-total-epochs',type=int,default=60);p.add_argument('--min-extra',type=int,default=10);p.add_argument('--patience',type=int,default=8);a=p.parse_args();
    if (a.run/'deeplab/training_run.json').exists() and not a.predict_only:print('REUSE_COMPLETED_TRAINING deeplab',flush=True);return
    random.seed(20261018);np.random.seed(20261018);tf.set_random_seed(20261018);out=a.run/'deeplab';out.mkdir(parents=True,exist_ok=True);source=a.source/'deeplab';old=c.read(source/'training_run.json');old_hist=c.csvread(source/'history.csv');old_draws=c.csvread(source/'training_draws.csv');selected_old=int(c.read(source/'selection.json')['selected_epoch']);threshold=float(c.read(source/'selection.json')['threshold']);data=c.Data(a.run)
    vendor=a.vendor/'deeplab_model.py';assert c.sha(vendor)==old['original_vendor_SHA256'];raw=vendor.read_text();before='    if is_training:\n      exclude =';assert raw.count(before)==1;(out/'deeplab_model_adapted.py').write_text(raw.replace(before,'    if is_training and pre_trained_model:\n      exclude ='));sys.path.insert(0,str(a.vendor));sys.path.insert(0,str(out));import deeplab_model_adapted
    x=tf.placeholder(tf.float32,[None,256,256,3]);y=tf.placeholder(tf.float32,[None,256,256,1]);v=tf.placeholder(tf.float32,[None,256,256,1]);lr=tf.placeholder(tf.float32,[]);net=deeplab_model_adapted.deeplab_v3_plus_generator(2,16,'resnet_v2_50',None,.90);z=net((x-.5)/.25,True);modelvars=list(tf.global_variables());updates=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.variable_scope(tf.get_variable_scope(),reuse=True):zz=net((x-.5)/.25,False)
    prob=tf.sigmoid(zz[:,:,:,1]-zz[:,:,:,0])[:,:,:,None];loss=objective((z[:,:,:,1]-z[:,:,:,0])[:,:,:,None],y,v);reg=.0001*tf.add_n([tf.nn.l2_loss(q) for q in tf.trainable_variables() if len(q.shape)>1])
    with tf.control_dependencies(updates):
        opt=tf.train.AdamOptimizer(lr);gg=[(g,q) for g,q in opt.compute_gradients(loss+reg) if g is not None];gs,vs=zip(*gg);gs,_=tf.clip_by_global_norm(gs,5.);train=opt.apply_gradients(zip(gs,vs))
    # This loss uses the eval-mode logits and never runs training BN updates.
    val_loss=objective((zz[:,:,:,1]-zz[:,:,:,0])[:,:,:,None],y,v);allvars=list(tf.global_variables());names=[q.name for q in modelvars];allnames=[q.name for q in allvars];ph=[tf.placeholder(tf.float32,q.shape) for q in allvars];assign=[tf.assign(q,t) for q,t in zip(allvars,ph)];name_index={q:i for i,q in enumerate(allnames)};extra_vars=[q for q in allvars if q.name not in names]
    sess=tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=a.threads,inter_op_parallelism_threads=1,allow_soft_placement=True));sess.run(tf.global_variables_initializer());c.js(out/'weight_names.json',names)
    def restore_model(folder):
        nn=c.read(folder/'weight_names.json');vv=np.load(str(folder/'best_weights_fp16.npz'));values={name:vv['v%04d'%i].astype(np.float32) for i,name in enumerate(nn)};assert set(values)==set(names);sess.run([assign[name_index[name]] for name in names],{ph[name_index[name]]:values[name] for name in names})
    def save_model():
        temp=out/'best_weights_fp16.tmp.npz';np.savez_compressed(str(temp),**{'v%04d'%i:q.astype(np.float16) for i,q in enumerate(sess.run(modelvars))});os.replace(str(temp),str(out/'best_weights_fp16.npz'))
    tr=[r for r in data.rows if r['split']=='train'];vr=[r for r in data.rows if r['split']=='val'];queue=[]
    def next_rows():
        nonlocal queue
        rr=[]
        for _ in range(2):
            if not queue:queue=tr[:];random.shuffle(queue)
            rr.append(queue.pop())
        return rr
    def evaluate():
        cc=np.zeros(4,np.int64);fixed=np.zeros(4,np.int64);losses=[]
        for k in range(0,len(vr),2):
            rr=vr[k:k+2];xx,yy,vv=data.batch(rr,False);pr,lo=sess.run([prob,val_loss],{x:xx,y:yy[:,:,:,None],v:vv[:,:,:,None]});losses.append(float(lo))
            for r,pred in zip(rr,pr[:,:,:,0]):
                _,gt=data.load(r);cc+=c.counts(gt,pred>=.5);fixed+=c.counts(gt,pred>=threshold)
        met=c.metrics(cc);met['Accuracy']=float((cc[0]+cc[3])/cc.sum());return float(np.mean(losses)),met,c.metrics(fixed)
    def train_epoch(rate):
        losses=[];draws=[]
        for step in range(0,64,2):
            rr=next_rows();xx,yy,vv=data.batch(rr,True);lo,_=sess.run([loss,train],{x:xx,y:yy[:,:,:,None],v:vv[:,:,:,None],lr:rate});assert np.isfinite(lo);losses.append(float(lo));draws.extend(dict(draw=step+j,image_id=r['image_id'],tile_id=r['tile_id']) for j,r in enumerate(rr))
        return float(np.mean(losses)),draws
    def record(epoch,lo,vl,met,fixed,rate,phase):return dict(epoch=epoch,train_loss=lo,validation_loss=vl,validation_Dice=met['Dice'],validation_IoU=met['IoU'],validation_Accuracy=met['Accuracy'],validation_Precision=met['Precision'],validation_Recall=met['Recall'],validation_TP=met['TP'],validation_FP=met['FP'],validation_FN=met['FN'],validation_TN=met['TN'],selection_validation_Dice=fixed['Dice'],learning_rate=rate,phase=phase)
    def save_resume(path,meta):
        state=np.random.get_state();meta.update(python_rng=random.getstate(),numpy_rng=[state[1].tolist(),state[2],state[3],state[4]],queue=[tr.index(r) for r in queue]);arrays={'v%04d'%i:q for i,q in enumerate(sess.run(allvars))};arrays['metadata']=np.frombuffer(json.dumps(meta).encode('utf-8'),dtype=np.uint8);temp=path.with_name(path.stem+'.tmp.npz');np.savez(str(temp),**arrays);os.replace(str(temp),str(path))
    def restore_resume(path):
        nonlocal queue
        with np.load(str(path)) as z:
            meta=json.loads(z['metadata'].tobytes().decode('utf-8'));sess.run(assign,{q:z['v%04d'%i] for i,q in enumerate(ph)})
        random.setstate(tuples(meta['python_rng']));ns=meta['numpy_rng'];np.random.set_state(('MT19937',np.array(ns[0],np.uint32),ns[1],ns[2],ns[3]));queue=[tr[i] for i in meta['queue']];return meta
    if a.predict_only:
        restore_model(out)
        def infer(ims):return sess.run(prob,{x:ims})[:,:,:,0]
        reports={}
        for split in ['val','test']:
            maps=data.predict(infer,split,2);c.save_probabilities(out/'probabilities'/split,maps);reports[split],rr=data.evaluate(maps,threshold);c.csvout(out/(split+'_per_tile_metrics.csv'),rr)
        c.js(out/'metrics.json',reports);c.full_predict(infer,a.run,out/'upper',threshold,2);sess.close();print('FINAL_PREDICTIONS_COMPLETE deeplab',flush=True);return
    restore_model(Path(old['source_model_dir']));assert c.sha(Path(old['source_model_dir'])/'best_weights_fp16.npz')==old['starting_checkpoint_SHA256'];history=[];drawlog=[];differences=[];first=1;max_weight_difference=None;replay=out/'historical_replay_complete.json';resume=out/'replay_resume.npz'
    if not replay.exists():
        if not resume.exists():evaluate()
        if resume.exists():
            meta=restore_resume(resume);history=meta['history'];drawlog=meta['drawlog'];differences=meta['differences'];max_weight_difference=meta['max_weight_difference'];first=meta['epoch']+1
        for epoch in range(first,21):
            tic=time.time();rate=8e-5*(.1+.9*.5*(1+np.cos(np.pi*(epoch-1)/20)));lo,draws=train_epoch(rate);error=abs(lo-float(old_hist[epoch-1]['train_loss']));expected_ids=[r['image_id'] for r in old_draws if int(r['epoch'])==epoch];assert [r['image_id'] for r in draws]==expected_ids,'Historical draw sequence mismatch';assert np.isfinite(lo)
            vl,met,fixed=evaluate()
            if old_hist[epoch-1]['validation_Dice']:
                pass  # Compare all replayed scores without replacing the original log.
            if epoch==selected_old:
                archive=np.load(str(source/'best_weights_fp16.npz'));nn=c.read(source/'weight_names.json');expected={name:archive['v%04d'%i] for i,name in enumerate(nn)};actual=sess.run(modelvars);max_weight_difference=max(float(np.max(np.abs(value.astype(np.float16).astype(np.float32)-expected[name].astype(np.float32)))) for name,value in zip(names,actual))
            history.append(record(epoch,lo,vl,met,fixed,rate,'historical_replay_measurement'));drawlog.extend(dict(epoch=epoch,**d) for d in draws);differences.append(dict(epoch=epoch,training_loss_absolute_difference=error,training_loss_relative_difference=error/max(abs(float(old_hist[epoch-1]['train_loss'])),1e-12),validation_Dice_absolute_difference=abs(met['Dice']-float(old_hist[epoch-1]['validation_Dice'])) if old_hist[epoch-1]['validation_Dice'] else None));c.csvout(out/'history.csv',history);c.csvout(out/'training_draws.csv',drawlog);save_resume(resume,dict(epoch=epoch,history=history,drawlog=drawlog,differences=differences,max_weight_difference=max_weight_difference));c.js(out/'status.json',dict(stage='historical_replay',epoch=epoch,state='running'));print('REPLAY deeplab',epoch,'loss',lo,'val_loss',vl,'difference',error,'seconds',round(time.time()-tic,1),flush=True)
        c.js(replay,dict(replay_complete=True,exact_original_history=False,near_original_training_loss=max(r['training_loss_relative_difference'] for r in differences)<=.01,near_original_logged_Dice=max(r['validation_Dice_absolute_difference'] for r in differences if r['validation_Dice_absolute_difference'] is not None)<=.005,source_history_SHA256=c.sha(source/'history.csv'),source_checkpoint_SHA256=c.sha(source/'best_weights_fp16.npz'),max_weight_difference=max_weight_difference,epochs=differences,validation_loss_origin='Actual validation measurement on replayed models; not an original historical loss record'))
    else:
        history=c.csvread(out/'history.csv');history=[{k:(float(v) if k not in ['epoch','phase'] else (int(v) if k=='epoch' else v)) for k,v in r.items()} for r in history];history=[r for r in history if r['epoch']<=20];drawlog=c.csvread(out/'training_draws.csv');drawlog=[r for r in drawlog if int(r['epoch'])<=20]
    restore_model(source);sess.run(tf.variables_initializer(extra_vars));random.seed(20261021);np.random.seed(20261021);queue=[];base_loss,base_met,base_fixed=evaluate();best_score=base_fixed['Dice'];best_epoch=selected_old;bad=0;loss_bad=0;min_loss=base_loss;rate=1e-5;first=21;last=out/'resume.npz'
    if not (out/'best_weights_fp16.npz').exists():
        import shutil;shutil.copy2(str(source/'best_weights_fp16.npz'),str(out/'best_weights_fp16.npz'))
    if last.exists():
        meta=restore_resume(last);history=meta['history'];drawlog=meta['drawlog'];first=meta['epoch']+1;best_score=meta['best_score'];best_epoch=meta['best_epoch'];bad=meta['bad'];loss_bad=meta['loss_bad'];min_loss=meta['min_loss'];rate=meta['next_lr']
    c.js(out/'baseline_before_extension.json',dict(validation_loss=base_loss,validation_at_05=base_met,validation_at_original_threshold=base_fixed,threshold=threshold,original_final_checkpoint_SHA256=c.sha(source/'best_weights_fp16.npz')))
    for epoch in range(first,a.max_total_epochs+1):
        tic=time.time();lo,draws=train_epoch(rate);vl,met,fixed=evaluate();history.append(record(epoch,lo,vl,met,fixed,rate,'additional_training_from_original_final_weights'));drawlog.extend(dict(epoch=epoch,**d) for d in draws);eligible=(fixed['Dice']>best_score+1e-4 and fixed['Precision']>=base_fixed['Precision']-.005 and fixed['Recall']>=base_fixed['Recall']-.005 and vl<=base_loss+.01)
        if eligible:best_score=fixed['Dice'];best_epoch=epoch;bad=0;save_model()
        else:bad+=1
        if vl<min_loss-1e-4:min_loss=vl;loss_bad=0
        else:loss_bad+=1
        next_lr=rate
        if loss_bad>=4:next_lr=max(1e-7,rate*.5);loss_bad=0
        c.csvout(out/'history.csv',history);c.csvout(out/'training_draws.csv',drawlog);save_resume(last,dict(epoch=epoch,history=history,drawlog=drawlog,best_score=best_score,best_epoch=best_epoch,bad=bad,loss_bad=loss_bad,min_loss=min_loss,next_lr=next_lr));c.js(out/'status.json',dict(stage='additional_training',epoch=epoch,state='running',best_epoch=best_epoch,baseline_retained=best_epoch<=20));print('EXTEND deeplab',epoch,'train_loss',lo,'val_loss',vl,'selection_Dice',fixed['Dice'],'best',best_epoch,'seconds',round(time.time()-tic,1),flush=True);rate=next_lr
        if epoch>=20+a.min_extra and bad>=a.patience:break
    c.js(out/'training_run.json',dict(model='DeepLabV3+',source_run=str(a.source.resolve()),starting_final_checkpoint_SHA256=c.sha(source/'best_weights_fp16.npz'),best_epoch=best_epoch,epochs_completed=max(int(r['epoch']) for r in history),extra_epochs_completed=max(int(r['epoch']) for r in history)-20,threshold=threshold,validation_loss_available_all_epochs=True,historical_validation_loss_measured_on_replay=True,original_historical_validation_loss_exists=False,baseline_retained=best_epoch<=20,checkpoint_SHA256=c.sha(out/'best_weights_fp16.npz'),test_used_for_selection=False,threads=a.threads,training_draws_per_epoch=64,not_full_dataset_pass=True));c.js(out/'status.json',dict(state='training_complete',best_epoch=best_epoch));print('TRAINING_COMPLETE deeplab',best_epoch,flush=True);sess.close()

if __name__=='__main__':main()

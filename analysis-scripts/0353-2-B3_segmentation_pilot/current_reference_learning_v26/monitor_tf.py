"""TF1.12 trajectory measurement with current train/validation references."""
from __future__ import print_function, division
import argparse, json, os, random, sys, time
from pathlib import Path
import numpy as np
import tensorflow as tf
import common as c


def erode(x):
    return tf.minimum(-tf.nn.max_pool(-x,[1,3,1,1],[1,1,1,1],'SAME'),-tf.nn.max_pool(-x,[1,1,3,1],[1,1,1,1],'SAME'))
def dilate(x):return tf.nn.max_pool(x,[1,3,3,1],[1,1,1,1],'SAME')
def skeleton(x):
    s=tf.nn.relu(x-dilate(erode(x)))
    for _ in range(5):
        x=erode(x);d=tf.nn.relu(x-dilate(erode(x)));s=s+tf.nn.relu(d-s*d)
    return s
def objective(logit,y,v):
    b=tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=logit)
    positive=tf.boolean_mask(b,(y>.5)&(v>.5));negative=tf.boolean_mask(b,(y<.5)&(v>.5))
    k=tf.minimum(tf.size(negative),tf.maximum(1,4*tf.size(positive)+256*tf.shape(y)[0]));hard=tf.nn.top_k(negative,k=k).values
    bce=(tf.reduce_sum(positive)+tf.reduce_sum(hard))/tf.maximum(1.,tf.cast(tf.size(positive)+tf.size(hard),tf.float32))
    pp=tf.sigmoid(logit)*v;g=y*v;dice=1-(2*tf.reduce_sum(pp*g)+1)/(tf.reduce_sum(pp)+tf.reduce_sum(g)+1)
    sp=skeleton(pp);sg=skeleton(g);pr=(tf.reduce_sum(sp*g)+1)/(tf.reduce_sum(sp)+1);re=(tf.reduce_sum(sg*pp)+1)/(tf.reduce_sum(sg)+1)
    return bce+dice+.15*(1-2*pr*re/(pr+re+1e-7))
def tuples(value):return tuple(tuples(x) for x in value) if isinstance(value,list) else value


def main():
    parser=argparse.ArgumentParser()
    for key in ['run','source','trajectory','vendor']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--threads',type=int,default=1);args=parser.parse_args();out=args.run/'deeplab';source=args.source/'deeplab'
    if (out/'completion.json').exists():print('REUSE_COMPLETED_CURRENT_REFERENCE_MONITOR deeplab',flush=True);return
    random.seed(20261018);np.random.seed(20261018);tf.set_random_seed(20261018)
    archived=c.read(source/'training_run.json');initial=Path(archived['source_model_dir'])
    assert c.sha(initial/'best_weights_fp16.npz')==archived['starting_checkpoint_SHA256']
    vendor=args.vendor/'deeplab_model.py';assert c.sha(vendor)==archived['original_vendor_SHA256']
    raw=vendor.read_text();before='    if is_training:\n      exclude =';assert raw.count(before)==1
    (out/'deeplab_model_adapted.py').write_text(raw.replace(before,'    if is_training and pre_trained_model:\n      exclude ='))
    sys.path.insert(0,str(args.vendor));sys.path.insert(0,str(out));import deeplab_model_adapted
    x=tf.placeholder(tf.float32,[None,256,256,3]);y=tf.placeholder(tf.float32,[None,256,256,1]);v=tf.placeholder(tf.float32,[None,256,256,1]);rate=tf.placeholder(tf.float32,[])
    net=deeplab_model_adapted.deeplab_v3_plus_generator(2,16,'resnet_v2_50',None,.90);z=net((x-.5)/.25,True)
    modelvars=list(tf.global_variables());updates=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.variable_scope(tf.get_variable_scope(),reuse=True):zz=net((x-.5)/.25,False)
    probability=tf.sigmoid(zz[:,:,:,1]-zz[:,:,:,0])[:,:,:,None]
    loss=objective((z[:,:,:,1]-z[:,:,:,0])[:,:,:,None],y,v)
    reg=.0001*tf.add_n([tf.nn.l2_loss(q) for q in tf.trainable_variables() if len(q.shape)>1])
    with tf.control_dependencies(updates):
        optimizer=tf.train.AdamOptimizer(rate);gg=[(g,q) for g,q in optimizer.compute_gradients(loss+reg) if g is not None]
        gs,vs=zip(*gg);gs,_=tf.clip_by_global_norm(gs,5.);train=optimizer.apply_gradients(zip(gs,vs))
    val_loss=objective((zz[:,:,:,1]-zz[:,:,:,0])[:,:,:,None],y,v)
    allvars=list(tf.global_variables());names=[q.name for q in modelvars];allnames=[q.name for q in allvars]
    ph=[tf.placeholder(tf.float32,q.shape) for q in allvars];assign=[tf.assign(q,t) for q,t in zip(allvars,ph)]
    indices={name:i for i,name in enumerate(allnames)};extra=[q for q in allvars if q.name not in names]
    session=tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=args.threads,inter_op_parallelism_threads=1,allow_soft_placement=True))
    session.run(tf.global_variables_initializer());c.js(out/'weight_names.json',names)
    def restore(folder):
        nn=c.read(folder/'weight_names.json')
        with np.load(str(folder/'best_weights_fp16.npz')) as weights:
            values={name:weights['v%04d'%i].astype(np.float32) for i,name in enumerate(nn)}
        assert set(values)==set(names)
        session.run([assign[indices[name]] for name in names],{ph[indices[name]]:values[name] for name in names})
    restore(initial)
    data,old_data=c.Data(args.run),c.Data(args.source);training=[r for r in data.rows if r['split']=='train'];validation=[r for r in data.rows if r['split']=='val'];old_lookup={r['image_id']:r for r in old_data.rows}
    threshold=float(c.read(source/'selection.json')['threshold'])
    previous={int(r['epoch']):r for r in c.csvread(args.trajectory/'deeplab/history.csv')};assert set(previous)==set(range(1,31))
    expected_draws=c.csvread(args.trajectory/'deeplab/training_draws.csv');queue=[];history=[];draw_log=[]
    snapshots=out/'epoch_checkpoints';snapshots.mkdir(exist_ok=True);checkpoint=out/'resume.npz'
    def evaluate():
        losses=[];current=np.zeros(4,np.int64);original=np.zeros(4,np.int64);at05=np.zeros(4,np.int64)
        for k in range(0,len(validation),2):
            rows=validation[k:k+2];xx,yy,vv=data.batch(rows,False)
            pp,lo=session.run([probability,val_loss],{x:xx,y:yy[:,:,:,None],v:vv[:,:,:,None]});losses.append(float(lo))
            for row,p in zip(rows,pp[:,:,:,0]):
                _,target=data.load(row);_,old_target=old_data.load(old_lookup[row['image_id']])
                current+=c.counts(target,p>=threshold);at05+=c.counts(target,p>=.5);original+=c.counts(old_target,p>=.5)
        met=c.metrics(current);met['Accuracy']=float((current[0]+current[3])/current.sum());met['Balanced_accuracy']=float(.5*(current[0]/(current[0]+current[2])+current[3]/(current[3]+current[1])))
        return float(np.mean(losses)),met,c.metrics(at05),c.metrics(original)
    def next_rows():
        nonlocal queue
        rows=[]
        for _ in range(2):
            if not queue:queue=training[:];random.shuffle(queue)
            rows.append(queue.pop())
        return rows
    def save_state(epoch):
        rng=np.random.get_state();meta=dict(epoch=epoch,history=history,draw_log=draw_log,python_rng=random.getstate(),numpy_rng=[rng[1].tolist(),rng[2],rng[3],rng[4]],queue=[training.index(r) for r in queue])
        values={'v%04d'%i:q for i,q in enumerate(session.run(allvars))};values['metadata']=np.frombuffer(json.dumps(meta).encode('utf-8'),np.uint8)
        temporary=out/'resume.tmp.npz';np.savez(str(temporary),**values);os.replace(str(temporary),str(checkpoint))
    first=1
    if checkpoint.exists():
        with np.load(str(checkpoint)) as weights:
            meta=json.loads(weights['metadata'].tobytes().decode('utf-8'))
            session.run(assign,{q:weights['v%04d'%i] for i,q in enumerate(ph)})
        random.setstate(tuples(meta['python_rng']));ns=meta['numpy_rng'];np.random.set_state(('MT19937',np.array(ns[0],np.uint32),ns[1],ns[2],ns[3]));queue=[training[i] for i in meta['queue']];history=meta['history'];draw_log=meta['draw_log'];first=meta['epoch']+1
    else:
        lo,met,_,_=evaluate();c.js(out/'initial_reference_evaluation.json',dict(loss=lo,current_reference=met,threshold=threshold,checkpoint_SHA256=c.sha(initial/'best_weights_fp16.npz')))
    config=dict(model='DeepLabV3+',reference_manifest_SHA256=c.sha(args.run/'data/manifest.csv'),training_reference='current',validation_reference='current',training_views=200,validation_views=20,epoch_draws=64,batch_size=2,epochs=30,record_type='Actual newly measured replay; not recovered original validation logs',optimizer='Adam',learning_rates='Exactly the previously used per-epoch rates',starting_checkpoint_SHA256=c.sha(initial/'best_weights_fp16.npz'),reported_final_checkpoint_SHA256=c.sha(source/'best_weights_fp16.npz'),reported_final_model_unchanged=True,old_reference_curves_plotted=False,threads=args.threads,OMP_threads=os.environ.get('OMP_NUM_THREADS'),OMP_wait_policy=os.environ.get('OMP_WAIT_POLICY'),KMP_blocktime=os.environ.get('KMP_BLOCKTIME'),tensorflow_version=tf.__version__,vendor_SHA256=c.sha(vendor))
    c.js(out/'config.json',config)
    for epoch in range(first,31):
        tic=time.time();learning_rate=float(previous[epoch]['learning_rate'])
        if epoch==21:
            restore(source);session.run(tf.variables_initializer(extra));random.seed(20261021);np.random.seed(20261021);queue=[]
            lo,met,_,_=evaluate();c.js(out/'phase2_start_reference_evaluation.json',dict(loss=lo,current_reference=met,threshold=threshold,reported_final_checkpoint_SHA256=c.sha(source/'best_weights_fp16.npz')))
        losses=[];draws=[]
        for step in range(0,64,2):
            rows=next_rows();xx,yy,vv=data.batch(rows,True)
            lo,_=session.run([loss,train],{x:xx,y:yy[:,:,:,None],v:vv[:,:,:,None],rate:learning_rate});assert np.isfinite(lo);losses.append(float(lo))
            draws.extend(dict(epoch=epoch,draw=step+j,image_id=r['image_id'],tile_id=r['tile_id']) for j,r in enumerate(rows))
        assert [r['image_id'] for r in draws]==[r['image_id'] for r in expected_draws if int(r['epoch'])==epoch],'Training draw sequence changed'
        vl,met,at05,old=evaluate();train_loss=float(np.mean(losses))
        record=dict(epoch=epoch,train_loss=train_loss,validation_loss=vl,validation_Dice=met['Dice'],validation_IoU=met['IoU'],validation_Accuracy=met['Accuracy'],validation_Balanced_accuracy=met['Balanced_accuracy'],validation_Precision=met['Precision'],validation_Recall=met['Recall'],validation_TP=met['TP'],validation_FP=met['FP'],validation_FN=met['FN'],validation_TN=met['TN'],validation_Dice_at_05=at05['Dice'],learning_rate=learning_rate,threshold=threshold,phase='revised_target_training_replay' if epoch<=20 else 'reported_checkpoint_continuation_replay',train_loss_difference_from_V25=train_loss-float(previous[epoch]['train_loss']),audit_old_reference_Dice_at_05=old['Dice'],audit_old_Dice_difference=old['Dice']-float(previous[epoch]['validation_Dice']),seconds=time.time()-tic)
        history.append(record);draw_log.extend(draws)
        snapshot=snapshots/('epoch_%03d.npz'%epoch);temporary=snapshots/('epoch_%03d.tmp.npz'%epoch)
        np.savez(str(temporary),**{'v%04d'%i:q for i,q in enumerate(session.run(modelvars))});os.replace(str(temporary),str(snapshot))
        c.csvout(out/'history.csv',history);c.csvout(out/'training_draws.csv',draw_log);save_state(epoch)
        c.js(out/'status.json',dict(state='running',epoch=epoch,reference='current',checkpoint=str(snapshot)))
        print('CURRENT_REFERENCE_EPOCH deeplab',epoch,'train',train_loss,'val',vl,'Dice',met['Dice'],'seconds',round(record['seconds'],1),flush=True)
    c.js(out/'completion.json',dict(state='complete',epochs=30,per_epoch_snapshots=30,current_reference=True,reported_final_model_unchanged=True,reported_final_checkpoint_SHA256=c.sha(source/'best_weights_fp16.npz')))
    c.js(out/'status.json',dict(state='complete',epochs=30));session.close();print('CURRENT_REFERENCE_COMPLETE deeplab',flush=True)


if __name__=='__main__':main()

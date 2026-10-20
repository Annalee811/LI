"""Train and evaluate TensorFlow DeepLabV3+ binary segmentation."""
from __future__ import print_function,division
import argparse,hashlib,random,sys,time
from pathlib import Path
import numpy as np
import tensorflow as tf
import common as c

def erode(x): return tf.minimum(-tf.nn.max_pool(-x,[1,3,1,1],[1,1,1,1],'SAME'),-tf.nn.max_pool(-x,[1,1,3,1],[1,1,1,1],'SAME'))
def dilate(x): return tf.nn.max_pool(x,[1,3,3,1],[1,1,1,1],'SAME')
def skeleton(x):
    s=tf.nn.relu(x-dilate(erode(x)))
    for _ in range(5):
        x=erode(x); d=tf.nn.relu(x-dilate(erode(x))); s=s+tf.nn.relu(d-s*d)
    return s

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True); p.add_argument('--vendor',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=20); p.add_argument('--threads',type=int,default=3); p.add_argument('--batch',type=int,default=2)
    p.add_argument('--benchmark',action='store_true'); p.add_argument('--predict-only',action='store_true'); a=p.parse_args()
    source=(a.run/Path(c.read(a.run/'data/summary.json')['initial_checkpoint_dir'])).resolve()/'deeplab'
    random.seed(c.SEED); np.random.seed(c.SEED); tf.set_random_seed(c.SEED)
    out=a.run/'deeplab'; out.mkdir(parents=True,exist_ok=True); data=c.Data(a.run)
    vendor=a.vendor/'deeplab_model.py'; assert c.sha(vendor)=='ba0e931160d08519b9c8f3fe23c61b0678c4faa0b8818c38e04d533ef75ad361'
    raw=vendor.read_text(); before='    if is_training:\n      exclude ='; assert raw.count(before)==1
    # Only skip the optional pretrained-checkpoint loader, preserving all architecture operations.
    adapted=raw.replace(before,'    if is_training and pre_trained_model:\n      exclude =')
    (out/'deeplab_model_adapted.py').write_text(adapted)
    sys.path.insert(0,str(a.vendor)); sys.path.insert(0,str(out)); import deeplab_model_adapted
    x=tf.placeholder(tf.float32,[None,256,256,3]); y=tf.placeholder(tf.float32,[None,256,256,1]); v=tf.placeholder(tf.float32,[None,256,256,1]); lr=tf.placeholder(tf.float32,[])
    net=deeplab_model_adapted.deeplab_v3_plus_generator(2,16,'resnet_v2_50',None,.90)
    if a.predict_only:
        z=net((x-.5)/.25,False); modelvars=list(tf.global_variables()); prob=tf.sigmoid(z[:,:,:,1]-z[:,:,:,0])[:,:,:,None]
    else:
        z=net((x-.5)/.25,True); modelvars=list(tf.global_variables()); updates=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.variable_scope(tf.get_variable_scope(),reuse=True): zz=net((x-.5)/.25,False)
        prob=tf.sigmoid(zz[:,:,:,1]-zz[:,:,:,0])[:,:,:,None]
        logit=(z[:,:,:,1]-z[:,:,:,0])[:,:,:,None]; b=tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=logit)
        positive=tf.boolean_mask(b,(y>.5)&(v>.5)); negative=tf.boolean_mask(b,(y<.5)&(v>.5))
        k=tf.minimum(tf.size(negative),tf.maximum(1,4*tf.size(positive)+256*tf.shape(y)[0])); hard=tf.nn.top_k(negative,k=k).values
        bce=(tf.reduce_sum(positive)+tf.reduce_sum(hard))/tf.maximum(1.,tf.cast(tf.size(positive)+tf.size(hard),tf.float32))
        pp=tf.sigmoid(logit)*v; g=y*v; dice=1-(2*tf.reduce_sum(pp*g)+1)/(tf.reduce_sum(pp)+tf.reduce_sum(g)+1)
        sp=skeleton(pp); sg=skeleton(g); pr=(tf.reduce_sum(sp*g)+1)/(tf.reduce_sum(sp)+1); re=(tf.reduce_sum(sg*pp)+1)/(tf.reduce_sum(sg)+1)
        loss=bce+dice+.15*(1-2*pr*re/(pr+re+1e-7))
        # Same explicit L2 coefficient on trainable weights as the torch weight-decay protocol.
        reg=.0001*tf.add_n([tf.nn.l2_loss(z) for z in tf.trainable_variables() if len(z.shape)>1])
        with tf.control_dependencies(updates):
            opt=tf.train.AdamOptimizer(lr); gg=[(g,z) for g,z in opt.compute_gradients(loss+reg) if g is not None]
            gs,vs=zip(*gg); gs,_=tf.clip_by_global_norm(gs,5.); train=opt.apply_gradients(zip(gs,vs))
    names=[z.name for z in modelvars]; ph=[tf.placeholder(tf.float32,z.shape) for z in modelvars]; restoreops=[tf.assign(z,q) for z,q in zip(modelvars,ph)]
    sess=tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=a.threads,inter_op_parallelism_threads=1,allow_soft_placement=True)); sess.run(tf.global_variables_initializer())
    def infer(ims): return sess.run(prob,{x:ims})[:,:,:,0]
    def restore(folder=None):
        folder=out if folder is None else folder
        nn=c.read(folder/'weight_names.json'); vv=np.load(str(folder/'best_weights_fp16.npz'))
        values={name:vv['v%04d'%i].astype(np.float32) for i,name in enumerate(nn)}; assert set(names)==set(values)
        sess.run(restoreops,{q:values[name] for q,name in zip(ph,names)})
    if a.predict_only:
        restore(); c.full_predict(infer,a.run,out/'upper',c.read(out/'selection.json')['threshold'],a.batch); sess.close(); return
    restore(source)
    hh=hashlib.sha256()
    for name,z in zip(names,sess.run(modelvars)): hh.update(name.encode()); hh.update(z.tobytes())
    config=dict(model='DeepLabV3+',task='root/background',framework=tf.__version__,seed=c.SEED,initialization='Provided initial TensorFlow checkpoint',
        starting_checkpoint_SHA256=c.sha(source/'best_weights_fp16.npz'),source_model_dir=str(source),
        initial_random_state_SHA256=hh.hexdigest(),manifest_SHA256=c.sha(a.run/'data/manifest.csv'),original_vendor_SHA256=c.sha(vendor),adapted_source_SHA256=c.sha(out/'deeplab_model_adapted.py'),
        vendor_change='Guard pretrained loading with is_training and pre_trained_model; backbone and decoder unchanged',
        backbone='ResNet-v2-50',ASPP_channels=256,ASPP_rates=[6,12,18],decoder_channels=256,low_level_channels=48,output_stride=16,
        parameters_including_BN_state=sum(int(np.prod(z.shape.as_list())) for z in modelvars),CUDA=bool(tf.test.is_gpu_available()),
        epochs_max=a.epochs,samples_per_epoch=64,batch_size=a.batch,patch_size=256,threads=a.threads,optimizer='TensorFlow Adam + explicit L2 coefficient0.0001',
        learning_rate=.00008,normalization='(RGB/255-0.5)/0.25',BN_statistics='Updated on training tiles only, decay0.90',training_precision='FP32',evaluation_precision='FP32',
        loss='Hard-negative pixel BCE (4*positives+256*batch negatives) +soft Dice +0.15 soft clDice (5 iterations)',
        selection='Maximum validation Dice at0.5, every2 epochs; patience12 checks after30 epochs; threshold on validation only',
        annotation_source='Supplied human-reviewed masks',RGB_resized=False,test_used_for_training_or_selection=False,checkpoint_format='Original variables in FP16 NPZ; restored to FP32 for calibration and test')
    c.js(out/'config.json',config); c.js(out/'weight_names.json',names)
    tr=[r for r in data.rows if r['split']=='train']; queue=[]
    def next_rows():
        nonlocal queue
        rr=[]
        for _ in range(a.batch):
            if not queue: queue=tr[:]; random.shuffle(queue)
            rr.append(queue.pop())
        return rr
    baseline,_=data.evaluate(data.predict(infer,'val',a.batch),.5)
    best=baseline['Dice'];bad=0;hist=[];draws=[];start=time.time();bestepoch=0
    import shutil
    shutil.copyfile(str(source/'best_weights_fp16.npz'),str(out/'best_weights_fp16.npz'))
    c.js(out/'baseline_validation.json',baseline)
    c.js(out/'best_epoch.json',dict(epoch=0,validation_Dice_at_05=best))
    for epoch in range(1,a.epochs+1):
        tic=time.time(); losses=[]; rate=.00008*(.1+.9*.5*(1+np.cos(np.pi*(epoch-1)/a.epochs)))
        for step in range(0,4 if a.benchmark else 64,a.batch):
            rr=next_rows(); xx,yy,vv=data.batch(rr,True)
            lo,_=sess.run([loss,train],{x:xx,y:yy[:,:,:,None],v:vv[:,:,:,None],lr:rate})
            assert np.isfinite(lo); losses.append(float(lo)); draws.extend(dict(epoch=epoch,draw=step+j,image_id=r['image_id'],tile_id=r['tile_id']) for j,r in enumerate(rr))
        if a.benchmark: print('BENCHMARK deeplab',time.time()-tic,flush=True); return
        rec=dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_Dice=None,validation_Recall=None,validation_Precision=None,seconds=time.time()-tic)
        if epoch%2==0 or epoch==a.epochs:
            maps=data.predict(infer,'val',a.batch); m,_=data.evaluate(maps,.5); rec.update(validation_Dice=m['Dice'],validation_Recall=m['Recall'],validation_Precision=m['Precision'])
            if m['Dice']>best+1e-5:
                best=m['Dice']; bestepoch=epoch; bad=0
                np.savez_compressed(str(out/'best_weights_fp16.npz'),**{'v%04d'%i:z.astype(np.float16) for i,z in enumerate(sess.run(modelvars))})
                c.js(out/'best_epoch.json',dict(epoch=epoch,validation_Dice_at_05=best))
            else: bad+=1
        hist.append(rec); c.csvout(out/'history.csv',hist)
        print('EPOCH deeplab',epoch,'loss',round(rec['train_loss'],4),'val_Dice',rec['validation_Dice'],'sec',round(time.time()-tic,1),flush=True)
        if epoch>=30 and bad>=12: break
    restore(); maps=data.predict(infer,'val',a.batch); selected,grid=c.calibrate(data,maps); c.csvout(out/'validation_thresholds.csv',grid)
    c.js(out/'selection.json',dict(selected,selected_epoch=bestepoch,checkpoint_SHA256=c.sha(out/'best_weights_fp16.npz'),test_seen=False))
    c.save_probabilities(out/'probabilities/val',maps); test=data.predict(infer,'test',a.batch); c.save_probabilities(out/'probabilities/test',test)
    reports={}
    for split,pr in [('val',maps),('test',test)]:
        m,rr=data.evaluate(pr,selected['threshold']); reports[split]=m; c.csvout(out/(split+'_per_tile_metrics.csv'),rr)
    c.js(out/'metrics.json',reports); c.csvout(out/'training_draws.csv',draws)
    c.js(out/'training_run.json',dict(config,best_epoch=bestepoch,epochs_completed=len(hist),elapsed_seconds=time.time()-start,
        checkpoint_SHA256=c.sha(out/'best_weights_fp16.npz'),threshold=selected['threshold'],optimizer_updates=len(draws)//a.batch,unique_training_images=len({r['image_id'] for r in draws})))
    if (a.run/'data/upper').is_dir(): c.full_predict(infer,a.run,out/'upper',selected['threshold'],a.batch)
    sess.close(); print('MODEL_DONE deeplab',reports,flush=True)

if __name__=='__main__': main()

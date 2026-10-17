#!/usr/bin/env python
"""Original uploaded TensorFlow 1.12 DeepLabV3+, binary root/background only."""
from __future__ import print_function,division
import argparse,json,random,sys,time
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

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--vendor',type=Path,required=True)
    p.add_argument('--initial',type=Path,required=True);p.add_argument('--epochs',type=int,default=36);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--batch',type=int,default=2);p.add_argument('--lr',type=float,default=.0001);p.add_argument('--evaluate',choices=['test','train','upper'])
    a=p.parse_args();random.seed(c.SEED);np.random.seed(c.SEED);tf.set_random_seed(c.SEED)
    out=a.run/'deeplab';out.mkdir(parents=True,exist_ok=True);data=c.Data(a.run)
    sys.path.insert(0,str(a.vendor));import deeplab_model
    x=tf.placeholder(tf.float32,[None,256,256,3],name='RGB_div255')
    y=tf.placeholder(tf.float32,[None,256,256,1]);v=tf.placeholder(tf.float32,[None,256,256,1]);lr=tf.placeholder(tf.float32,[])
    # Frozen BN moving statistics; convolution weights and affine parameters remain trainable.
    logits=deeplab_model.deeplab_v3_plus_generator(2,16,'resnet_v2_50',None,.90)(x,False)
    logit=(logits[:,:,:,1]-logits[:,:,:,0])[:,:,:,None];prob=tf.sigmoid(logit)
    modelvars=list(tf.global_variables());names=[z.name for z in modelvars]
    restoreph=[tf.placeholder(tf.float32,z.shape) for z in modelvars];restoreop=[tf.assign(z,pv) for z,pv in zip(modelvars,restoreph)]
    if not a.evaluate:
        b=tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=logit)
        pos=tf.boolean_mask(b,(y>.5)&(v>.5));neg=tf.boolean_mask(b,(y<.5)&(v>.5))
        k=tf.maximum(1,tf.cast(tf.cast(tf.size(neg),tf.float32)*.2,tf.int32));negative=tf.reduce_mean(tf.nn.top_k(neg,k=k).values)
        positive=tf.reduce_sum(pos)/tf.maximum(1.,tf.cast(tf.size(pos),tf.float32))
        pp=prob*v;g=y*v;dice=1-(2*tf.reduce_sum(pp*g)+1)/(tf.reduce_sum(pp)+tf.reduce_sum(g)+1)
        sp=skeleton(pp);sg=skeleton(g);prec=(tf.reduce_sum(sp*g)+1)/(tf.reduce_sum(sp)+1);rec=(tf.reduce_sum(sg*pp)+1)/(tf.reduce_sum(sg)+1)
        loss=.5*(positive+negative)+dice+.15*(1-2*prec*rec/(prec+rec+1e-7))
        opt=tf.train.AdamOptimizer(lr);grads=[(g,z) for g,z in opt.compute_gradients(loss) if g is not None]
        gs,vs=zip(*grads);gs,_=tf.clip_by_global_norm(gs,5.);train=opt.apply_gradients(zip(gs,vs))
    sess=tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=a.threads,inter_op_parallelism_threads=1,allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())
    def restore(folder):
        nn=json.loads((folder/'weight_names.json').read_text());vv=np.load(str(folder/'best_weights_fp16.npz'))
        values={name:vv['v%04d'%i].astype(np.float32) for i,name in enumerate(nn)}
        assert set(values)==set(names),'Original model variables must match exactly'
        sess.run(restoreop,{ph:values[name] for ph,name in zip(restoreph,names)})
    def infer(ims):return sess.run(prob,{x:ims})[:,:,:,0]
    if a.evaluate:
        assert (a.run/'selection.json').exists(),'Freeze validation-only model selection before test'
        restore(out);run=json.loads((out/'training_run.json').read_text());threshold=run['frozen_threshold']
        if a.evaluate=='upper':
            from PIL import Image
            folder=out/'predictions/upper';folder.mkdir(parents=True,exist_ok=True)
            for t,rgb in data.rgb.items():
                h,w=rgb.shape[:2];xs=list(range(0,w-255,192));ys=list(range(0,h-255,192))
                if xs[-1]!=w-256:xs.append(w-256)
                if ys[-1]!=h-256:ys.append(h-256)
                coords=[(xx,yy) for yy in ys for xx in xs];sm=np.zeros((h,w),np.float32);weight=np.zeros_like(sm)
                han=np.hanning(256).astype(np.float32);blend=np.maximum(han[:,None]*han[None,:],.02)
                for i in range(0,len(coords),a.batch):
                    local=coords[i:i+a.batch];ims=np.stack([rgb[yy:yy+256,xx:xx+256] for xx,yy in local]).astype(np.float32)/255
                    for (xx,yy),pr in zip(local,infer(ims)):sm[yy:yy+256,xx:xx+256]+=pr*blend;weight[yy:yy+256,xx:xx+256]+=blend
                assert (weight>0).all();pr=sm/weight
                np.save(str(folder/(t+'_probability.npy')),pr)
                Image.fromarray((pr>=threshold).astype(np.uint8)*255).save(str(folder/(t+'_root_mask.png')))
            print('ROOT_UPPER_DONE deeplab',flush=True);return
        cols=[4] if a.evaluate=='test' else [0,1,2];reports=[]
        for col in cols:
            maps=data.column(infer,col,a.batch);c.save_maps(out/'predictions'/a.evaluate/('C'+str(col)),maps)
            reports.append(data.evaluate(maps,col,threshold))
        total=sum((np.array([r['root'][k] for k in ['TP','FP','FN','TN']]) for r in reports),np.zeros(4,np.int64))
        c.js(out/(a.evaluate+'_metrics.json'),dict(root=c.metrics(total),columns=reports,threshold=threshold))
        print('ROOT_EVALUATED deeplab',a.evaluate,c.metrics(total),flush=True);sess.close();return
    restore(a.initial.parent);c.js(out/'weight_names.json',names)
    config=dict(task='Standalone binary root/background only',model='Original uploaded TensorFlow DeepLabV3+',
        input_channels=3,output_classes=2,previous_period_input=False,temporal_loss=False,death_class=False,
        backbone='ResNet-v2-50',ASPP_channels=256,ASPP_rates=[6,12,18],low_level_channels=48,decoder_channels=256,output_stride=16,
        original_source_sha256=c.sha(a.vendor/'deeplab_model.py'),original_source_changed=False,
        initial_checkpoint_sha256=c.sha(a.initial),initialization_path=str(a.initial),
        initialization='Previously fitted T1/T2 binary root model; same training columns, no C4 fitted labels',
        batch_normalization='Frozen moving statistics from root-only checkpoint; affine and ALL convolution weights optimized',
        loss='0.5 positive BCE +0.5 hardest20% negative BCE +soft Dice +0.15 soft clDice',
        samples_per_epoch=48,epochs_max=a.epochs,learning_rate=a.lr,batch_size=a.batch,threads=a.threads,
        RGB_normalization='RGB/255',seed=c.SEED,patch_size=256,tensorflow_version=tf.__version__,
        device='CUDA' if tf.test.is_gpu_available() else 'CPU',cuda_available=bool(tf.test.is_gpu_available()),
        parameters_including_BN_state=sum(int(np.prod(z.shape.as_list())) for z in modelvars),
        test_used_during_training_or_selection=False,labels_modified=False,
        manifest_sha256=c.sha(a.run/'dataset/manifest.csv'),provenance_sha256=c.sha(a.run/'dataset/provenance.json'),
        selection='Maximum validation strict Dice at threshold0.5, every3 epochs; patience6 checks after minimum18 epochs',
        checkpoint_format='Original317 variables fp16 NPZ, restored to float32 before threshold selection and final evaluation')
    c.js(out/'config.json',config);hist=[];best=-1;bad=0;start=time.time()
    for epoch in range(1,a.epochs+1):
        tic=time.time();losses=[];rate=a.lr*(.05+.95*.5*(1+np.cos(np.pi*(epoch-1)/a.epochs)))
        for i in range(0,48,a.batch):
            samples=[data.sample(j) for j in range(i,min(i+a.batch,48))]
            feed={x:np.stack([z[0] for z in samples]),y:np.stack([z[1] for z in samples])[:,:,:,None],v:np.stack([z[2] for z in samples])[:,:,:,None],lr:rate}
            lo,_=sess.run([loss,train],feed);losses.append(float(lo))
        rec=dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_Dice=None,validation_Recall=None,validation_Precision=None,seconds=time.time()-tic)
        if epoch%3==0 or epoch==a.epochs:
            maps=data.column(infer,3,a.batch);m=data.evaluate(maps,3,.5)['root']
            rec.update(validation_Dice=m['Dice'],validation_Recall=m['Recall'],validation_Precision=m['Precision'])
            if m['Dice']>best+1e-5:
                best=m['Dice'];bad=0;values=sess.run(modelvars)
                np.savez_compressed(str(out/'best_weights_fp16.npz'),**{'v%04d'%i:z.astype(np.float16) for i,z in enumerate(values)})
                c.js(out/'best_epoch.json',dict(epoch=epoch,validation_Dice_at_05=best))
            else:bad+=1
            print('ROOT_EPOCH deeplab',epoch,'DICE',round(m['Dice'],4),'RECALL',round(m['Recall'],4),flush=True)
        hist.append(rec);c.csvout(out/'history.csv',hist)
        if epoch>=18 and bad>=6:break
    restore(out);maps=data.column(infer,3,a.batch);threshold,thresholds=c.calibrate(data,maps,3)
    c.csvout(out/'validation_thresholds.csv',thresholds);c.save_maps(out/'predictions/val/C3',maps)
    c.js(out/'validation_metrics.json',data.evaluate(maps,3,threshold));bestrec=json.loads((out/'best_epoch.json').read_text())
    run=dict(config);run.update(dict(epochs_completed=len(hist),best_epoch=bestrec['epoch'],frozen_threshold=threshold,
        checkpoint_sha256=c.sha(out/'best_weights_fp16.npz'),elapsed_seconds=time.time()-start));c.js(out/'training_run.json',run)
    print('ROOT_TRAIN_DONE deeplab',data.evaluate(maps,3,threshold)['root'],flush=True);sess.close()
if __name__=='__main__':main()

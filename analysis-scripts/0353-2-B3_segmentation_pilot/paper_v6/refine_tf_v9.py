#!/usr/bin/env python
"""Python3.5/TF1.12: original uploaded DeepLabV3+, native-pixel fine-tuning."""
from __future__ import print_function,division
import argparse,json,random,sys,time,shutil
from pathlib import Path
import numpy as np
import tensorflow as tf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'root_background_v5'))
import common as c
from train_tf import skeleton

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--vendor',type=Path,required=True);p.add_argument('--initial',type=Path,required=True);p.add_argument('--epochs',type=int,default=8);p.add_argument('--infer-only',action='store_true');a=p.parse_args()
 random.seed(c.SEED);np.random.seed(c.SEED);tf.set_random_seed(c.SEED);data=c.Data(a.data);out=a.run/'deeplab';out.mkdir(parents=True,exist_ok=True)
 sys.path.insert(0,str(a.vendor));import deeplab_model
 assert c.sha(a.vendor/'deeplab_model.py')=='ba0e931160d08519b9c8f3fe23c61b0678c4faa0b8818c38e04d533ef75ad361'
 x=tf.placeholder(tf.float32,[None,256,256,3]);y=tf.placeholder(tf.float32,[None,256,256,1]);v=tf.placeholder(tf.float32,[None,256,256,1])
 logits=deeplab_model.deeplab_v3_plus_generator(2,16,'resnet_v2_50',None,.90)(x,False);logit=(logits[:,:,:,1]-logits[:,:,:,0])[:,:,:,None];prob=tf.sigmoid(logit)
 modelvars=list(tf.global_variables());names=[z.name for z in modelvars];ph=[tf.placeholder(tf.float32,z.shape) for z in modelvars];restoreop=[tf.assign(z,q) for z,q in zip(modelvars,ph)]
 if not a.infer_only:
  b=tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=logit);g=y*v;q=prob*v
  thin=tf.cast((tf.nn.avg_pool(g,[1,5,5,1],[1,1,1,1],'SAME')<.3)&(g>.5),tf.float32);ww=1+thin
  pos=(g>.5)&(v>.5);neg=(g<.5)&(v>.5);pv=tf.boolean_mask(b*ww,pos);nv=tf.boolean_mask(b,neg)
  hard=tf.nn.top_k(nv,k=tf.maximum(1,tf.cast(tf.cast(tf.size(nv),tf.float32)*.2,tf.int32))).values
  bce=(tf.reduce_sum(pv)+tf.reduce_sum(hard))/tf.maximum(1.,tf.reduce_sum(tf.boolean_mask(ww,pos))+tf.cast(tf.size(hard),tf.float32))
  dice=1-(2*tf.reduce_sum(q*g)+1)/(tf.reduce_sum(q)+tf.reduce_sum(g)+1);sp=skeleton(q);sg=skeleton(g)
  pr=(tf.reduce_sum(sp*g)+1)/(tf.reduce_sum(sp)+1);re=(tf.reduce_sum(sg*q)+1)/(tf.reduce_sum(sg)+1)
  loss=bce+dice+.4*(1-2*pr*re/(pr+re+1e-7));opt=tf.train.AdamOptimizer(2e-5);gg=[(z,w) for z,w in opt.compute_gradients(loss) if z is not None];gs,vs=zip(*gg);gs,_=tf.clip_by_global_norm(gs,5);train=opt.apply_gradients(zip(gs,vs))
 sess=tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=3,inter_op_parallelism_threads=1,allow_soft_placement=True));sess.run(tf.global_variables_initializer())
 def restore(folder):
  nn=json.loads((folder/'weight_names.json').read_text());vv=np.load(str(folder/'best_weights_fp16.npz'));values={name:vv['v%04d'%i].astype(np.float32) for i,name in enumerate(nn)};assert set(names)==set(values);sess.run(restoreop,{z:values[n] for z,n in zip(ph,names)})
 def infer(ims):return sess.run(prob,{x:ims})[:,:,:,0]
 def save():np.savez_compressed(str(out/'best_weights_fp16.npz'),**{'v%04d'%i:z.astype(np.float16) for i,z in enumerate(sess.run(modelvars))})
 restore(out if a.infer_only else a.initial.parent);start=time.time()
 if not a.infer_only:
  config=dict(model='Original uploaded TensorFlow DeepLabV3+',framework=tf.__version__,initial_checkpoint_SHA256=c.sha(a.initial),vendor_SHA256=c.sha(a.vendor/'deeplab_model.py'),
   epochs=a.epochs,samples_per_epoch=32,batch=2,training_patch_size=256,inference_patch_size=256,TTA='none',seed=c.SEED,CUDA=bool(tf.test.is_gpu_available()),
   objective='Thin-labelled-root weighted BCE (2x, local5x5 occupancy<0.3), hardest20% background +soft Dice +0.4clDice',
   annotations_edited=False,train_columns=[0,1,2],validation_column=3,external_test_images=0,C4_status='Repeated same-position benchmark; excluded from fitting/selection',
   validation_selection='Maximum calibrated strict C3 Dice among epoch0,2,4,6,8; original start can be retained',RGB_only=True,death=False,BN_statistics='Frozen; affine trainable',learning_rate=2e-5)
  c.js(out/'config.json',config);c.js(out/'weight_names.json',names);shutil.copyfile(str(a.initial),str(out/'best_weights_fp16.npz'))
  maps=data.column(infer,3,2);threshold,grid=c.calibrate(data,maps,3);row=next(z for z in grid if z['threshold']==threshold);best=row['Dice'];hist=[dict(epoch=0,loss=None,validation_Dice=best,threshold=threshold)]
  c.save_maps(out/'probabilities/val',maps);c.js(out/'selection.json',dict(epoch=0,**row));c.csvout(out/'thresholds.csv',grid)
  for epoch in range(1,a.epochs+1):
   losses=[]
   for i in range(0,32,2):
    ss=[data.sample(i+j) for j in range(2)];feed={x:np.stack([s[0] for s in ss]),y:np.stack([s[1] for s in ss])[:,:,:,None],v:np.stack([s[2] for s in ss])[:,:,:,None]};lo,_=sess.run([loss,train],feed);losses.append(float(lo))
   rec=dict(epoch=epoch,loss=float(np.mean(losses)),validation_Dice=None,threshold=None)
   if epoch%2==0 or epoch==a.epochs:
    # Evaluate exactly the quantized checkpoint format that is delivered.
    current=sess.run(modelvars);save();restore(out);maps=data.column(infer,3,2);threshold,grid=c.calibrate(data,maps,3);row=next(z for z in grid if z['threshold']==threshold);rec.update(validation_Dice=row['Dice'],threshold=threshold)
    candidate=out/'best_weights_fp16.npz'
    if row['Dice']>best+1e-5:
     best=row['Dice'];shutil.copyfile(str(candidate),str(out/'selected_weights_fp16.npz'));c.save_maps(out/'probabilities/val',maps);c.js(out/'selection.json',dict(epoch=epoch,**row));c.csvout(out/'thresholds.csv',grid)
    sess.run(restoreop,{z:q for z,q in zip(ph,current)})
   hist.append(rec);c.csvout(out/'history.csv',hist);print('V9_TRAIN deeplab',rec,flush=True)
  selected=json.loads((out/'selection.json').read_text())
  shutil.copyfile(str(out/'selected_weights_fp16.npz' if selected['epoch'] else a.initial),str(out/'best_weights_fp16.npz'));restore(out)
  c.js(out/'training_run.json',dict(config,epochs_completed=a.epochs,optimizer_updates=a.epochs*16,selected=selected,selected_checkpoint_SHA256=c.sha(out/'best_weights_fp16.npz'),elapsed_seconds=time.time()-start))
 for col in [0,1,2,4]:
  c.save_maps(out/'probabilities'/('test' if col==4 else 'train')/('C'+str(col)),data.column(infer,col,2));print('V9_COLUMN deeplab',col,flush=True)
 folder=out/'probabilities/upper';folder.mkdir(parents=True,exist_ok=True);side=256;han=np.hanning(side).astype(np.float32);blend=np.maximum(han[:,None]*han[None,:],.05)
 for t,rgb in data.rgb.items():
  h,w=rgb.shape[:2];xs=list(range(0,w-side+1,128));ys=list(range(0,h-side+1,128))
  if xs[-1]!=w-side:xs.append(w-side)
  if ys[-1]!=h-side:ys.append(h-side)
  coords=[(xx,yy) for yy in ys for xx in xs];ss=np.zeros((h,w),np.float32);weight=np.zeros_like(ss)
  for i in range(0,len(coords),2):
   xy=coords[i:i+2];ims=np.stack([rgb[yy:yy+side,xx:xx+side] for xx,yy in xy]).astype(np.float32)/255
   for (xx,yy),pr in zip(xy,infer(ims)):ss[yy:yy+side,xx:xx+side]+=pr*blend;weight[yy:yy+side,xx:xx+side]+=blend
  assert (weight>0).all();np.save(str(folder/(t+'_probability.npy')),ss/weight);print('V9_UPPER deeplab',t,flush=True)
 sess.close();print('V9_INFERENCE_DONE deeplab',flush=True)
if __name__=='__main__':main()

#!/usr/bin/env python
"""Restore the supplied TF1 DeepLab portable weights and predict native-size RGB."""
from __future__ import print_function,division
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--images',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--threads',type=int,default=4);a=p.parse_args()
run=a.model_dir/'training';sys.path.insert(0,str(a.model_dir/'vendor'));sys.path.insert(0,str(run))
import deeplab_model_adapted as mod
config=json.loads((run/'training_run.json').read_text());threshold=config['frozen_threshold']
x=tf.placeholder(tf.float32,[None,256,256,3]);net=mod.deeplab_v3_plus_generator(2,16,'resnet_v2_50',None,.9);logits=net(x,False);probability=tf.nn.softmax(logits)[:,:,:,1]
sess=tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=a.threads,inter_op_parallelism_threads=1));sess.run(tf.global_variables_initializer())
names=json.loads((run/'weight_names.json').read_text());arrays=np.load(str(run/'best_weights_fp16.npz'));values={name:arrays['v%04d'%i] for i,name in enumerate(names)}
variables=tf.global_variables();assert set(values)=={v.name for v in variables},'Checkpoint variable names differ'
for v in variables:assert list(values[v.name].shape)==v.shape.as_list()
restore_inputs=[tf.placeholder(tf.float32,v.shape) for v in variables]
restore_ops=[tf.assign(v,ph) for v,ph in zip(variables,restore_inputs)]
sess.run(restore_ops,{ph:values[v.name].astype(np.float32) for ph,v in zip(restore_inputs,variables)})
a.output.mkdir(parents=True,exist_ok=True)
files=[a.images] if a.images.is_file() else sorted(f for f in a.images.iterdir() if f.suffix.lower() in ['.png','.jpg','.jpeg','.webp','.bmp'])
for f in files:
    image=np.array(Image.open(str(f)).convert('RGB'));h0,w0=image.shape[:2]
    image=np.pad(image,((0,max(0,256-h0)),(0,max(0,256-w0)),(0,0)),mode='edge');h,w=image.shape[:2]
    xs=list(range(0,w-256+1,192));ys=list(range(0,h-256+1,192))
    if xs[-1]!=w-256:xs.append(w-256)
    if ys[-1]!=h-256:ys.append(h-256)
    coords=[(xx,yy) for yy in ys for xx in xs];value=np.zeros((h,w),np.float32);weight=np.zeros((h,w),np.float32)
    hann=np.hanning(256).astype(np.float32);blend=np.maximum(hann[:,None]*hann[None,:],.02)
    for i in range(0,len(coords),4):
        chunk=coords[i:i+4];batch=np.stack([image[yy:yy+256,xx:xx+256] for xx,yy in chunk]).astype(np.float32)/255
        for (xx,yy),pr in zip(chunk,sess.run(probability,{x:batch})):
            value[yy:yy+256,xx:xx+256]+=pr*blend;weight[yy:yy+256,xx:xx+256]+=blend
    result=(value/weight)[:h0,:w0];np.save(str(a.output/(f.stem+'_probability.npy')),result)
    Image.fromarray((result>=threshold).astype(np.uint8)*255).save(str(a.output/(f.stem+'_prediction.png')))
    print('PREDICTED',f.name,w0,h0,flush=True)
sess.close()

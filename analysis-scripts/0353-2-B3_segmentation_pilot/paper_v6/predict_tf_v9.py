#!/usr/bin/env python
"""Original TF1.12 DeepLab inference without any human labels."""
from __future__ import print_function,division
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

def main():
 a=argparse.ArgumentParser();a.add_argument('--weights',type=Path,required=True);a.add_argument('--vendor',type=Path,required=True);a.add_argument('--rgb-dir',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 sys.path.insert(0,str(args.vendor));import deeplab_model
 x=tf.placeholder(tf.float32,[None,256,256,3]);logits=deeplab_model.deeplab_v3_plus_generator(2,16,'resnet_v2_50',None,.90)(x,False);prob=tf.sigmoid(logits[:,:,:,1]-logits[:,:,:,0]);variables=list(tf.global_variables());names=json.loads((args.weights/'weight_names.json').read_text());npz=np.load(str(args.weights/'best_weights_fp16.npz'));values={n:npz['v%04d'%i].astype(np.float32) for i,n in enumerate(names)}
 assert set(values)=={z.name for z in variables};ph=[tf.placeholder(tf.float32,z.shape) for z in variables];ops=[tf.assign(z,p) for z,p in zip(variables,ph)]
 sess=tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads=1,allow_soft_placement=True));sess.run(ops,{p:values[z.name] for p,z in zip(ph,variables)})
 files=sorted(f for f in args.rgb_dir.iterdir() if f.suffix.lower() in ['.png','.jpg','.jpeg','.tif','.tiff']);assert files;assert len({f.stem for f in files})==len(files);han=np.hanning(256).astype(np.float32);blend=np.maximum(han[:,None]*han[None,:],.05)
 for f in files:
  rgb=np.array(Image.open(str(f)).convert('RGB'));h,w=rgb.shape[:2];rgb=np.pad(rgb,((0,max(0,256-h)),(0,max(0,256-w)),(0,0)),mode='edge');hh,ww=rgb.shape[:2];xs=list(range(0,ww-255,128));ys=list(range(0,hh-255,128))
  if xs[-1]!=ww-256:xs.append(ww-256)
  if ys[-1]!=hh-256:ys.append(hh-256)
  coords=[(xx,yy) for yy in ys for xx in xs];s=np.zeros((hh,ww),np.float32);weight=np.zeros_like(s)
  for i in range(0,len(coords),2):
   xy=coords[i:i+2];ims=np.stack([rgb[yy:yy+256,xx:xx+256] for xx,yy in xy]).astype(np.float32)/255
   for (xx,yy),pr in zip(xy,sess.run(prob,{x:ims})):s[yy:yy+256,xx:xx+256]+=pr*blend;weight[yy:yy+256,xx:xx+256]+=blend
  assert (weight>0).all();np.save(str(args.output/(f.stem+'_probability.npy')),(s/weight)[:h,:w]);print('TF_PREDICTED',f.name,flush=True)
 sess.close()
if __name__=='__main__':main()

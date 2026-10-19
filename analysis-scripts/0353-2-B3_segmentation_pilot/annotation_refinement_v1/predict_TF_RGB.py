"""Frozen original DeepLabV3+ inference on other RGBs; Python3.5/TF1.12."""
from __future__ import print_function,division
import argparse,sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import common as c
from native_RGB import predict_files
from dataset_identity import normalize_prediction_io,policy

def main():
    p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--vendor',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threads',type=int,default=3);p.add_argument('--batch',type=int,default=2);a=p.parse_args()
    a.output,position_id=normalize_prediction_io(a.input,a.output)
    sel=c.read(a.model_dir/'selection.json');ck=a.model_dir/'best_weights_fp16.npz';assert c.sha(ck)==sel['checkpoint_SHA256']
    assert c.sha(a.vendor/'deeplab_model.py')=='ba0e931160d08519b9c8f3fe23c61b0678c4faa0b8818c38e04d533ef75ad361'
    sys.path.insert(0,str(a.vendor));import deeplab_model
    x=tf.placeholder(tf.float32,[None,256,256,3]);z=deeplab_model.deeplab_v3_plus_generator(2,16,'resnet_v2_50',None,.90)((x-.5)/.25,False)
    prob=tf.sigmoid(z[:,:,:,1]-z[:,:,:,0]);variables=tf.global_variables();ph=[tf.placeholder(tf.float32,v.shape) for v in variables];ops=[tf.assign(v,q) for v,q in zip(variables,ph)]
    sess=tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=a.threads,inter_op_parallelism_threads=1,allow_soft_placement=True));sess.run(tf.global_variables_initializer())
    names=c.read(a.model_dir/'weight_names.json');saved=np.load(str(ck));values={name:saved['v%04d'%i].astype(np.float32) for i,name in enumerate(names)}
    assert set(values)=={v.name for v in variables};sess.run(ops,{q:values[v.name] for v,q in zip(variables,ph)})
    def infer(ims):return sess.run(prob,{x:ims})
    records=predict_files(infer,a.input,a.output,sel['threshold'],a.batch)
    c.js(a.output/'prediction_provenance.json',dict(model='Original TensorFlow DeepLabV3+',position_id=position_id,dataset_identity_policy=policy()['version'],checkpoint_SHA256=c.sha(ck),CUDA=bool(tf.test.is_gpu_available()),
        threshold_selected_on='Original validation only; not adapted on new images',RGB_resized=False,foreground_RGB_unchanged=True,
        evaluation_on_new_images='Not measured without corresponding human labels',files=records))
    sess.close()

if __name__=='__main__':main()

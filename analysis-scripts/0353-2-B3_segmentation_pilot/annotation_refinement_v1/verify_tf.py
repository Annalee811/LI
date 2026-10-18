"""Fresh original TF1.12 inference graph, independently reloaded saved weights."""
from __future__ import print_function,division
import argparse,sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import common as c

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--vendor',type=Path,required=True);a=ap.parse_args()
    out=a.run/'deeplab';sel=c.read(out/'selection.json');assert c.sha(out/'best_weights_fp16.npz')==sel['checkpoint_SHA256']
    sys.path.insert(0,str(a.vendor));import deeplab_model
    x=tf.placeholder(tf.float32,[None,256,256,3]);z=deeplab_model.deeplab_v3_plus_generator(2,16,'resnet_v2_50',None,.90)((x-.5)/.25,False)
    prob=tf.sigmoid(z[:,:,:,1]-z[:,:,:,0]);vv=tf.global_variables();ph=[tf.placeholder(tf.float32,v.shape) for v in vv];ops=[tf.assign(v,q) for v,q in zip(vv,ph)]
    sess=tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads=1,allow_soft_placement=True));sess.run(tf.global_variables_initializer())
    names=c.read(out/'weight_names.json');ck=np.load(str(out/'best_weights_fp16.npz'));values={name:ck['v%04d'%i].astype(np.float32) for i,name in enumerate(names)}
    assert set(values)==set(v.name for v in vv);sess.run(ops,{q:values[v.name] for v,q in zip(vv,ph)})
    def infer(ims):return sess.run(prob,{x:ims})
    data=c.Data(a.run);records=[];recorded=c.read(out/'metrics.json')
    for split in ['val','test']:
        maps=data.predict(infer,split,2);delta=0.;changed=0
        for key,p in maps.items():
            previous=np.load(str(out/'probabilities'/split/(key+'_probability.npy')))
            delta=max(delta,float(abs(p-previous).max()));changed+=int(((p>=sel['threshold'])!=(previous>=sel['threshold'])).sum())
        m,_=data.evaluate(maps,sel['threshold']);assert changed==0
        assert all(m[k]==recorded[split][k] for k in ['TP','FP','FN','TN'])
        records.append(dict(split=split,largest_probability_difference=delta,changed_mask_pixels=changed,counts=m))
    c.js(a.run/'tensorflow_checkpoint_reload_verification.json',dict(passed=True,framework=tf.__version__,fresh_original_inference_graph=True,reports=records))
    print('ORIGINAL_TENSORFLOW_CHECKPOINT_RELOAD_PASSED',flush=True);sess.close()

if __name__=='__main__':main()

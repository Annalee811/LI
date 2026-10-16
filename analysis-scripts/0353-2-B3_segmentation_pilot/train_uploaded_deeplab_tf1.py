#!/usr/bin/env python
"""Train the user's original TensorFlow 1.12 DeepLabV3+ on the fixed T1/T2 tiles.
Run inside tensorflow/tensorflow:1.12.0-py3; no modern-framework replacement.
"""
from __future__ import print_function,division
import argparse,csv,hashlib,json,os,random,time,sys
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

SEED=20261016;SIDE=256;COLS=[22+512*i for i in range(5)]

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def js(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvout(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def confusion(gt,pred):
    valid=gt!=128;g=gt==255;p=pred.astype(bool)
    return np.array([(g&p&valid).sum(),(~g&p&valid).sum(),(g&~p&valid).sum(),(~g&~p&valid).sum()],np.int64)
def metric(c):
    tp,fp,fn,tn=map(int,c)
    return dict(TP=tp,FP=fp,FN=fn,TN=tn,Precision=tp/max(1,tp+fp),Recall=tp/max(1,tp+fn),Dice=2*tp/max(1,2*tp+fp+fn),IoU=tp/max(1,tp+fp+fn))

def source_module(vendor,out):
    original=(vendor/'deeplab_model.py').read_text()
    old='    if is_training:\n      exclude ='
    assert original.count(old)==1
    adapted=original.replace(old,'    if is_training and pre_trained_model:\n      exclude =')
    (out/'deeplab_model_adapted.py').write_text(adapted)
    sys.path.insert(0,str(vendor));sys.path.insert(0,str(out))
    import deeplab_model_adapted
    return deeplab_model_adapted


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset-run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--vendor',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=40);parser.add_argument('--patience',type=int,default=10);parser.add_argument('--batch',type=int,default=2);parser.add_argument('--threads',type=int,default=4);parser.add_argument('--lr',type=float,default=.0003);parser.add_argument('--benchmark',action='store_true');parser.add_argument('--evaluate-only',action='store_true');a=parser.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);random.seed(SEED);np.random.seed(SEED);tf.set_random_seed(SEED)
    with (a.dataset_run/'dataset/manifest.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    for r in rows:
        for k in ['column','x','y','width','height']:r[k]=int(r[k])
    assert len(rows)==100 and {s:sum(r['split']==s for r in rows) for s in ['train','val','test']}==dict(train=60,val=20,test=20)
    for r in rows:
        assert digest(a.dataset_run/r['images_path'])==r['image_sha256']
        assert digest(a.dataset_run/r['mask_path'])==r['mask_sha256']
    cache={}
    for r in rows:
        if r['split']=='test':continue
        cache[r['image_id']]=(np.array(Image.open(str(a.dataset_run/r['images_path'])).convert('RGB')),np.array(Image.open(str(a.dataset_run/r['mask_path'])).convert('L')))
    trainrows=[r for r in rows if r['split']=='train'];roots=sum((cache[r['image_id']][1]==255).sum() for r in trainrows);valid=sum((cache[r['image_id']][1]!=128).sum() for r in trainrows);posweight=min(6.,float(np.sqrt((valid-roots)/max(1,roots))))
    mod=source_module(a.vendor,a.output)
    x=tf.placeholder(tf.float32,[None,SIDE,SIDE,3],name='rgb_normalized');y=tf.placeholder(tf.float32,[None,SIDE,SIDE],name='human_root');v=tf.placeholder(tf.float32,[None,SIDE,SIDE],name='valid_pixels');lr=tf.placeholder(tf.float32,[],name='learning_rate')
    net=mod.deeplab_v3_plus_generator(2,16,'resnet_v2_50',None,.90)
    logits=net(x,True);modelvars=list(tf.global_variables());updates=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.variable_scope(tf.get_variable_scope(),reuse=True):
        test_logits=net(x,False)
    logit=logits[:,:,:,1]-logits[:,:,:,0];prob=tf.sigmoid(logit)
    bce=tf.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(targets=y,logits=logit,pos_weight=posweight)*v)/tf.maximum(tf.reduce_sum(v),1.)
    dice=1-(2*tf.reduce_sum(prob*y*v)+1)/(tf.reduce_sum(prob*v)+tf.reduce_sum(y*v)+1)
    loss=bce+dice
    with tf.control_dependencies(updates):
        optim=tf.train.AdamOptimizer(lr);grads=optim.compute_gradients(loss);grads=[(g,vr) for g,vr in grads if g is not None];gs,vs=zip(*grads);gs,_=tf.clip_by_global_norm(gs,5.);trainop=optim.apply_gradients(zip(gs,vs))
    prediction=tf.nn.softmax(test_logits)[:,:,:,1]
    sess=tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=a.threads,inter_op_parallelism_threads=1,allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())
    def batch(rr,augment):
        ims=[];gts=[];valids=[]
        for r in rr:
            im,gt=cache[r['image_id']];im=im.copy();gt=gt.copy()
            if augment:
                k=random.randrange(4);im=np.rot90(im,k);gt=np.rot90(gt,k)
                if random.random()<.5:im=np.fliplr(im);gt=np.fliplr(gt)
                if random.random()<.5:im=np.flipud(im);gt=np.flipud(gt)
                im=np.clip(im.astype(np.float32)/255*random.uniform(.9,1.1),0,1)**random.uniform(.9,1.1)
            else:im=im.astype(np.float32)/255
            ims.append(im.copy());gts.append((gt==255).copy());valids.append((gt!=128).copy())
        return {x:np.stack(ims),y:np.array(gts,np.float32),v:np.array(valids,np.float32)}
    def infer_images(ims):
        result=[]
        for i in range(0,len(ims),a.batch):
            bx=np.stack(ims[i:i+a.batch]).astype(np.float32)/255
            result.extend(sess.run(prediction,{x:bx}))
        return result
    def column_maps(split):
        rr=[r for r in rows if r['split']==split];ims=[]
        for r in rr:
            if r['image_id'] in cache:ims.append(cache[r['image_id']][0])
            else:ims.append(np.array(Image.open(str(a.dataset_run/r['images_path'])).convert('RGB')))
        probabilities=infer_images(ims);maps={}
        for r,p in zip(rr,probabilities):
            key=(r['period'],r['column'])
            if key not in maps:maps[key]=[np.zeros((1108,448),np.float32),np.zeros((1108,448),np.float32)]
            sm,n=maps[key];xx,yy=r['x']-COLS[r['column']],r['y'];sm[yy:yy+SIDE,xx:xx+SIDE]+=p;n[yy:yy+SIDE,xx:xx+SIDE]+=1
        result={}
        for (t,c),(sm,n) in maps.items():
            assert (n>0).all();gt=np.array(Image.open(str(a.dataset_run/'dataset/upper'/(t+'_mask.png'))).convert('L'))[:,COLS[c]:COLS[c]+448]
            result[t,c]=(sm/n,gt)
        return result
    def evaluate(maps,threshold):
        total=np.zeros(4,np.int64);per=[]
        for (t,c),(p,gt) in maps.items():
            cc=confusion(gt,p>=threshold);total+=cc;per.append(dict(period=t,column=c,threshold=threshold,**metric(cc)))
        return metric(total),per
    params=sum(int(np.prod(z.shape.as_list())) for z in modelvars)
    config=dict(model='Uploaded TensorFlow DeepLabV3+',implementation='User-uploaded Rishizek TensorFlow 1.12 source; actual generator and original ASPP/decoder',
                backbone='resnet_v2_50',num_classes=2,output_stride=16,aspp_depth=256,aspp_rates=[6,12,18],decoder_low_level_channels=48,decoder_channels=256,
                normalization='RGB/255, same as U-Net',batch_norm_decay=.90,from_scratch=True,pretrained_weights=None,seed=SEED,
                optimizer='TensorFlow Adam',learning_rate=a.lr,loss='weighted binary cross entropy + soft Dice; same positive weight as U-Net',positive_bce_weight=posweight,
                batch_size=a.batch,epochs_max=a.epochs,early_stopping_patience=a.patience,threads=a.threads,device='CPU',tensorflow_version=tf.__version__,gpu_available=bool(tf.test.is_gpu_available()),
                parameters_including_batch_norm_state=params,manifest_sha256=digest(a.dataset_run/'dataset/manifest.csv'),
                uploaded_deeplab_model_sha256=digest(a.vendor/'deeplab_model.py'),adapted_source_sha256=digest(a.output/'deeplab_model_adapted.py'),
                source_change='Only guard ImageNet checkpoint loading when no checkpoint is supplied; backbone/head definitions unchanged.',
                checkpoint_format='Portable NPZ float16 weights; restored to native float32 graph before final validation calibration and held-out evaluation',
                test_used_for_training_or_selection=False,classification='root/background only')
    js(a.output/'config.json',config)
    print('TF_MODEL_READY',json.dumps(dict(parameters=params,tensorflow=tf.__version__,gpu=config['gpu_available'])),flush=True)
    if a.benchmark:
        for i in range(3):
            tic=time.time();feed=batch(trainrows[:a.batch],True);feed[lr]=a.lr;lo,_=sess.run([loss,trainop],feed);print('BENCHMARK',i,float(lo),time.time()-tic,flush=True)
        return
    names=[z.name for z in modelvars];js(a.output/'weight_names.json',names)
    best=-1.;bad=0;hist=[];start=time.time()
    if a.evaluate_only:
        with (a.output/'history.csv').open(encoding='utf-8-sig') as f:hist=list(csv.DictReader(f))
    for epoch in ([] if a.evaluate_only else range(1,a.epochs+1)):
        tic=time.time();order=trainrows[:];random.shuffle(order);losses=[];rate=a.lr*(.05+.95*.5*(1+np.cos(np.pi*(epoch-1)/a.epochs)))
        for i in range(0,len(order),a.batch):
            feed=batch(order[i:i+a.batch],True);feed[lr]=rate;lo,_=sess.run([loss,trainop],feed);losses.append(float(lo))
        maps=column_maps('val');m,_=evaluate(maps,.5);hist.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_Dice=m['Dice'],validation_IoU=m['IoU'],validation_Precision=m['Precision'],validation_Recall=m['Recall'],learning_rate=float(rate),elapsed_seconds=time.time()-tic));csvout(a.output/'history.csv',hist)
        if m['Dice']>best+1e-5:
            best=m['Dice'];bad=0;values=sess.run(modelvars);np.savez_compressed(str(a.output/'best_weights_fp16.npz'),**{'v%04d'%i:val.astype(np.float16) for i,val in enumerate(values)})
            js(a.output/'best_epoch.json',dict(epoch=epoch,validation_Dice_at_05=best))
        else:bad+=1
        print('TF_EPOCH',epoch,'LOSS',round(float(np.mean(losses)),4),'VAL_DICE',round(m['Dice'],4),'SECONDS',round(time.time()-tic,1),flush=True)
        if bad>=a.patience:break
    # Reload the actual portable checkpoint before any final scoring.
    saved=np.load(str(a.output/'best_weights_fp16.npz'))
    restore_inputs=[tf.placeholder(tf.float32,var.shape,name='restore_weight_%d'%i) for i,var in enumerate(modelvars)]
    restore_ops=[tf.assign(var,value) for var,value in zip(modelvars,restore_inputs)]
    sess.run(restore_ops,{ph:saved['v%04d'%i].astype(np.float32) for i,ph in enumerate(restore_inputs)})
    print('CHECKPOINT_RESTORED',len(modelvars),flush=True)
    vm=column_maps('val');tr=[]
    for threshold in np.arange(.1,.901,.05):
        th=float(round(float(threshold),2));m,_=evaluate(vm,th);tr.append(dict(threshold=th,**m))
    chosen=max(tr,key=lambda r:(r['Dice'],-abs(r['threshold']-.5)));threshold=chosen['threshold'];csvout(a.output/'validation_thresholds.csv',tr)
    # Threshold is now frozen; first test prediction occurs here.
    reports={};per=[]
    for split,maps in [('val',vm),('test',column_maps('test'))]:
        m,rr=evaluate(maps,threshold)
        for r in rr:r['split']=split
        per.extend(rr);reports[split]=dict(micro_metrics=m,n_patch_images=20,n_physical_windows=10,n_spatial_columns=1,n_unique_evaluated_pixels=2*1108*448,threshold=threshold)
        folder=a.output/'predictions'/split;folder.mkdir(parents=True,exist_ok=True)
        for (t,c),(prob,gt) in maps.items():
            Image.fromarray((prob>=threshold).astype(np.uint8)*255).save(str(folder/(t+'_C%d_prediction.png'%c)))
            Image.fromarray(gt).save(str(folder/(t+'_C%d_human_mask.png'%c)))
            np.save(str(folder/(t+'_C%d_probability.npy'%c)),prob.astype(np.float16))
    csvout(a.output/'per_column_metrics.csv',per);js(a.output/'metrics.json',dict(model=config['model'],validation=reports['val'],spatial_holdout_test=reports['test'],comparison_target='User masks, not independent anatomical truth',independent_manual_root_length_agreement=None,scope='One position, T1/T2 upper region only; binary root segmentation'))
    # Native-size full upper predictions are qualitative (they include training pixels).
    full=a.output/'predictions/upper';full.mkdir(exist_ok=True)
    for t in ['T1','T2']:
        rgb=np.array(Image.open(str(a.dataset_run/'dataset/upper'/(t+'_RGB.png'))).convert('RGB'));h,w=rgb.shape[:2]
        xs=list(range(0,w-SIDE+1,192));ys=list(range(0,h-SIDE+1,192))
        if xs[-1]!=w-SIDE:xs.append(w-SIDE)
        if ys[-1]!=h-SIDE:ys.append(h-SIDE)
        coords=[(xx,yy) for yy in ys for xx in xs];val=np.zeros((h,w),np.float32);weight=np.zeros((h,w),np.float32);hann=np.hanning(SIDE).astype(np.float32);blend=np.maximum(hann[:,None]*hann[None,:],.02)
        for i in range(0,len(coords),a.batch):
            local=coords[i:i+a.batch];images=[rgb[yy:yy+SIDE,xx:xx+SIDE] for xx,yy in local]
            for (xx,yy),pr in zip(local,infer_images(images)):
                val[yy:yy+SIDE,xx:xx+SIDE]+=pr*blend;weight[yy:yy+SIDE,xx:xx+SIDE]+=blend
        assert (weight>0).all();prob=val/weight
        Image.fromarray((prob>=threshold).astype(np.uint8)*255).save(str(full/(t+'_upper_prediction_mask.png')));np.save(str(full/(t+'_upper_probability.npy')),prob.astype(np.float16))
    bestrec=json.loads((a.output/'best_epoch.json').read_text());run=dict(config);run.update(dict(epochs_completed=len(hist),best_epoch=bestrec['epoch'],best_validation_Dice_at_05=bestrec['validation_Dice_at_05'],frozen_threshold=threshold,recorded_training_seconds=sum(float(r['elapsed_seconds']) for r in hist),finalization_seconds=time.time()-start,checkpoint_sha256=digest(a.output/'best_weights_fp16.npz')));js(a.output/'training_run.json',run)
    print('TF_FINAL_METRICS',json.dumps(reports),flush=True);sess.close()

if __name__=='__main__':main()

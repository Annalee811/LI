#!/usr/bin/env python3
"""Fine-tune selected root architecture on paired RGB for actual temporal labels."""
from pathlib import Path
import argparse,csv,json,math,random,time
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
import train_uploaded_upper as core
from train_segformer_progress import RootSegFormer

CLASS_NAMES={0:'background',1:'persistent',2:'newly_observed',3:'elongating'}

def root_model(architecture,checkpoint):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=core.SmallUNet(ck['width']) if architecture=='unet' else RootSegFormer()
    model.load_state_dict(ck['state_dict']);return model

def paired_model(architecture,checkpoint):
    model=root_model(architecture,checkpoint)
    if architecture=='unet':first=model.enc[0][0];head=model.head
    else:first=model.net.segformer.encoder.patch_embeddings[0].proj;head=model.net.decode_head.classifier
    conv=nn.Conv2d(8,first.out_channels,first.kernel_size,first.stride,first.padding,bias=first.bias is not None)
    conv.weight.data.zero_();conv.weight.data[:,:6].copy_(torch.cat([first.weight.data*.5,first.weight.data*.5],1))
    if first.bias is not None:conv.bias.data.copy_(first.bias.data)
    classifier=nn.Conv2d(head.in_channels,4,1);classifier.weight.data.zero_();classifier.bias.data.zero_()
    for k in [1,2,3]:classifier.weight.data[k].copy_(head.weight.data[0]);classifier.bias.data[k]=head.bias.data[0]-math.log(3)
    if architecture=='unet':model.enc[0][0]=conv;model.head=classifier
    else:model.net.segformer.encoder.patch_embeddings[0].proj=conv;model.net.decode_head.classifier=classifier
    return model

def metrics(gt,pred):
    valid=gt!=255;rows=[]
    for k in [1,2,3]:
        g=(gt==k)&valid;p=(pred==k)&valid;tp=int((g&p).sum());fp=int((~g&p&valid).sum());fn=int((g&~p&valid).sum());support=int(g.sum())
        rows.append(dict(class_id=k,name=CLASS_NAMES[k],support_pixels=support,TP=tp,FP=fp,FN=fn,Precision=tp/(tp+fp) if tp+fp else None,Recall=tp/(tp+fn) if support else None,Dice=2*tp/(2*tp+fp+fn) if support else None,IoU=tp/(tp+fp+fn) if support else None,assessable=support>0))
    present=[r['Dice'] for r in rows if r['Dice'] is not None]
    return dict(per_class=rows,macro_Dice_supported_classes=float(np.mean(present)) if present else None,classes_with_ground_truth=len(present),meaning='Compared to user labels; absent classes are not reported as zero or perfect recognition')

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--segformer',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=80);a=p.parse_args()
    random.seed(core.SEED);np.random.seed(core.SEED);torch.manual_seed(core.SEED);torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');selection=json.loads(a.selection.read_text());arch=selection['winner'];assert arch in ['unet','segformer'],'DeepLab was already lower than U-Net on validation, cannot be selected'
    checkpoint=(a.dataset if arch=='unet' else a.segformer)/'training/best.pt'
    model=paired_model(arch,checkpoint).to(device);a.output.mkdir(parents=True,exist_ok=True)
    with (a.dataset/'dataset/manifest.csv').open(encoding='utf-8-sig') as f:rows=[r for r in csv.DictReader(f) if r['period']=='T2']
    for r in rows:
        for k in ['column','x','y']:r[k]=int(r[k])
    rgb={t:np.array(Image.open(a.dataset/f'dataset/upper/{t}_RGB.png').convert('RGB')) for t in ['T1','T2']}
    mask=np.array(Image.open(a.dataset/'dataset/upper/T2_mask.png'));temporal=np.array(Image.open(a.dataset/'dataset/upper/T2_temporal.png'))
    target=np.zeros(mask.shape,np.uint8);target[(mask==128)|((mask==255)&(temporal==0))]=255
    for source,label in [(3,1),(1,2),(2,3)]:target[(mask==255)&(temporal==source)]=label
    Image.fromarray(target).save(a.output/'T2_human_temporal_target.png')
    training=[r for r in rows if r['split']=='train'];freq=np.zeros(4,np.int64)
    for r in training:
        y,x=r['y'],r['x'];freq+=np.array([(target[y:y+256,x:x+256]==k).sum() for k in range(4)])
    weight=freq[1:].sum()/np.maximum(freq,1);weight[0]=0;weight[1:]=np.clip(weight[1:]/weight[1:].min(),1,20);weight=weight.astype(np.float32);tw=torch.tensor(weight,device=device)
    optim=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=a.epochs,eta_min=5e-5)
    # The previously frozen root model supplies the same root support to all temporal evaluations.
    frozen=next(r for r in selection['scores'] if r['model_id']==arch)['threshold']
    binary_path=a.dataset/'predictions' if arch=='unet' else a.segformer/'predictions'
    # Predicted root features are inferred within each guarded column, never across splits.
    baseline=root_model(arch,checkpoint).to(device).eval();root_features={t:np.zeros(mask.shape,np.float32) for t in ['T1','T2']}
    all_rows=[]
    with (a.dataset/'dataset/manifest.csv').open(encoding='utf-8-sig') as f:all_rows=list(csv.DictReader(f))
    for r in all_rows:
        for k in ['column','x','y']:r[k]=int(r[k])
    feature_arrays={t:(rgb[t],{'mask':np.array(Image.open(a.dataset/f'dataset/upper/{t}_mask.png'))}) for t in ['T1','T2']}
    for split in ['train','val','test']:
        maps=core.fused_columns(baseline,all_rows,feature_arrays,split,device)
        for (t,col),(pr,ignored_gt) in maps.items():root_features[t][:,core.COLS[col]:core.COLS[col]+448]=pr
    del baseline
    def batch(rr,augment):
        images=[];labels=[]
        for r in rr:
            y,x=r['y'],r['x'];im=np.concatenate([rgb['T1'][y:y+256,x:x+256].astype(np.float32)/255,rgb['T2'][y:y+256,x:x+256].astype(np.float32)/255,root_features['T1'][y:y+256,x:x+256,None],root_features['T2'][y:y+256,x:x+256,None]],2).copy();gt=target[y:y+256,x:x+256].copy()
            if augment:
                rot=random.randrange(4);im=np.rot90(im,rot);gt=np.rot90(gt,rot)
                if random.random()<.5:im=np.fliplr(im);gt=np.fliplr(gt)
                if random.random()<.5:im=np.flipud(im);gt=np.flipud(gt)
            images.append(im.transpose(2,0,1).copy());labels.append(gt.copy())
        return torch.from_numpy(np.stack(images)).to(device).float(),torch.from_numpy(np.stack(labels).astype(np.int64)).to(device)
    def infer_split(split):
        selected=[r for r in rows if r['split']==split];total=np.zeros((4,1108,448),np.float32);count=np.zeros((1108,448),np.float32);col=3 if split=='val' else 4
        model.eval()
        with torch.inference_mode():
            for i in range(0,len(selected),4):
                rr=selected[i:i+4];xx,_=batch(rr,False);probs=torch.softmax(model(xx),1).cpu().numpy()
                for r,prob in zip(rr,probs):
                    x,y=r['x']-core.COLS[col],r['y'];total[:,y:y+256,x:x+256]+=prob;count[y:y+256,x:x+256]+=1
        assert (count>0).all();prob=total/count[None];root=np.array(Image.open(binary_path/split/f'T2_C{col}_prediction.png'))==255
        pred=np.where(root,prob[1:].argmax(0)+1,0).astype(np.uint8);gt=target[:,core.COLS[col]:core.COLS[col]+448]
        return metrics(gt,pred),pred,prob,gt
    config=dict(architecture=arch,input='8 channels: previous RGB/255, current RGB/255, frozen previous and current root probabilities; features remain inside their spatial splits',outputs=CLASS_NAMES,root_support='Frozen selected binary root model; temporal classification loss is conditional on manually labelled root pixels only',initialization_checkpoint_sha256=core.sha(checkpoint),selection_rule='Maximum validation macro Dice over all 3 foreground classes',seed=core.SEED,device=str(device),train_pairs=30,validation_pairs=10,spatial_holdout_pairs=10,class_weights=weight.tolist(),epochs_max=a.epochs,patience=20,learning_rate=1e-3,test_used_for_selection=False,full_reviewed_training_transition='T1 to T2 only',T3_T4_unreviewed_labels_used=False,manifest_sha256=selection['manifest_sha256'])
    core.save_json(a.output/'config.json',config);best=-1;bad=0;history=[];start=time.time()
    for epoch in range(1,a.epochs+1):
        model.train();order=training.copy();random.shuffle(order);losses=[]
        for i in range(0,len(order),4):
            xx,yy=batch(order[i:i+4],True);optim.zero_grad(set_to_none=True);logits=model(xx);root_target=yy.clone();root_target[yy==0]=255;ce=F.cross_entropy(logits,root_target,weight=tw,ignore_index=255)
            conditional=torch.softmax(logits[:,1:],1);prob=torch.cat([torch.zeros_like(conditional[:,:1]),conditional],1);valid=((yy>0)&(yy!=255)).float();dices=[]
            for k in [1,2,3]:
                g=(yy==k).float();pr=prob[:,k]*valid;dices.append(1-(2*(pr*g).sum()+1)/(pr.sum()+g.sum()+1))
            loss=ce+torch.stack(dices).mean();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optim.step();losses.append(float(loss.detach()))
        metric,_,_,_=infer_split('val');score=metric['macro_Dice_supported_classes'];history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_macro_Dice=score));core.write_csv(a.output/'history.csv',history)
        if score>best+1e-5:
            best=score;bad=0;torch.save(dict(state_dict=model.state_dict(),architecture=arch,epoch=epoch,validation_macro_Dice=score,config=config),a.output/'best.pt')
        else:bad+=1
        scheduler.step();print('TEMPORAL_EPOCH',epoch,'VAL_MACRO_DICE',round(score,4),flush=True)
        if bad>=20 and epoch>=40:break
    checkpoint=torch.load(a.output/'best.pt',map_location=device,weights_only=True);model.load_state_dict(checkpoint['state_dict']);report={}
    for split in ['val','test']:
        metric,pred,prob,gt=infer_split(split);report[split]=metric;folder=a.output/split;folder.mkdir(exist_ok=True)
        Image.fromarray(pred).save(folder/'T2_class_prediction.png');Image.fromarray(gt).save(folder/'T2_human_target.png');np.save(folder/'T2_class_probability.npy',prob.astype(np.float16))
    core.save_json(a.output/'metrics.json',report);core.save_json(a.output/'training_run.json',dict(**config,epochs_completed=len(history),best_epoch=checkpoint['epoch'],best_validation_macro_Dice=best,total_seconds=time.time()-start,checkpoint_sha256=core.sha(a.output/'best.pt')))
    print('TEMPORAL_DONE',json.dumps(report),flush=True)
if __name__=='__main__':main()

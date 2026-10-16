#!/usr/bin/env python3
"""Class-balanced native-pixel temporal training. Human labels remain unchanged."""
from pathlib import Path
import argparse,csv,json,random,time
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
import train_uploaded_upper as core
from train_pair_temporal import paired_model,metrics
from train_sequential_joint import features,infer_tiles,target_map

def make_temporal(binary,previous_temporal):
    model=paired_model('unet',binary);model.load_state_dict(torch.load(previous_temporal,map_location='cpu',weights_only=True)['state_dict']);first=model.enc[0][0]
    conv=nn.Conv2d(10,first.out_channels,3,padding=1,bias=False);conv.weight.data.zero_();conv.weight.data[:,:8].copy_(first.weight.data);model.enc[0][0]=conv
    model.head=nn.Conv2d(12,4,1)
    # Head zero is deliberately unused. Reset all conditional classes equally.
    model.head.weight.data[0].zero_();model.head.bias.data[0]=0
    return model

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--previous-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=100);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    random.seed(core.SEED+1);np.random.seed(core.SEED+1);torch.manual_seed(core.SEED+1);torch.set_num_threads(4);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    binary=a.dataset/'training/best.pt';prior=a.previous_run/'temporal/best.pt';model=make_temporal(binary,prior).to(device)
    with (a.dataset/'dataset/manifest.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    for r in rows:
        for k in ['column','x','y']:r[k]=int(r[k])
    rgb={t:np.array(Image.open(a.dataset/f'dataset/upper/{t}_RGB.png')) for t in ['T1','T2']};masks={t:np.array(Image.open(a.dataset/f'dataset/upper/{t}_mask.png')) for t in rgb};target=target_map(masks['T2'],np.array(Image.open(a.dataset/'dataset/upper/T2_temporal.png')))
    baseline=core.SmallUNet(12).to(device);baseline.load_state_dict(torch.load(binary,map_location=device,weights_only=True)['state_dict']);baseline.eval();arrays={t:(rgb[t],{'mask':masks[t]}) for t in rgb};seeds={}
    for split in ['train','val','test']:
        for (t,col),(prob,_) in core.fused_columns(baseline,rows,arrays,split,device).items():seeds[t,col]=prob
    del baseline
    inputs={};pools={k:[] for k in [1,2,3]}
    for col in range(5):
        sl=slice(core.COLS[col],core.COLS[col]+448);manual=(masks['T1'][:,sl]==255).astype(np.float32)
        inputs[col,'manual']=features(rgb['T1'][:,sl],rgb['T2'][:,sl],manual,seeds['T2',col]);inputs[col,'predicted']=features(rgb['T1'][:,sl],rgb['T2'][:,sl],seeds['T1',col],seeds['T2',col])
        if col<3:
            for k in [1,2,3]:
                yy,xx=np.where(target[:,sl]==k);pools[k].extend([(col,int(y),int(x)) for y,x in zip(yy,xx)])
    weights=np.sqrt(np.array([len(pools[1])/len(pools[k]) for k in [1,2,3]],np.float32));weights=np.maximum(weights,1);cw=torch.tensor(weights,device=device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=a.epochs,eta_min=2e-5)
    config={'model':'U-Net conditional temporal classifier, 10 inputs, three conditional outputs','unused_head_zero':'not a segmentation output; ignored','input_channels':'previous/current RGB, previous predicted-or-manual root support, current frozen root seed, previous tip/branch cues','sampling':'60 native 128x128 crops per epoch, class-centred in turn, 20 centres for each class; centre jitter +/-24px; never leave the three guarded training columns','loss':'centre-weighted conditional cross entropy with sqrt inverse frequency + foreground three-class Dice','label_modification':False,'mask_dilation':False,'resize':False,'previous_reference_during_training':'50% manual T1 and 50% predicted T1; no target T2 label enters input','evaluation_reference':'manual T1 root support, current predicted seed; T2 labels are target only','training_unique_class_pixels':{str(k):len(v) for k,v in pools.items()},'class_weights':weights.tolist(),'learning_rate':3e-4,'seed':core.SEED+1,'epochs_max':a.epochs,'selection_rule':'validation three-class macro Dice with frozen single-frame root mask threshold 0.7','T3_T4_training':False,'death_browning_training':False,'manifest_sha256':core.sha(a.dataset/'dataset/manifest.csv'),'device':str(device),'binary_initialization_sha256':core.sha(binary),'temporal_encoder_initialization_sha256':core.sha(prior),'exploratory':'Additional candidate after the first joint experiment still missed the rare elongating class; same validation split reused, not independent confirmation'}
    core.save_json(a.output/'config.json',config)
    def evaluate(col):
        coords=[(r['x']-core.COLS[col],r['y']) for r in rows if r['column']==col and r['period']=='T2'];pr=infer_tiles(model,inputs[col,'manual'],device,coords)[1:];root=seeds['T2',col]>=.7;pred=np.where(root,pr.argmax(0)+1,0).astype(np.uint8);sl=slice(core.COLS[col],core.COLS[col]+448)
        return metrics(target[:,sl],pred),pred,pr
    best=-1;bad=0;history=[];start=time.time()
    for epoch in range(1,a.epochs+1):
        model.train();losses=[];classes=([1,2,3]*20);random.shuffle(classes)
        for offset in range(0,60,4):
            ims=[];labels=[];focus=[]
            for k in classes[offset:offset+4]:
                col,y,x=random.choice(pools[k]);left=max(0,min(448-128,x-64+random.randint(-24,24)));top=max(0,min(1108-128,y-64+random.randint(-24,24)));mode='manual' if random.random()<.5 else 'predicted';im=inputs[col,mode][top:top+128,left:left+128].copy();gt=target[top:top+128,core.COLS[col]+left:core.COLS[col]+left+128].copy();yy,xx=np.mgrid[:128,:128];weight=np.exp(-((xx-(x-left))**2+(yy-(y-top))**2)/(2*32**2)).astype(np.float32)
                rotation=random.randrange(4);im=np.rot90(im,rotation);gt=np.rot90(gt,rotation);weight=np.rot90(weight,rotation)
                if random.random()<.5:im=np.fliplr(im);gt=np.fliplr(gt);weight=np.fliplr(weight)
                if random.random()<.5:im=np.flipud(im);gt=np.flipud(gt);weight=np.flipud(weight)
                im[:,:,:6]=np.clip(im[:,:,:6]*random.uniform(.95,1.05),0,1);ims.append(im.transpose(2,0,1).copy());labels.append(gt.copy());focus.append(weight.copy())
            xx=torch.from_numpy(np.stack(ims)).to(device);gt=torch.from_numpy(np.stack(labels).astype(np.int64)).to(device);weight=torch.from_numpy(np.stack(focus)).to(device);valid=(gt>0)&(gt<255);target_class=gt.clone()-1;target_class[~valid]=255
            optimizer.zero_grad(set_to_none=True);logits=model(xx)[:,1:];ce=F.cross_entropy(logits,target_class,weight=cw,ignore_index=255,reduction='none');ce=(ce*weight*valid).sum()/(weight*valid).sum().clamp_min(1);pr=torch.softmax(logits,1);dices=[]
            for k in [1,2,3]:
                truth=(gt==k).float();prob=pr[:,k-1]*valid;dices.append(1-(2*(prob*truth*weight).sum()+1)/((prob*weight).sum()+(truth*weight).sum()+1))
            loss=ce+torch.stack(dices).mean();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append(float(loss.detach()))
        metric,_,_=evaluate(3);score=metric['macro_Dice_supported_classes'];row={'epoch':epoch,'loss':float(np.mean(losses)),'validation_macro_Dice':score,'persistent_Dice':metric['per_class'][0]['Dice'],'new_Dice':metric['per_class'][1]['Dice'],'elongating_Dice':metric['per_class'][2]['Dice']};history.append(row);core.write_csv(a.output/'history.csv',history)
        if score>best+1e-5:
            best=score;bad=0;torch.save({'state_dict':model.state_dict(),'epoch':epoch,'config':config},a.output/'best.pt')
        else:bad+=1
        scheduler.step();print('BALANCED_EPOCH',epoch,'MACRO',round(score,4),'ELONGATING',round(row['elongating_Dice'],4),flush=True)
        if epoch>=60 and bad>=25:break
    ck=torch.load(a.output/'best.pt',map_location=device,weights_only=True);model.load_state_dict(ck['state_dict']);report={}
    for split,col in [('val',3),('test',4)]:
        metric,pred,pr=evaluate(col);report[split]=metric;folder=a.output/split;folder.mkdir(exist_ok=True);Image.fromarray(pred).save(folder/'T2_class_prediction.png');np.save(folder/'T2_conditional_probability.npy',pr.astype(np.float32))
    core.save_json(a.output/'metrics.json',report);core.save_json(a.output/'training_run.json',{**config,'epochs_completed':len(history),'best_epoch':ck['epoch'],'seconds':time.time()-start,'checkpoint_sha256':core.sha(a.output/'best.pt')});print('BALANCED_DONE',json.dumps(report),flush=True)

if __name__=='__main__':main()

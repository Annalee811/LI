#!/usr/bin/env python3
"""Exploratory adjacent-period joint segmentation, without changing human labels.

Previous predicted root support, tip and branch cues are inputs, never generated truth.
Training and evaluation use the original guarded columns; later periods are excluded.
"""
from pathlib import Path
import argparse,csv,json,random,time
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize,remove_small_objects
import torch
from torch import nn
from torch.nn import functional as F
import train_uploaded_upper as core
from train_pair_temporal import metrics as class_metrics

ENCODING={0:'background',1:'persistent previous root segment',2:'newly visible root segment, including new lateral roots',3:'elongation from previous root tip',4:'T1 baseline root',255:'undetermined'}

def topology(probability):
    support=remove_small_objects(probability>=.5,min_size=12)
    skeleton=skeletonize(support);neighbors=ndi.convolve(skeleton.astype(np.uint8),np.ones((3,3),np.uint8),mode='constant')-skeleton
    tips=skeleton&(neighbors==1);branches=skeleton&(neighbors>=3)
    # Smooth distance cues tolerate small displacement; no hard anatomical decisions.
    maps=[]
    for points in [tips,branches]:
        maps.append(np.exp(-ndi.distance_transform_edt(~points)**2/(2*10.**2)).astype(np.float32) if points.any() else np.zeros(probability.shape,np.float32))
    return maps

def make_model(binary_checkpoint,temporal_checkpoint=None):
    ck=torch.load(binary_checkpoint,map_location='cpu',weights_only=True);model=core.SmallUNet(ck['width']);model.load_state_dict(ck['state_dict'])
    first=model.enc[0][0];conv=nn.Conv2d(10,first.out_channels,3,padding=1,bias=False);conv.weight.data.zero_()
    # Preserve the single-frame predictor exactly before fitting; previous RGB learns a residual.
    conv.weight.data[:,3:6].copy_(first.weight.data);model.enc[0][0]=conv
    head=model.head;model.head=nn.Conv2d(head.in_channels,4,1);model.head.weight.data[0].copy_(head.weight.data[0]);model.head.bias.data[0]=head.bias.data[0]
    if temporal_checkpoint:
        time_ck=torch.load(temporal_checkpoint,map_location='cpu',weights_only=True)['state_dict']
        model.head.weight.data[1:].copy_(time_ck['head.weight'][1:]);model.head.bias.data[1:].copy_(time_ck['head.bias'][1:])
    return model

def features(previous_rgb,current_rgb,previous_probability,current_seed):
    h,w=current_rgb.shape[:2]
    if previous_rgb is None:
        previous_rgb=np.zeros_like(current_rgb);previous_probability=np.zeros((h,w),np.float32);tips=branches=previous_probability
    else:tips,branches=topology(previous_probability)
    return np.concatenate([previous_rgb.astype(np.float32)/255,current_rgb.astype(np.float32)/255,previous_probability[...,None],current_seed[...,None],tips[...,None],branches[...,None]],2)

def infer_tiles(model,image,device,coordinates=None,blend=False):
    h,w=image.shape[:2]
    if coordinates is None:
        xs=list(range(0,w-255,192));ys=list(range(0,h-255,192))
        if xs[-1]!=w-256:xs.append(w-256)
        if ys[-1]!=h-256:ys.append(h-256)
        coordinates=[(x,y) for y in ys for x in xs]
    total=np.zeros((4,h,w),np.float32);count=np.zeros((h,w),np.float32)
    v=np.hanning(256).astype(np.float32);weight=np.maximum(v[:,None]*v[None,:],.02) if blend else np.ones((256,256),np.float32)
    model.eval()
    with torch.inference_mode():
        for offset in range(0,len(coordinates),4):
            rr=coordinates[offset:offset+4];xx=torch.from_numpy(np.stack([image[y:y+256,x:x+256].transpose(2,0,1) for x,y in rr]).copy()).to(device)
            logits=model(xx);pr=torch.cat([torch.sigmoid(logits[:,:1]),torch.softmax(logits[:,1:],1)],1).cpu().numpy()
            for (x,y),p in zip(rr,pr):total[:,y:y+256,x:x+256]+=p*weight;count[y:y+256,x:x+256]+=weight
    assert (count>0).all();return total/count[None]

def target_map(mask,temporal):
    out=np.zeros(mask.shape,np.uint8);out[(mask==128)|((mask==255)&(temporal==0))]=255
    for source,label in [(3,1),(1,2),(2,3)]:out[(mask==255)&(temporal==source)]=label
    return out

def run():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--previous-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=100);a=p.parse_args()
    random.seed(core.SEED);np.random.seed(core.SEED);torch.manual_seed(core.SEED);torch.set_num_threads(4);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');a.output.mkdir(parents=True,exist_ok=True)
    with (a.dataset/'dataset/manifest.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    for r in rows:
        for k in ['x','y','column']:r[k]=int(r[k])
    rgb={t:np.array(Image.open(a.dataset/f'dataset/upper/{t}_RGB.png')) for t in ['T1','T2']}
    masks={t:np.array(Image.open(a.dataset/f'dataset/upper/{t}_mask.png')) for t in ['T1','T2']}
    temporal=np.array(Image.open(a.dataset/'dataset/upper/T2_temporal.png'));target=target_map(masks['T2'],temporal)
    binary=a.dataset/'training/best.pt';model=make_model(binary,a.previous_run/'temporal/best.pt').to(device)
    baseline=core.SmallUNet(12).to(device);baseline.load_state_dict(torch.load(binary,map_location=device,weights_only=True)['state_dict']);baseline.eval()
    arrays={t:(rgb[t],{'mask':masks[t]}) for t in rgb};seeds={}
    # No held-out target enters fitting: feature extraction is unsupervised and confined to a column.
    for split in ['train','val','test']:
        for (t,col),(prob,_) in core.fused_columns(baseline,rows,arrays,split,device).items():seeds[t,col]=prob
    del baseline
    inputs={};train_rows=[r for r in rows if r['split']=='train'];training_columns=[0,1,2]
    for col in range(5):
        sl=slice(core.COLS[col],core.COLS[col]+448)
        inputs['T1',col]=features(None,rgb['T1'][:,sl],None,seeds['T1',col])
        inputs['T2',col]=features(rgb['T1'][:,sl],rgb['T2'][:,sl],seeds['T1',col],seeds['T2',col])
    train_support=np.zeros(4,np.int64)
    for col in training_columns:
        sl=slice(core.COLS[col],core.COLS[col]+448)
        for k in [1,2,3]:train_support[k]+=(target[:,sl]==k).sum()
    class_weight=np.sqrt(train_support[1:].max()/np.maximum(train_support[1:],1)).astype(np.float32);cw=torch.tensor(class_weight,device=device)
    config={'architecture':'10-input joint U-Net, root sigmoid plus three-class conditional softmax','inputs':['previous RGB (3)','current RGB (3)','previous predicted root probability','current frozen root probability cue','previous predicted tip distance cue','previous predicted branch distance cue'],'outputs':ENCODING,'semantics':{'baseline':'All T1 roots already exist at baseline','persistent':'Previously visible segment still visible this period','new':'New lateral root; old parent segment remains persistent','elongating':'New segment beyond previous root tip; old segment remains persistent'},'labels_auto_modified':False,'T3_T4_used_for_training':False,'death_and_browning_classes':False,'manifest_sha256':core.sha(a.dataset/'dataset/manifest.csv'),'binary_initialization_sha256':core.sha(binary),'temporal_head_initialization_sha256':core.sha(a.previous_run/'temporal/best.pt'),'seed':core.SEED,'device':str(device),'epochs_max':a.epochs,'learning_rate':3e-4,'class_weights':class_weight.tolist(),'train_class_pixels_unique':train_support.tolist(),'checkpoint_rule':'maximum arithmetic mean of validation two-period root Dice and validation T2 three-class macro Dice at root threshold 0.5','threshold_selection':'validation only, maximizing two-period root Dice after checkpoint frozen','test_used_for_model_selection':False,'refresh_previous_prediction':'Every 5 epochs, recompute T1 predictions on training columns only; T2 receives predicted T1, not manual T1 mask','validation_recurrence':'T1 single-frame joint prediction is fed to T2; no manual prior mask','exploratory_limit':'Same validation and spatial holdout as prior experiments; not a newly independent confirmatory study'}
    core.save_json(a.output/'config.json',config);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=a.epochs,eta_min=2e-5)
    def column_prediction(col):
        selected=[r for r in rows if r['column']==col and r['period']=='T1'];coordinates=[(r['x']-core.COLS[col],r['y']) for r in selected]
        first=infer_tiles(model,inputs['T1',col],device,coordinates);sl=slice(core.COLS[col],core.COLS[col]+448)
        second_input=features(rgb['T1'][:,sl],rgb['T2'][:,sl],first[0],seeds['T2',col]);second=infer_tiles(model,second_input,device,coordinates)
        return {'T1':first,'T2':second}
    def evaluate(maps,col,threshold):
        sl=slice(core.COLS[col],core.COLS[col]+448);cm=sum((core.confusion(masks[t][:,sl],maps[t][0]>=threshold) for t in ['T1','T2']),np.zeros(4,np.int64));pred=np.where(maps['T2'][0]>=threshold,maps['T2'][1:].argmax(0)+1,0).astype(np.uint8)
        return {'root_micro':core.metrics(cm),'root_by_period':{t:core.metrics(core.confusion(masks[t][:,sl],maps[t][0]>=threshold)) for t in ['T1','T2']},'temporal':class_metrics(target[:,sl],pred)}
    best=-1;bad=0;history=[];start=time.time()
    for epoch in range(1,a.epochs+1):
        if epoch>1 and (epoch-1)%5==0:
            for col in training_columns:
                selected=[r for r in rows if r['column']==col and r['period']=='T1'];coords=[(r['x']-core.COLS[col],r['y']) for r in selected];first=infer_tiles(model,inputs['T1',col],device,coords);sl=slice(core.COLS[col],core.COLS[col]+448)
                inputs['T2',col]=features(rgb['T1'][:,sl],rgb['T2'][:,sl],first[0],seeds['T2',col])
        model.train();order=train_rows.copy();random.shuffle(order);losses=[]
        for i in range(0,len(order),4):
            ims=[];roots=[];valids=[];times=[]
            for r in order[i:i+4]:
                t,col=r['period'],r['column'];y,x=r['y'],r['x'];xx=x-core.COLS[col];im=inputs[t,col][y:y+256,xx:xx+256].copy();mask=masks[t][y:y+256,x:x+256];rt=(mask==255).astype(np.float32);valid=(mask!=128).astype(np.float32);tt=target[y:y+256,x:x+256].copy() if t=='T2' else np.full((256,256),255,np.uint8)
                k=random.randrange(4);im=np.rot90(im,k);rt=np.rot90(rt,k);valid=np.rot90(valid,k);tt=np.rot90(tt,k)
                if random.random()<.5:im=np.fliplr(im);rt=np.fliplr(rt);valid=np.fliplr(valid);tt=np.fliplr(tt)
                if random.random()<.5:im=np.flipud(im);rt=np.flipud(rt);valid=np.flipud(valid);tt=np.flipud(tt)
                # Modest acquisition variation affects RGB only, never target width/position.
                im[:,:,:6]=np.clip(im[:,:,:6]*random.uniform(.95,1.05),0,1)
                ims.append(im.transpose(2,0,1).copy());roots.append(rt.copy());valids.append(valid.copy());times.append(tt.copy())
            xx=torch.from_numpy(np.stack(ims)).to(device);yy=torch.from_numpy(np.stack(roots)).to(device);vv=torch.from_numpy(np.stack(valids)).to(device);tt=torch.from_numpy(np.stack(times).astype(np.int64)).to(device)
            optimizer.zero_grad(set_to_none=True);logits=model(xx);prob=torch.sigmoid(logits[:,0]);bce=(F.binary_cross_entropy_with_logits(logits[:,0],yy,reduction='none',pos_weight=torch.tensor(3.,device=device))*vv).sum()/vv.sum().clamp_min(1);dice=1-(2*(prob*yy*vv).sum()+1)/((prob*vv).sum()+(yy*vv).sum()+1)
            foreground=(tt>0)&(tt<255);time_loss=logits.sum()*0
            if foreground.any():
                labels=tt.clone()-1;labels[~foreground]=255;ce=F.cross_entropy(logits[:,1:],labels,weight=cw,ignore_index=255);conditional=torch.softmax(logits[:,1:],1);dices=[]
                for k in [1,2,3]:
                    g=(tt==k).float();pr=conditional[:,k-1]*foreground;dices.append(1-(2*(pr*g).sum()+1)/(pr.sum()+g.sum()+1))
                time_loss=ce+torch.stack(dices).mean()
            loss=bce+dice+time_loss;loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append(float(loss.detach()))
        maps=column_prediction(3);metric=evaluate(maps,3,.5);score=(metric['root_micro']['Dice']+metric['temporal']['macro_Dice_supported_classes'])/2
        record={'epoch':epoch,'loss':float(np.mean(losses)),'validation_root_Dice':metric['root_micro']['Dice'],'validation_temporal_macro_Dice':metric['temporal']['macro_Dice_supported_classes'],'validation_elongating_Dice':metric['temporal']['per_class'][2]['Dice'],'selection_score':score};history.append(record);core.write_csv(a.output/'history.csv',history)
        if score>best+1e-5:
            best=score;bad=0;torch.save({'state_dict':model.state_dict(),'width':12,'epoch':epoch,'config':config,'selection_score':score},a.output/'best.pt')
        else:bad+=1
        scheduler.step();print('JOINT_EPOCH',epoch,'ROOT',round(metric['root_micro']['Dice'],4),'TEMPORAL',round(metric['temporal']['macro_Dice_supported_classes'],4),'ELONGATING',round(metric['temporal']['per_class'][2]['Dice'],4),flush=True)
        if bad>=25 and epoch>=50:break
    ck=torch.load(a.output/'best.pt',map_location=device,weights_only=True);model.load_state_dict(ck['state_dict']);vm=column_prediction(3);thresholds=[]
    for threshold in np.arange(.1,.901,.05):
        threshold=float(round(threshold,2));result=evaluate(vm,3,threshold);thresholds.append({'threshold':threshold,**result['root_micro']})
    chosen=max(thresholds,key=lambda r:(r['Dice'],-abs(r['threshold']-.5)));threshold=chosen['threshold'];core.write_csv(a.output/'validation_thresholds.csv',thresholds)
    reports={}
    for split,col in [('val',3),('test',4)]:
        maps=vm if split=='val' else column_prediction(col);reports[split]=evaluate(maps,col,threshold);folder=a.output/split;folder.mkdir(exist_ok=True)
        for t,prob in maps.items():
            np.save(folder/f'{t}_joint_probability.npy',prob.astype(np.float16));Image.fromarray((prob[0]>=threshold).astype(np.uint8)*255).save(folder/f'{t}_root_prediction.png')
            if t=='T2':Image.fromarray(np.where(prob[0]>=threshold,prob[1:].argmax(0)+1,0).astype(np.uint8)).save(folder/'T2_temporal_prediction.png')
    core.save_json(a.output/'metrics.json',reports);core.save_json(a.output/'training_run.json',{**config,'epochs_completed':len(history),'best_epoch':ck['epoch'],'best_validation_selection_score':best,'frozen_root_threshold':threshold,'seconds':time.time()-start,'checkpoint_sha256':core.sha(a.output/'best.pt')});print('JOINT_DONE',json.dumps(reports),flush=True)

if __name__=='__main__':run()

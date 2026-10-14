#!/usr/bin/env python3
"""Human-labelled, spatially held-out root segmentation pilot. No pseudo ground truth."""
from pathlib import Path
import argparse, csv, hashlib, json, math, random, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
PHASES = {1:'苗期',2:'抽穗期',3:'成熟期',4:'枯黄期'}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fields=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def save_json(path, obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def prepare(args):
    src=args.source; kit=args.output; kit.mkdir(parents=True,exist_ok=True)
    audit=json.loads(args.audit.read_text()); assert audit['order']==['A','B']
    sha=json.loads((src/'SHA256.json').read_text()); arrays={}
    for band in ['A','B']:
        assert not audit['region_manual_review'][band]
        for t in range(1,5):
            f=src/f'T{t}_region_{band}_RGB.png';assert digest(f)==sha[f.name]
            arrays[band,t]=np.array(Image.open(f).convert('RGB'))
    side=448;guard=64;common_x0=max(b[0] for b in audit['regions_T1_xyxy'].values())
    common_x1=min(b[2] for b in audit['regions_T1_xyxy'].values())
    left=common_x0+(common_x1-common_x0-(5*side+4*guard))//2
    columns=list(range(5));random.Random(20261014).shuffle(columns)
    assignment={c:('train' if c in columns[:3] else 'val' if c==columns[3] else 'test') for c in columns}
    rows=[]; windows=[]; images=kit/'images';images.mkdir(exist_ok=True)
    for band in ['A','B']:
        bx,by,bxx,byy=audit['regions_T1_xyxy'][band];nr=2 if band=='A' else 1
        top=by+(byy-by-(nr*side+(nr-1)*guard))//2
        for rr in range(nr):
            for cc in range(5):
                x=left+cc*(side+guard);y=top+rr*(side+guard)
                window=f'B3_{band}{rr}_C{cc}';windows.append(dict(window_id=window,band=band,row=rr,column=cc,split=assignment[cc],T1_xyxy=[x,y,x+side,y+side]))
                for t in range(1,5):
                    image_id=f'{window}_T{t}';p=images/(image_id+'.png');rgb=arrays[band,t][y-by:y-by+side,x-bx:x-bx+side]
                    assert rgb.shape==(side,side,3);Image.fromarray(rgb).save(p,dpi=(600,600))
                    np.testing.assert_array_equal(np.array(Image.open(p)),rgb)
                    rows.append(dict(image_id=image_id,image_path='images/'+p.name,image_sha256=digest(p),source_crop=f'T{t}_region_{band}_RGB.png',source_crop_sha256=sha[f'T{t}_region_{band}_RGB.png'],period=t,band=band,window_id=window,spatial_group=f'C{cc}',split=assignment[cc],T1_x=x,T1_y=y,crop_x=x-bx,crop_y=y-by,width=side,height=side))
    assert len(rows)==60 and {s:sum(x['split']==s for x in rows) for s in ['train','val','test']}=={'train':36,'val':12,'test':12}
    for w in windows:
        assert len({x['split'] for x in rows if x['window_id']==w['window_id']})==1
    for i,a in enumerate(windows):
        for b in windows[i+1:]:
            if a['split']==b['split']:continue
            xa,ya,aa,ba=a['T1_xyxy'];xb,yb,ab,bb=b['T1_xyxy']
            dx=max(xa-xb-(ab-xb),xb-xa-(aa-xa),0);dy=max(ya-yb-(bb-yb),yb-ya-(ba-ya),0)
            assert max(dx,dy)>=guard
    write_csv(kit/'split_manifest.csv',rows)
    write_csv(kit/'manual_lengths.csv',[dict(image_id=x['image_id'],manual_length_px='',method='',annotator='',fully_traced='',mm_per_pixel='') for x in rows])
    write_csv(kit/'quality_tags.csv',[dict(image_id=x['image_id'],water_dense='',specular_reflection='',low_root_soil_contrast='',fine_roots='',thick_roots='',brown_roots='',notes='') for x in rows])
    write_csv(kit/'root_tracking.csv',[],['window_id','root_id','period','first_observed','elongating','persistent','brown','suspected_dead','confirmed_dead','death_evidence','not_observed','manual_length_px','correspondence_confidence','annotator','notes'])
    font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font=lambda n:ImageFont.truetype(font_path,n);views=kit/'views';views.mkdir(exist_ok=True)
    for t in range(1,5):
        canvas=Image.new('RGB',(5*250,100+3*290),'white');draw=ImageDraw.Draw(canvas)
        draw.text((15,12),f'0353-2-B3 T{t} {PHASES[t]}：15个待人工标注观察窗',font=font(26),fill='#24343c')
        draw.text((15,56),'预览缩小显示；独立RGB为原色448×448像素，尚无人工真值',font=font(21),fill='#52626d')
        for k,w in enumerate(windows):
            ox=w['column']*250;oy=100+(w['row'] if w['band']=='A' else 2)*290
            im=Image.open(images/(w['window_id']+f'_T{t}.png'));canvas.paste(im.resize((240,240),Image.Resampling.LANCZOS),(ox,oy+40))
            draw.text((ox+3,oy+5),w['window_id'],font=font(20),fill='#24343c')
        canvas.save(views/f'T{t}_annotation_contact_sheet.png',dpi=(600,600))
    pw=1000;head=110;gap=40;heights=[round(arrays[b,1].shape[0]*pw/arrays[b,1].shape[1]) for b in ['A','B']]
    canvas=Image.new('RGB',(4*pw,head+sum(heights)+gap+75),'white');draw=ImageDraw.Draw(canvas)
    colors={'train':'#3e7866','val':'#b08336','test':'#6f69aa'}
    for t in range(1,5):
        ox=(t-1)*pw;draw.text((ox+12,12),f'T{t} {PHASES[t]} | 标注窗位置',font=font(30),fill='#24343c');draw.text((ox+12,60),'448像素窗；跨组保留64像素缓冲',font=font(23),fill='#52626d');oy=head
        for b,hh in zip(['A','B'],heights):
            a=arrays[b,t];canvas.paste(Image.fromarray(a).resize((pw,hh),Image.Resampling.LANCZOS),(ox,oy));bx,by,_,_=audit['regions_T1_xyxy'][b]
            for w in windows:
                if w['band']!=b:continue
                x,y,xx,yy=w['T1_xyxy'];scale=pw/a.shape[1];box=(ox+(x-bx)*scale,oy+(y-by)*scale,ox+(xx-bx)*scale,oy+(yy-by)*scale);draw.rectangle(box,outline=colors[w['split']],width=2)
            oy+=hh+gap
    draw.text((15,canvas.height-48),'绿：训练；赭：验证；紫：留出测试。白带为非数据分隔。所有时期同一观察窗同组。',font=font(24),fill='#52626d')
    canvas.save(views/'0353-2-B3_annotation_windows.png',dpi=(600,600))
    save_json(kit/'dataset_summary.json',dict(position='0353-2-B3',status='awaiting_human_dense_annotations',original_acquisitions=4,source_region_crops=8,spatial_windows=15,spatial_column_groups=5,patch_size_wh=[448,448],guard_band_T1_px=64,seed=20261014,counts={s:dict(image_patches=sum(x['split']==s for x in rows),spatial_windows=sum(x['split']==s for x in windows),column_groups=sum(v==s for v in assignment.values()),human_reviewed_images=0) for s in ['train','val','test']},column_assignment=assignment,source_crop_sha256={f'T{t}_region_{b}_RGB.png':sha[f'T{t}_region_{b}_RGB.png'] for b in ['A','B'] for t in range(1,5)},windows=windows,limitations='Single microtube position, spatial holdout only. Temporal observations are correlated and roots may cross ROI boundaries; this is not biological or external-location independence.64px guards reduce but cannot prove absence of root-level dependence. No rescaling, colour enhancement, blank ground-truth masks or claimed model accuracy. No physical scale supplied; lengths initially in T1 pixels.'))
    write_csv(kit/'model_comparison_pending.csv',[dict(model=m,status='not_trained_no_human_ground_truth',validation_precision='',validation_recall='',validation_dice='',validation_iou='',holdout_precision='',holdout_recall='',holdout_dice='',holdout_iou='',root_length_R2_identity='',root_length_RMSE_px='',root_length_CCC='') for m in ['RootPainter','U-Net','DeepLabV3+']])
    print('PREPARED_60_UNLABELLED_PATCHES',kit)

def grey(path):
    a=np.array(Image.open(path))
    if a.ndim==3:
        assert a.shape[2] in [3,4] and np.array_equal(a[...,0],a[...,1]) and np.array_equal(a[...,0],a[...,2]),f'Not a class-code grayscale mask: {path}'
        if a.shape[2]==4:assert (a[...,3]==255).all(),'Transparent pixels need explicit ignore128'
        a=a[...,0]
    return a

def checked_label(kit,row):
    p=kit/'labels'/(row['image_id']+'.png');q=p.with_suffix('.json')
    if not p.exists() or not q.exists():raise ValueError(f'Missing human mask/review record: {row["image_id"]}')
    review=json.loads(q.read_text());a=grey(p)
    assert a.shape==(int(row['height']),int(row['width']))
    assert set(np.unique(a)).issubset({0,128,255}),'Unknown64/unapproved label values are forbidden'
    assert review.get('reviewed_entire_image') is True and str(review.get('annotator','')).strip()
    assert review['image_sha256']==row['image_sha256'] and digest(kit/row['image_path'])==row['image_sha256']
    assert review['mask_sha256']==digest(p),'Mask changed after human review'
    assert (a!=128).any(),'Entirely ignored image cannot be evaluated'
    return a,review

def check_labels(args):
    rows=read_csv(args.kit/'split_manifest.csv');counts={s:0 for s in ['train','val','test']};missing=[]
    for row in rows:
        if args.split!='all' and row['split']!=args.split:continue
        try:checked_label(args.kit,row);counts[row['split']]+=1
        except (ValueError,AssertionError,KeyError) as e:missing.append(dict(image_id=row['image_id'],reason=str(e)))
    save_json(args.kit/'annotation_status.json',dict(reviewed=counts,unready=missing))
    print(json.dumps(dict(reviewed=counts,unready=len(missing)),ensure_ascii=False))
    if missing:raise SystemExit(2)

def metrics_from_counts(tp,fp,fn,tn):
    def ratio(a,b):return float(a/b) if b else None
    return dict(TP=int(tp),FP=int(fp),FN=int(fn),TN=int(tn),Precision=ratio(tp,tp+fp),Recall=ratio(tp,tp+fn),Dice=ratio(2*tp,2*tp+fp+fn) if 2*tp+fp+fn else 1.,IoU=ratio(tp,tp+fp+fn) if tp+fp+fn else 1.)

def counts(gt,pred):
    valid=gt!=128;g=(gt==255)&valid;p=pred.astype(bool)&valid
    return (int((g&p).sum()),int((~g&p&valid).sum()),int((g&~p&valid).sum()),int((~g&~p&valid).sum()))

def skeleton_length(binary):
    from skimage.morphology import skeletonize
    s=skeletonize(binary.astype(bool));length=float((s[:,:-1]&s[:,1:]).sum()+(s[:-1,:]&s[1:,:]).sum())
    dr=s[:-1,:-1]&s[1:,1:]&~s[:-1,1:]&~s[1:,:-1]
    dl=s[:-1,1:]&s[1:,:-1]&~s[:-1,:-1]&~s[1:,1:]
    return length+math.sqrt(2)*float(dr.sum()+dl.sum())

def agreement(manual,predicted):
    x=np.asarray(manual,float);y=np.asarray(predicted,float);assert len(x)==len(y) and len(x)>=2
    residual=y-x;var=float(np.sum((x-x.mean())**2));den=float(np.var(x)+np.var(y)+(x.mean()-y.mean())**2)
    out=dict(n=len(x),R2_identity=float(1-np.sum(residual**2)/var) if var>0 else None,RMSE=float(np.sqrt(np.mean(residual**2))),MAE=float(np.mean(abs(residual))),bias=float(residual.mean()),CCC=float(2*np.mean((x-x.mean())*(y-y.mean()))/den) if den>0 else None,unit='T1_px',interpretation='R2 against identity, not squared Pearson correlation. No biological-independence or significance claim.')
    if var>0:
        slope,intercept=np.polyfit(x,y,1);out.update(regression_slope=float(slope),regression_intercept=float(intercept),Pearson_r2=float(np.corrcoef(x,y)[0,1]**2) if np.var(y)>0 else None)
    sd=float(np.std(residual,ddof=1));out.update(Bland_Altman_lower=float(residual.mean()-1.96*sd),Bland_Altman_upper=float(residual.mean()+1.96*sd));return out

def evaluate(args):
    assert 0<=args.threshold<=1,'Probability threshold must be in [0,1]'
    rows=[r for r in read_csv(args.kit/'split_manifest.csv') if r['split']==args.split];report=[];total=np.zeros(4,dtype=np.int64);manuals={r['image_id']:r for r in read_csv(args.kit/'manual_lengths.csv')};pairs=[]
    for row in rows:
        gt,_=checked_label(args.kit,row);key=row['image_id'];p=args.predictions/(key+'.npy');q=p.with_suffix('.png')
        if p.exists():prob=np.load(p);assert np.isfinite(prob).all() and prob.min()>=0 and prob.max()<=1;pred=prob>=args.threshold
        elif q.exists():a=grey(q);assert set(np.unique(a)).issubset({0,255});pred=a==255
        else:raise ValueError('Missing prediction '+key)
        assert pred.shape==gt.shape;cc=counts(gt,pred);total+=cc;length=skeleton_length(pred&(gt!=128));m=manuals[key]
        item=dict(image_id=key,window_id=row['window_id'],spatial_group=row['spatial_group'],period=row['period'],**metrics_from_counts(*cc),valid_pixels=int((gt!=128).sum()),ignored_pixels=int((gt==128).sum()),prediction_skeleton_length_px=length,human_mask_derived_skeleton_length_px=skeleton_length((gt==255)),manual_independent_length_px=None)
        if m['manual_length_px']!='':
            assert str(m['fully_traced']).lower() in ['1','true','yes'] and m['annotator'].strip() and m['method'].strip()
            value=float(m['manual_length_px']);assert value>=0 and math.isfinite(value);item['manual_independent_length_px']=value;pairs.append((value,length,row['window_id']))
        report.append(item)
    out=args.output;out.mkdir(parents=True,exist_ok=True);write_csv(out/'per_image_metrics.csv',report)
    result=dict(model=args.model,split=args.split,threshold=args.threshold,n_images=len(rows),n_physical_windows=len({r['window_id'] for r in rows}),n_spatial_column_groups=len({r['spatial_group'] for r in rows}),n_microtube_positions=1,micro_metrics=metrics_from_counts(*total),macro_Dice=float(np.mean([x['Dice'] for x in report])),macro_IoU=float(np.mean([x['IoU'] for x in report])),root_length_agreement=agreement([x[0] for x in pairs],[x[1] for x in pairs]) if len(pairs)>=2 else None,manual_length_pairs=len(pairs),uncertainty='Metrics conditional on dense reviewed masks; human ignore128 excluded.12 held-out period images represent3 physical windows in1 column group and1 position. No independent-site generalization, nominal pixel p-values or preset Dice. Mask-derived skeleton length is not independent manual measurement.',label_sha256={r['image_id']:digest(args.kit/'labels'/(r['image_id']+'.png')) for r in rows})
    result['ignored_area_fraction_total']=sum(r['ignored_pixels'] for r in report)/sum(r['valid_pixels']+r['ignored_pixels'] for r in report)
    result['foreground_fraction_valid']=float((total[0]+total[2])/max(1,total.sum()))
    save_json(out/'metrics.json',result)
    if len(pairs)>=2:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        x=np.array([p[0] for p in pairs]);y=np.array([p[1] for p in pairs]);fig,ax=plt.subplots(figsize=(5,5));ax.scatter(x,y,s=22,c='#456b80');lo=min(x.min(),y.min());hi=max(x.max(),y.max());ax.plot([lo,hi],[lo,hi],color='#444',lw=1);ax.set(xlabel='Independent manual root length (T1 px)',ylabel='Predicted skeleton length (T1 px)');fig.tight_layout();fig.savefig(out/'root_length_identity.png',dpi=600);fig.savefig(out/'root_length_identity.pdf');plt.close(fig)
    print(json.dumps(result['micro_metrics']))

def build_model(name,weights=None):
    import segmentation_models_pytorch as smp
    cls=smp.Unet if name=='unet' else smp.DeepLabV3Plus
    return cls(encoder_name='resnet18',encoder_weights=weights,in_channels=3,classes=1)

def train(args):
    import torch
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed);torch.set_num_threads(args.threads)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');rows=read_csv(args.kit/'split_manifest.csv');selected={s:[r for r in rows if r['split']==s] for s in ['train','val']};cache={}
    # This must finish before model construction; missing human truth never triggers synthetic training.
    for s in selected:
        for row in selected[s]:gt,_=checked_label(args.kit,row);cache[row['image_id']]=(np.array(Image.open(args.kit/row['image_path']).convert('RGB')),gt)
    if not selected['train'] or not selected['val']:raise ValueError('Empty training/validation split')
    model=build_model(args.model,None if args.encoder_weights=='none' else args.encoder_weights).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4);out=args.output;out.mkdir(parents=True,exist_ok=True);history=[];best=-1.;bad=0
    mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    def batch(batch_rows,augment):
        images=[];targets=[];valids=[]
        for row in batch_rows:
            im,gt=cache[row['image_id']];im=im.copy();gt=gt.copy()
            if augment:
                k=random.randrange(4);im=np.rot90(im,k);gt=np.rot90(gt,k)
                if random.random()<.5:im=np.fliplr(im);gt=np.fliplr(gt)
            images.append(im.transpose(2,0,1).copy());targets.append((gt==255).copy());valids.append((gt!=128).copy())
        x=torch.from_numpy(np.stack(images)).float().to(device)/255.;y=torch.from_numpy(np.stack(targets)).float().to(device)[:,None];v=torch.from_numpy(np.stack(valids)).float().to(device)[:,None];return (x-mean)/std,y,v
    for epoch in range(args.epochs):
        model.train();order=selected['train'].copy();random.shuffle(order);loss_sum=0.
        for i in range(0,len(order),args.batch_size):
            x,y,v=batch(order[i:i+args.batch_size],True);optimizer.zero_grad();logit=model(x);p=torch.sigmoid(logit);bce=(torch.nn.functional.binary_cross_entropy_with_logits(logit,y,reduction='none')*v).sum()/v.sum().clamp_min(1);dice=1-(2*(p*y*v).sum()+1)/((p*v).sum()+(y*v).sum()+1);loss=bce+dice;loss.backward();optimizer.step();loss_sum+=loss.item()*len(x)
        model.eval();tot=np.zeros(4,np.int64)
        with torch.inference_mode():
            for row in selected['val']:
                x,_,_=batch([row],False);prob=torch.sigmoid(model(x))[0,0].cpu().numpy();tot+=counts(cache[row['image_id']][1],prob>=.5)
        score=metrics_from_counts(*tot)['Dice'];history.append(dict(epoch=epoch+1,training_loss=loss_sum/len(order),validation_Dice=score));write_csv(out/'history.csv',history)
        if score>best:
            best=score;bad=0;torch.save(dict(state_dict=model.state_dict(),model=args.model,encoder='resnet18',encoder_weights=args.encoder_weights,seed=args.seed,threshold=.5,epoch=epoch+1,normalization=dict(mean=[.485,.456,.406],std=[.229,.224,.225]),source_manifest_sha256=digest(args.kit/'split_manifest.csv'),training_label_sha256={r['image_id']:digest(args.kit/'labels'/(r['image_id']+'.png')) for r in selected['train']},validation_label_sha256={r['image_id']:digest(args.kit/'labels'/(r['image_id']+'.png')) for r in selected['val']}),out/'best.pt')
        else:bad+=1
        print('EPOCH',epoch+1,'VALIDATION_DICE',score,flush=True)
        if bad>=args.patience:break
    save_json(out/'training_run.json',dict(model=args.model,device=str(device),encoder='resnet18',encoder_weights=args.encoder_weights,seed=args.seed,epochs_completed=len(history),n_train=len(selected['train']),n_val=len(selected['val']),test_images_used_for_training=0,best_validation_Dice=best,parameters=sum(p.numel() for p in model.parameters()),checkpoint_sha256=digest(out/'best.pt')))

def predict(args):
    import torch
    torch.set_num_threads(args.threads);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');d=torch.load(args.checkpoint,map_location='cpu',weights_only=True);assert d['source_manifest_sha256']==digest(args.kit/'split_manifest.csv')
    model=build_model(d['model'],None);model.load_state_dict(d['state_dict']);model.to(device).eval();args.output.mkdir(parents=True,exist_ok=True);rows=[r for r in read_csv(args.kit/'split_manifest.csv') if r['split']==args.split]
    for row in rows:
        assert digest(args.kit/row['image_path'])==row['image_sha256'];im=np.array(Image.open(args.kit/row['image_path']).convert('RGB'));x=torch.from_numpy(im.transpose(2,0,1).copy()).float().to(device)[None]/255.;mean=torch.tensor(d['normalization']['mean'],device=device)[None,:,None,None];std=torch.tensor(d['normalization']['std'],device=device)[None,:,None,None]
        with torch.inference_mode():prob=torch.sigmoid(model((x-mean)/std))[0,0].cpu().numpy()
        np.save(args.output/(row['image_id']+'.npy'),prob);Image.fromarray((prob>=d['threshold']).astype('uint8')*255).save(args.output/(row['image_id']+'.png'))
    save_json(args.output/'prediction_provenance.json',dict(checkpoint_sha256=digest(args.checkpoint),split=args.split,threshold=d['threshold'],n_images=len(rows),model=d['model'],device=str(device)))

def export_rootpainter(args):
    # Official RootPainter trainer reads foreground from red and background from green.
    rows=[r for r in read_csv(args.kit/'split_manifest.csv') if r['split'] in ['train','val']]
    for row in rows:
        gt,_=checked_label(args.kit,row);a=np.zeros((*gt.shape,4),np.uint8);a[...,0]=(gt==255)*255;a[...,1]=(gt==0)*255;a[...,3]=(gt!=128)*255;p=args.output/'annotations'/row['split']/(row['image_id']+'.png');p.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(a).save(p)
        import shutil
        image_dir=args.output/'dataset';image_dir.mkdir(exist_ok=True);shutil.copy2(args.kit/row['image_path'],image_dir/(row['image_id']+'.png'))
    save_json(args.output/'export_provenance.json',dict(format='RootPainter RGBA red=foreground,green=background,undefined=zero alpha',source='https://github.com/Abe404/root_painter/blob/354fb93a9c14bfdba05ba00129d8e9b9547415cd/trainer/src/model_utils.py',n_train=sum(r['split']=='train' for r in rows),n_val=sum(r['split']=='val' for r in rows),test_annotations_exported=0,test_images_exported=0,rootpainter_training_performed=False))

def compare(args):
    rows=[]
    for p in args.reports:
        d=json.loads(p.read_text());m=d['micro_metrics'];a=d['root_length_agreement'] or {};rows.append(dict(model=d['model'],split=d['split'],n_images=d['n_images'],Precision=m['Precision'],Recall=m['Recall'],Dice=m['Dice'],IoU=m['IoU'],root_length_R2_identity=a.get('R2_identity'),root_length_RMSE_px=a.get('RMSE'),root_length_CCC=a.get('CCC'),manual_length_pairs=d['manual_length_pairs']))
    assert len({(x['model'],x['split']) for x in rows})==len(rows)
    # Same reviewed labels and same split required across methods.
    for split in {x['split'] for x in rows}:
        ds=[json.loads(p.read_text()) for p in args.reports if json.loads(p.read_text())['split']==split];assert all(d['label_sha256']==ds[0]['label_sha256'] for d in ds)
    write_csv(args.output,rows)

def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--source',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.set_defaults(func=prepare)
    p=sub.add_parser('check-labels');p.add_argument('--kit',type=Path,required=True);p.add_argument('--split',choices=['all','train','val','test'],default='all');p.set_defaults(func=check_labels)
    p=sub.add_parser('train');p.add_argument('--kit',type=Path,required=True);p.add_argument('--model',choices=['unet','deeplabv3plus'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=50);p.add_argument('--patience',type=int,default=10);p.add_argument('--seed',type=int,default=20261014);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--threads',type=int,default=2);p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--encoder-weights',choices=['none','imagenet'],default='none');p.set_defaults(func=train)
    p=sub.add_parser('predict');p.add_argument('--kit',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--split',choices=['val','test'],required=True);p.add_argument('--threads',type=int,default=2);p.set_defaults(func=predict)
    p=sub.add_parser('evaluate');p.add_argument('--kit',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--model',required=True);p.add_argument('--split',choices=['val','test'],required=True);p.add_argument('--threshold',type=float,default=.5);p.add_argument('--output',type=Path,required=True);p.set_defaults(func=evaluate)
    p=sub.add_parser('export-rootpainter');p.add_argument('--kit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.set_defaults(func=export_rootpainter)
    p=sub.add_parser('compare');p.add_argument('--reports',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.set_defaults(func=compare)
    args=parser.parse_args();args.func(args)

if __name__=='__main__':main()

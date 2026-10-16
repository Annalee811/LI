#!/usr/bin/env python3
"""Small U-Net pilot using the user's real T1/T2 masks; native pixels, spatial holdout.
No new labels, mask dilation, pseudo labels, downsampling, or cross-period split.
"""
from pathlib import Path
import argparse, csv, hashlib, io, json, math, random, time, zipfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn
from torch.nn import functional as F

SEED = 20261016
SIDE = 256
UPPER = (0, 0, 2541, 1108)
COLS = [22 + 512 * i for i in range(5)]
YS = [0, 256, 512, 768, 852]
PHASES = ['T1', 'T2']
SPLITS = {0: 'train', 1: 'train', 2: 'train', 3: 'val', 4: 'test'}
LAYERS = {'mask': '', 'color': '_color', 'temporal': '_temporal', 'viability': '_viability'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def read_mask(data):
    a = np.asarray(Image.open(io.BytesIO(data)))
    if a.ndim == 3:
        assert np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 0], a[..., 2])
        if a.shape[-1] == 4:
            assert (a[..., 3] == 255).all()
        a = a[..., 0]
    return a.copy()


def prepare(args):
    out = args.output; data = out / 'dataset'; data.mkdir(parents=True, exist_ok=True)
    originals = data / 'upper'; originals.mkdir(exist_ok=True)
    rows = []; source_records = []; arrays = {}
    for t, zpath in zip(PHASES, [args.t1_zip, args.t2_zip]):
        stem = t + '_multiregion_RGB'; src = args.originals / (stem + '.png')
        with zipfile.ZipFile(zpath) as z:
            meta = json.loads(z.read(stem + '.json'))
            assert sha(src) == meta['image_sha256'], 'Original RGB differs from the labelled image'
            raw = z.read(stem + '.png')
            assert hashlib.sha256(raw).hexdigest() == meta['mask_sha256']
            rgb_full = np.array(Image.open(src).convert('RGB'))
            assert rgb_full.shape == (1867, 2541, 3)
            assert (rgb_full[1108:1132] == 255).all(), 'Expected separator location differs'
            rgb = rgb_full[:1108].copy()
            layers = {k: read_mask(z.read(stem + s + '.png'))[:1108].copy() for k, s in LAYERS.items()}
            assert all(v.shape == rgb.shape[:2] for v in layers.values())
            assert set(np.unique(layers['mask'])).issubset({0, 128, 255})
            for k, a in layers.items():
                if k != 'mask':
                    assert not ((a > 0) & (layers['mask'] != 255)).any()
            arrays[t] = (rgb, layers)
            Image.fromarray(rgb).save(originals / (t + '_RGB.png'))
            for k, a in layers.items():
                Image.fromarray(a).save(originals / (t + '_' + k + '.png'))
            source_records.append(dict(period=t, annotation_zip_sha256=sha(zpath), original_rgb_sha256=sha(src),
                                       full_mask_sha256=meta['mask_sha256'], source_editor=meta['editor'],
                                       annotator=meta['annotator'], source_created_at=meta['created_at'],
                                       crop_xyxy_exclusive=list(UPPER), root_pixels=int((layers['mask'] == 255).sum()),
                                       original_full_image_review_flag=meta['reviewed_entire_image'],
                                       scope='Only upper region accepted by user; lower region is excluded, not background'))
        for col, x0 in enumerate(COLS):
            split = SPLITS[col]
            for yi, y in enumerate(YS):
                for xi, dx in enumerate([0, 192]):
                    x = x0 + dx; window = f'C{col}_X{xi}_Y{yi}'
                    image_id = window + '_' + t
                    paths = {}
                    for k in ['images', *LAYERS]:
                        folder = data / 'patches' / split / k; folder.mkdir(parents=True, exist_ok=True)
                        p = folder / (image_id + '.png')
                        value = rgb if k == 'images' else layers[k]
                        patch = value[y:y + SIDE, x:x + SIDE]
                        assert patch.shape[:2] == (SIDE, SIDE)
                        Image.fromarray(patch).save(p)
                        assert np.array_equal(np.array(Image.open(p)), patch)
                        paths[k] = str(p.relative_to(out))
                    gt = layers['mask'][y:y + SIDE, x:x + SIDE]
                    rows.append(dict(image_id=image_id, window_id=window, period=t, spatial_group=f'C{col}',
                                     column=col, split=split, x=x, y=y, width=SIDE, height=SIDE,
                                     **{k + '_path': v for k, v in paths.items()},
                                     image_sha256=sha(out / paths['images']), mask_sha256=sha(out / paths['mask']),
                                     root_pixels=int((gt == 255).sum()), ignored_pixels=int((gt == 128).sum())))
    # Guard distances are checked from actual patch coordinates, across every period and split.
    for r in rows:
        assert {x['split'] for x in rows if x['window_id'] == r['window_id']} == {r['split']}
        assert len([x for x in rows if x['window_id'] == r['window_id']]) == 2
        for q in rows:
            if r['split'] == q['split']:
                continue
            dx = max(r['x'] - (q['x'] + SIDE), q['x'] - (r['x'] + SIDE), 0)
            dy = max(r['y'] - (q['y'] + SIDE), q['y'] - (r['y'] + SIDE), 0)
            assert max(dx, dy) >= 64
    write_csv(data / 'manifest.csv', rows)
    summary = dict(original_acquisitions=2, microtube_positions=1, upper_roi_xyxy=list(UPPER), patch_size=[SIDE, SIDE],
                   resize=False, color_enhancement=False, label_modification=False, seed=SEED,
                   image_patches={s: sum(r['split'] == s for r in rows) for s in ['train', 'val', 'test']},
                   physical_windows={s: len({r['window_id'] for r in rows if r['split'] == s}) for s in ['train', 'val', 'test']},
                   paired_periods_same_group=True, min_cross_split_guard_pixels=64, columns=COLS,
                   column_splits=SPLITS, source_records=source_records,
                   overlap='Tiles overlap only within a split. Evaluation fuses overlapping probabilities before counting each pixel once.',
                   limitations=['Only one microtube position and two periods; no independent-site or biological replicate test.',
                                'Human masks are used exactly as supplied. Unlabelled fine roots may be encoded as background.',
                                'Root-level independence cannot be guaranteed: a root may span different guarded columns.',
                                'T2 temporal attributes are preserved but this model learns root/background only.',
                                'No independently measured root lengths or physical pixel scale are available.'])
    save_json(data / 'dataset_summary.json', summary)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 32)
    im = Image.fromarray(arrays['T1'][0]).convert('RGB'); draw = ImageDraw.Draw(im)
    for col, x in enumerate(COLS):
        color = {'train': '#27b070', 'val': '#e6ad1b', 'test': '#9156dd'}[SPLITS[col]]
        draw.rectangle((x, 0, x + 447, 1107), outline=color, width=8)
        draw.rectangle((x, 0, x + 447, 65), fill='#ffffff')
        draw.text((x + 10, 8), f'C{col} {SPLITS[col]}', font=font, fill=color)
    im.save(out / 'spatial_split_map.png', dpi=(300, 300))
    print('DATASET_READY', json.dumps(summary['image_patches']), flush=True)
    return rows, arrays, summary


class DoubleConv(nn.Sequential):
    def __init__(self, a, b):
        super().__init__(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.GroupNorm(4, b), nn.ReLU(inplace=True),
                         nn.Conv2d(b, b, 3, padding=1, bias=False), nn.GroupNorm(4, b), nn.ReLU(inplace=True))


class SmallUNet(nn.Module):
    """Four-level encoder/decoder U-Net with skip connections; width=12, from scratch."""
    def __init__(self, width=12):
        super().__init__()
        self.enc = nn.ModuleList([DoubleConv(3, width), DoubleConv(width, width*2),
                                 DoubleConv(width*2, width*4), DoubleConv(width*4, width*8)])
        self.bridge = DoubleConv(width*8, width*16)
        self.up = nn.ModuleList([nn.ConvTranspose2d(width*16, width*8, 2, 2), nn.ConvTranspose2d(width*8, width*4, 2, 2),
                                 nn.ConvTranspose2d(width*4, width*2, 2, 2), nn.ConvTranspose2d(width*2, width, 2, 2)])
        self.dec = nn.ModuleList([DoubleConv(width*16, width*8), DoubleConv(width*8, width*4),
                                 DoubleConv(width*4, width*2), DoubleConv(width*2, width)])
        self.head = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        skips = []
        for conv in self.enc:
            x = conv(x); skips.append(x); x = F.max_pool2d(x, 2)
        x = self.bridge(x)
        for up, conv, skip in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.pad(x, (0, skip.shape[-1] - x.shape[-1], 0, skip.shape[-2] - x.shape[-2]))
            x = conv(torch.cat([skip, x], 1))
        return self.head(x)


def confusion(gt, pred):
    valid = gt != 128; g = gt == 255; p = pred.astype(bool)
    return np.array([(g & p & valid).sum(), (~g & p & valid).sum(), (g & ~p & valid).sum(), (~g & ~p & valid).sum()], dtype=np.int64)


def metrics(c):
    tp, fp, fn, tn = map(int, c)
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, Precision=tp/max(1, tp+fp), Recall=tp/max(1, tp+fn),
                Dice=2*tp/max(1, 2*tp+fp+fn), IoU=tp/max(1, tp+fp+fn))


def infer(model, images, device, batch=8):
    out = []
    model.eval()
    with torch.inference_mode():
        for j in range(0, len(images), batch):
            x = torch.from_numpy(np.stack(images[j:j+batch]).transpose(0, 3, 1, 2).copy()).to(device).float()/255
            out.extend(torch.sigmoid(model(x)).squeeze(1).cpu().numpy())
    return out


def fused_columns(model, rows, arrays, split, device):
    selected = [r for r in rows if r['split'] == split]
    images = [arrays[r['period']][0][r['y']:r['y']+SIDE, r['x']:r['x']+SIDE] for r in selected]
    predictions = infer(model, images, device)
    maps = {}
    for row, prob in zip(selected, predictions):
        key = (row['period'], row['column']); x0 = COLS[row['column']]
        if key not in maps:
            maps[key] = [np.zeros((1108, 448), np.float32), np.zeros((1108, 448), np.float32)]
        val, weight = maps[key]; x, y = row['x'] - x0, row['y']
        val[y:y+SIDE, x:x+SIDE] += prob; weight[y:y+SIDE, x:x+SIDE] += 1
    result = {}
    for (t, col), (a, n) in maps.items():
        assert (n > 0).all()
        result[t, col] = (a/n, arrays[t][1]['mask'][:, COLS[col]:COLS[col]+448])
    return result


def evaluate_maps(maps, threshold):
    total = np.zeros(4, np.int64); rows = []
    for (t, col), (prob, gt) in maps.items():
        c = confusion(gt, prob >= threshold); total += c
        rows.append(dict(period=t, column=col, threshold=threshold, valid_pixels=int((gt != 128).sum()), **metrics(c)))
    return metrics(total), rows


def train(args, rows, arrays, model_factory=None, model_name="Small U-Net") :
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = args.output/'training'; out.mkdir(exist_ok=True)
    tr = [r for r in rows if r['split'] == 'train']
    # Only training and validation rows enter fitting or model selection.
    cache = {}
    for r in rows:
        if r['split'] == 'test':
            continue
        rgb, layers = arrays[r['period']]; y, x = r['y'], r['x']
        cache[r['image_id']] = (rgb[y:y+SIDE, x:x+SIDE].copy(), layers['mask'][y:y+SIDE, x:x+SIDE].copy())
    roots = sum((cache[r['image_id']][1] == 255).sum() for r in tr)
    valid = sum((cache[r['image_id']][1] != 128).sum() for r in tr)
    pos_weight = min(6., math.sqrt((valid-roots)/max(roots, 1)))
    model = (model_factory or SmallUNet)(args.width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr*0.05)
    history = []; best = -1; bad = 0; start = time.monotonic()
    config = dict(seed=SEED, device=str(device), cuda_available=torch.cuda.is_available(), model=model_name, width=args.width,
                  architecture=getattr(model,'architecture_description','4 encoder levels, bridge, 4 transposed-convolution upsamplers and skip connections; GroupNorm'),
                  parameters=sum(p.numel() for p in model.parameters()), from_scratch=True, pretrained_weights=None,
                  patch_size=SIDE, epochs_max=args.epochs, batch_size=args.batch, learning_rate=args.lr, threads=args.threads,
                  positive_bce_weight=float(pos_weight), loss='weighted binary cross entropy + soft Dice loss',
                  augmentation='train only: 90 degree rotations, reflections, brightness and gamma variation',
                  normalization='RGB / 255', early_stopping_patience=args.patience,
                  checkpoint_selection='Maximum validation Dice at a pre-fixed probability threshold 0.5',
                  threshold_selection='After checkpoint selection, maximize Dice on validation only over 0.10..0.90 in 0.05 steps',
                  test_access_for_model_selection=False, labels_auto_modified=False,
                  manifest_sha256=sha(args.output/'dataset/manifest.csv'), torch_version=str(torch.__version__))
    save_json(out/'config.json', config)
    for epoch in range(1, args.epochs+1):
        tic = time.monotonic(); model.train(); order = tr.copy(); random.shuffle(order); losses = []
        for i in range(0, len(order), args.batch):
            ims=[]; targets=[]; valids=[]
            for r in order[i:i+args.batch]:
                im, gt = cache[r['image_id']]; im = im.copy(); gt = gt.copy(); k = random.randrange(4)
                im=np.rot90(im,k);gt=np.rot90(gt,k)
                if random.random()<.5:
                    im=np.fliplr(im);gt=np.fliplr(gt)
                if random.random()<.5:
                    im=np.flipud(im);gt=np.flipud(gt)
                value=np.clip(im.astype(np.float32)/255*random.uniform(.9,1.1),0,1)**random.uniform(.9,1.1)
                ims.append(value.transpose(2,0,1).copy());targets.append((gt==255).copy());valids.append((gt!=128).copy())
            x=torch.from_numpy(np.stack(ims)).to(device);y=torch.from_numpy(np.stack(targets)).to(device).float()[:,None];v=torch.from_numpy(np.stack(valids)).to(device).float()[:,None]
            opt.zero_grad(set_to_none=True); logits=model(x);p=torch.sigmoid(logits)
            bce=(F.binary_cross_entropy_with_logits(logits,y,reduction='none',pos_weight=torch.tensor(pos_weight,device=device))*v).sum()/v.sum().clamp_min(1)
            dice=1-(2*(p*y*v).sum()+1)/((p*v).sum()+(y*v).sum()+1)
            loss=bce+dice;loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()))
        val_maps=fused_columns(model,rows,arrays,'val',device);m,_=evaluate_maps(val_maps,.5)
        record=dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_Dice=m['Dice'],validation_IoU=m['IoU'],
                    validation_Precision=m['Precision'],validation_Recall=m['Recall'],learning_rate=opt.param_groups[0]['lr'],elapsed_seconds=time.monotonic()-tic)
        history.append(record);write_csv(out/'history.csv',history)
        if m['Dice']>best+1e-5:
            best=m['Dice'];bad=0
            torch.save(dict(state_dict=model.state_dict(),width=args.width,epoch=epoch,validation_Dice_at_05=best,
                            manifest_sha256=config['manifest_sha256'],seed=SEED,normalization='RGB/255',configuration=config),out/'best.pt')
        else:
            bad+=1
        scheduler.step()
        print('EPOCH',epoch,'loss',round(record['train_loss'],4),'VAL_DICE',round(m['Dice'],4),'sec',round(record['elapsed_seconds'],1),flush=True)
        if bad>=args.patience:
            break
    checkpoint=torch.load(out/'best.pt',map_location=device,weights_only=True);model.load_state_dict(checkpoint['state_dict']);model.eval()
    vm=fused_columns(model,rows,arrays,'val',device)
    threshold_rows=[]
    for thr in np.arange(.10,.901,.05):
        m,_=evaluate_maps(vm,float(round(thr,2)));threshold_rows.append(dict(threshold=float(round(thr,2)),**m))
    # Exact tie breaks towards 0.5; threshold is frozen before any test predictions.
    chosen=max(threshold_rows,key=lambda r:(r['Dice'],-abs(r['threshold']-.5)))
    write_csv(out/'validation_thresholds.csv',threshold_rows);threshold=chosen['threshold']
    result=dict(**config,epochs_completed=len(history),best_epoch=checkpoint['epoch'],best_validation_Dice_at_05=best,
                frozen_threshold=threshold,total_seconds=time.monotonic()-start,checkpoint_sha256=sha(out/'best.pt'))
    save_json(out/'training_run.json',result)
    return model,device,threshold,result


def whole_image(model,rgb,device):
    h,w=rgb.shape[:2];xs=list(range(0,w-SIDE+1,192));ys=list(range(0,h-SIDE+1,192))
    if xs[-1]!=w-SIDE:xs.append(w-SIDE)
    if ys[-1]!=h-SIDE:ys.append(h-SIDE)
    coords=[(x,y) for y in ys for x in xs]
    value=np.zeros((h,w),np.float32);weight=np.zeros((h,w),np.float32)
    # Hann weighting reduces seams; overlap never changes label data or held-out evaluation support.
    v=np.hanning(SIDE).astype(np.float32);blend=np.maximum(v[:,None]*v[None,:],.02)
    for j in range(0,len(coords),8):
        local=coords[j:j+8];patches=[rgb[y:y+SIDE,x:x+SIDE] for x,y in local]
        for (x,y),p in zip(local,infer(model,patches,device)):
            value[y:y+SIDE,x:x+SIDE]+=p*blend;weight[y:y+SIDE,x:x+SIDE]+=blend
    assert (weight>0).all()
    return value/weight


def visualize_and_evaluate(args,model,device,threshold,rows,arrays,summary,run):
    out=args.output;ev=out/'evaluation';ev.mkdir(exist_ok=True);pred_dir=out/'predictions';pred_dir.mkdir(exist_ok=True)
    reports={};per_col=[]
    # Held-out metrics only use guarded validation/test column predictions, not whole-image context.
    for split in ['val','test']:
        maps=fused_columns(model,rows,arrays,split,device);m,rr=evaluate_maps(maps,threshold)
        for r in rr:r['split']=split
        per_col.extend(rr)
        by_period={}
        for t in PHASES:
            by_period[t]=metrics(sum((confusion(g,p>=threshold) for (tt,c),(p,g) in maps.items() if tt==t),np.zeros(4,np.int64)))
        reports[split]=dict(micro_metrics=m,per_period=by_period,n_patch_images=summary['image_patches'][split],
                            n_physical_windows=summary['physical_windows'][split],n_spatial_column_groups=1,
                            n_unique_evaluated_pixels=sum(int((g!=128).sum()) for p,g in maps.values()),threshold=threshold)
        folder=pred_dir/split;folder.mkdir(exist_ok=True)
        for (t,col),(prob,gt) in maps.items():
            Image.fromarray((prob>=threshold).astype('uint8')*255).save(folder/f'{t}_C{col}_prediction.png')
            Image.fromarray(gt).save(folder/f'{t}_C{col}_human_mask.png')
            np.save(folder/f'{t}_C{col}_probability.npy',prob.astype(np.float16))
    write_csv(ev/'per_column_metrics.csv',per_col)
    save_json(ev/'metrics.json',dict(model='Small U-Net',task='binary root segmentation',validation=reports['val'],spatial_holdout_test=reports['test'],
              comparison_target='User-supplied mask, not independently adjudicated anatomical ground truth',
              test_interpretation='Spatial holdout within one microtube position, not an external or biological-independent test',
              manual_root_length_agreement=None,manual_root_length_reason='No independently measured lengths or physical scale supplied',
              model_comparison_status='One U-Net pilot trained; RootPainter and DeepLabV3+ have not been trained or compared',
              limitations=summary['limitations']))
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',38)
    for t,(rgb,layers) in arrays.items():
        p=whole_image(model,rgb,device);binary=p>=threshold
        Image.fromarray(binary.astype('uint8')*255).save(pred_dir/f'{t}_upper_prediction_mask.png')
        np.save(pred_dir/f'{t}_upper_probability.npy',p.astype(np.float16))
        human=rgb.copy();sel=layers['mask']==255;human[sel]=(human[sel]*.35+np.array([0,170,230])*.65).astype('uint8')
        pred=rgb.copy();pred[binary]=(pred[binary]*.35+np.array([30,185,95])*.65).astype('uint8')
        panel=Image.new('RGB',(rgb.shape[1]*3,1168),'white');d=ImageDraw.Draw(panel)
        for j,(im,label) in enumerate([(rgb,f'{t} original RGB'),(human,'Supplied human labels'),(pred,'U-Net prediction (all splits)')]):
            panel.paste(Image.fromarray(im),(j*rgb.shape[1],60));d.text((j*rgb.shape[1]+12,7),label,font=font,fill='#20352d')
        panel.save(out/f'{t}_comparison.png',dpi=(300,300))
        Image.fromarray(pred).save(pred_dir/f'{t}_upper_prediction_overlay.png',dpi=(300,300))
    # Test-only examples are displayed after all model and threshold choices have been frozen.
    test_rows=[r for r in rows if r['split']=='test']
    selected=[next(r for r in test_rows if r['period']==t and r['y']==y and r['x']==COLS[4]) for t in PHASES for y in [0,512]]
    canvas=Image.new('RGB',(4*SIDE,4*(SIDE+38)),'white');d=ImageDraw.Draw(canvas);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    for i,r in enumerate(selected):
        x,y=r['x'],r['y'];rgb,layers=arrays[r['period']];image=rgb[y:y+SIDE,x:x+SIDE];gt=layers['mask'][y:y+SIDE,x:x+SIDE]
        # Crop the same fused-column probabilities used for the unique-pixel test score.
        p=np.load(pred_dir/'test'/f'{r["period"]}_C4_probability.npy').astype(np.float32)[y:y+SIDE,:SIDE]>=threshold
        error=np.zeros((SIDE,SIDE,3),np.uint8);g=gt==255;error[g&p]=[235,235,235];error[~g&p]=[230,80,60];error[g&~p]=[30,150,230]
        panels=[image,np.repeat((g*255).astype('uint8')[...,None],3,2),np.repeat((p*255).astype('uint8')[...,None],3,2),error]
        for j,(im,label) in enumerate(zip(panels,['RGB','Human','Prediction','TP white / FP red / FN blue'])):
            yy=i*(SIDE+38);canvas.paste(Image.fromarray(im),(j*SIDE,yy+38));d.text((j*SIDE+3,yy+4),r['period']+' '+label,font=small,fill='black')
    canvas.save(out/'heldout_test_examples.png',dpi=(300,300))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history=list(csv.DictReader((out/'training/history.csv').open(encoding='utf-8-sig')))
    fig,ax=plt.subplots(1,2,figsize=(9,3.5));epochs=[int(r['epoch']) for r in history]
    ax[0].plot(epochs,[float(r['train_loss']) for r in history],c='#38624d');ax[0].set(xlabel='Epoch',ylabel='Training loss')
    ax[1].plot(epochs,[float(r['validation_Dice']) for r in history],c='#655291');ax[1].axvline(run['best_epoch'],ls='--',c='#777',lw=1);ax[1].set(xlabel='Epoch',ylabel='Validation Dice (threshold 0.5)',ylim=(0,1))
    fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=300);fig.savefig(out/'training_curves.pdf');plt.close(fig)
    v=reports['val']['micro_metrics'];te=reports['test']['micro_metrics']
    text=f'''0353-2-B3 上部 T1/T2 根系分割预实验

已使用用户提供的人工标注；不再要求追加标注。下方区域及白色分隔带全部排除。
上部原始 RGB：2张，各2541×1108像素；切块：256×256像素，不缩放、不修改标注。
训练60张／验证20张／空间留出测试20张，共100张；对应50个物理观察窗，每窗包含T1/T2。
同一区域T1/T2始终同组。跨组至少64像素隔离；重叠切块只发生在组内。
评估先融合组内重叠预测，每个有效像素只计数一次。

模型：小型 U-Net，随机初始化，设备{run['device']}，完成{run['epochs_completed']}轮，采用第{run['best_epoch']}轮模型。
阈值仅用验证集选取：{threshold:.2f}。
验证集：Precision={v['Precision']:.4f}，Recall={v['Recall']:.4f}，Dice={v['Dice']:.4f}，IoU={v['IoU']:.4f}。
空间留出测试：Precision={te['Precision']:.4f}，Recall={te['Recall']:.4f}，Dice={te['Dice']:.4f}，IoU={te['IoU']:.4f}。

上述结果仅表示模型与现有人工掩膜的一致程度。未标注细根与部分线状标注会影响训练及指标，不能据此判断所有根的真实识别率。
空间留出来自同一位置，不能作为跨位置泛化证明；64像素隔离不能保证根个体独立。
整幅预测图包含训练区，仅用于查看；报告的测试指标只来自未参与训练或调参的空间留出区。
模型只学习根／背景；T2新生、延伸、持续等属性原样保存，未训练时序分类。
未评估独立人工根长一致性，未训练RootPainter或DeepLabV3+，未生成虚构比较结果。

复现：详见source/train_uploaded_upper.py及training/config.json；best.pt包含模型参数。
'''
    (out/'结果说明.txt').write_text(text,encoding='utf-8-sig')
    print('FINAL_METRICS',json.dumps(reports),flush=True)
    return reports


def main():
    p=argparse.ArgumentParser();p.add_argument('--t1-zip',type=Path);p.add_argument('--t2-zip',type=Path)
    p.add_argument('--originals',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--existing-dataset',action='store_true',help='Use the exported dataset for portable retraining without source ZIPs')
    p.add_argument('--epochs',type=int,default=80);p.add_argument('--patience',type=int,default=18);p.add_argument('--batch',type=int,default=4)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--width',type=int,default=12);p.add_argument('--lr',type=float,default=.001)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.existing_dataset:
        with (args.output/'dataset/manifest.csv').open(encoding='utf-8-sig',newline='') as f:
            rows=list(csv.DictReader(f))
        for r in rows:
            for k in ['column','x','y','width','height','root_pixels','ignored_pixels']:
                r[k]=int(r[k])
            assert sha(args.output/r['images_path'])==r['image_sha256']
            assert sha(args.output/r['mask_path'])==r['mask_sha256']
        arrays={t:(np.array(Image.open(args.output/'dataset/upper'/f'{t}_RGB.png').convert('RGB')),
                   {k:np.array(Image.open(args.output/'dataset/upper'/f'{t}_{k}.png').convert('L')) for k in LAYERS}) for t in PHASES}
        summary=json.loads((args.output/'dataset/dataset_summary.json').read_text())
        for r in rows:
            rgb,layers=arrays[r['period']];x,y=r['x'],r['y']
            assert np.array_equal(rgb[y:y+SIDE,x:x+SIDE],np.array(Image.open(args.output/r['images_path'])))
            assert np.array_equal(layers['mask'][y:y+SIDE,x:x+SIDE],np.array(Image.open(args.output/r['mask_path'])))
    else:
        if not all([args.t1_zip,args.t2_zip,args.originals]):
            p.error('Supply both annotation ZIPs and original RGB folder, or --existing-dataset')
        rows,arrays,summary=prepare(args)
    model,device,threshold,run=train(args,rows,arrays)
    visualize_and_evaluate(args,model,device,threshold,rows,arrays,summary,run)

if __name__=='__main__':main()

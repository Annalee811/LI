#!/usr/bin/env python3
"""Training-only visibility screening and a matched-budget fine-tuning experiment."""
import argparse,csv,json,random,time,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy import ndimage as ndi
import torch
from torch import nn
import protocol as p
from train import RootUNet,objective,inference

def read(path):return json.loads(Path(path).read_text())
def csvout(path,rows):
 with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def visibility(rgb,mask):
 gray=np.mean(rgb.astype(np.float32)/255,axis=2)
 detail=ndi.gaussian_filter(gray,.65)-ndi.gaussian_filter(gray,3.)
 base=ndi.gaussian_filter(gray,5);std=np.sqrt(np.maximum(ndi.gaussian_filter(gray**2,5)-base**2,0))
 score=abs(detail)/(std+.025);root=mask==255
 return dict(root_pixels=int(root.sum()),root_visibility=float(np.median(score[root])) if root.any() else 0.,
  root_visible_fraction=float((score[root]>.3).mean()) if root.any() else 0.,
  root_clipped_fraction=float(((gray[root]<.025)|(gray[root]>.975)).mean()) if root.any() else 1.)

def prepare(data,run):
 run.mkdir(parents=True,exist_ok=True);rows=[];background=[]
 for t in p.PERIODS:
  for col,x0 in enumerate(p.COLS[:3]):
   local=[]
   for index,(x,y) in enumerate(p.XY):
    rgb=data.rgb[t][y:y+384,x0+x:x0+x+384];gt=data.gt[t][y:y+384,x0+x:x0+x+384]
    features=visibility(rgb,gt);quality=features['root_visibility']*(1-features['root_clipped_fraction'])
    row=dict(id=f'{t}_C{col}_W{index:02d}',period=t,column=col,x=x0+x,y=y,side=384,split='train',**features,quality=quality,selected=False)
    local.append(row)
   # Predeclared training-only rule; no model score, validation or C4 input.
   ranked=sorted(local,key=lambda r:(r['quality'],r['root_visible_fraction']),reverse=True)
   for row in ranked[:7]:row['selected']=True
   rows+=local
   # One native128 background example per period/column, if a root-free margin exists.
   gt=data.gt[t][:,x0:x0+448];distance=ndi.distance_transform_edt(gt==0);candidates=[]
   for y in range(0,981,64):
    for x in range(0,321,64):
     if distance[y:y+128,x:x+128].min()<6:continue
     rgb=data.rgb[t][y:y+128,x0+x:x0+x+128];gray=rgb.mean(axis=2)/255
     if gray.mean()<.06 or gray.std()<.015:continue
     candidates.append((float(gray.std()),x,y))
   if candidates:
    _,x,y=max(candidates);background.append(dict(id=f'{t}_C{col}_BG',period=t,column=col,x=x0+x,y=y,side=128,split='train',root_pixels=0))
 csvout(run/'training_visibility.csv',rows);csvout(run/'background_candidates.csv',background)
 review=run/'review';review.mkdir(exist_ok=True)
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
 for t in p.PERIODS:
  rr=[r for r in rows if r['period']==t];canvas=Image.new('RGB',(6*384,5*414),'white');draw=ImageDraw.Draw(canvas)
  for i,r in enumerate(rr):
   x,y=r['x'],r['y'];left=(i%6)*384;top=(i//6)*414
   canvas.paste(Image.fromarray(data.rgb[t][y:y+384,x:x+384]),(left,top+30))
   draw.text((left+3,top+5),r['id']+(' KEEP' if r['selected'] else ' OMIT')+f"  V={r['quality']:.2f}",font=font,fill='#006E65' if r['selected'] else '#A44820')
  canvas.save(review/(t+'_training_RGB_review.jpg'),quality=96)
 if background:
  canvas=Image.new('RGB',(6*256,((len(background)+5)//6)*286),'white');draw=ImageDraw.Draw(canvas)
  for i,r in enumerate(background):
   im=Image.fromarray(data.rgb[r['period']][r['y']:r['y']+128,r['x']:r['x']+128]).resize((256,256),Image.Resampling.NEAREST)
   left=i%6*256;top=i//6*286;canvas.paste(im,(left,top+30));draw.text((left+3,top+5),r['id'],font=font,fill='black')
  canvas.save(review/'background_RGB_review.png')
 p.js(run/'screening_protocol.json',dict(training_only=True,selected_root_patches=sum(r['selected'] for r in rows),omitted_root_patches=sum(not r['selected'] for r in rows),
  background_candidates=len(background),visibility_proxy='Median root-pixel local intensity deviation / local texture; clipping penalty. Not a manual certainty label.',
  rule='Keep7 of10 fixed native384 windows within every training period/column stratum, frozen before new validation evaluation.',
  original_ground_truth_edited=False,validation_or_C4_filtered=False,all_four_periods_spatial_groups_unchanged=True,
  original_progress_SHA256=p.sha(data.source/'original_human_progress.json'),source_provenance_SHA256=p.sha(data.source/'dataset/provenance.json')))
 print('TRAINING_SCREEN_PREPARED',sum(r['selected'] for r in rows),len(background),flush=True)

class ScreenedSampler:
 def __init__(self,data,run):
  self.data=data
  with (run/'training_visibility.csv').open(encoding='utf-8-sig') as f:self.pool=[r for r in csv.DictReader(f) if r['selected']=='True']
  with (run/'background_candidates.csv').open(encoding='utf-8-sig') as f:self.bg=list(csv.DictReader(f))
  # Explicit visual-review overrides, if provided; never evaluated on validation/test.
  review=read(run/'visual_review.json') if (run/'visual_review.json').exists() else {}
  self.bg=[r for r in self.bg if r['id'] not in review.get('exclude_background_ids',[])]
  self.pool=[r for r in self.pool if r['id'] not in review.get('exclude_root_patch_ids',[])]
 def sample(self,index):
  # Every tenth update can use a native128 background batch, with no image resizing.
  use_background=(index//2)%10==9 and bool(self.bg)
  row=random.choice(self.bg if use_background else self.pool);t=row['period'];x=int(row['x']);y=int(row['y']);side=int(row['side'])
  im=self.data.rgb[t][y:y+side,x:x+side].copy();gt=self.data.gt[t][y:y+side,x:x+side].copy()
  assert int(row['column'])<3 and x>=p.COLS[int(row['column'])] and x+side<=p.COLS[int(row['column'])]+448
  k=random.randrange(4);im=np.rot90(im,k);gt=np.rot90(gt,k)
  if random.random()<.5:im=np.fliplr(im);gt=np.fliplr(gt)
  if random.random()<.5:im=np.flipud(im);gt=np.flipud(gt)
  im=im.astype(np.float32)/255;im=np.clip(im*random.uniform(.85,1.15),0,1)**random.uniform(.85,1.15)
  return im.copy(),(gt==255).astype(np.float32),(gt!=128).astype(np.float32)

def train_arm(data,run,initial,arm,epochs,threads):
 folder=run/arm;folder.mkdir(exist_ok=True);torch.set_num_threads(threads);torch.manual_seed(p.SEED);random.seed(p.SEED);np.random.seed(p.SEED)
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=RootUNet().to(device)
 ck=torch.load(initial,map_location=device,weights_only=True);model.load_state_dict(ck['state_dict'])
 sampler=data if arm=='control' else ScreenedSampler(data,run)
 enc=list(model.encoder.parameters());ids={id(z) for z in enc};decoder=[z for z in model.parameters() if id(z) not in ids]
 opt=torch.optim.AdamW([{'params':enc,'lr':5e-6},{'params':decoder,'lr':5e-5}],weight_decay=1e-4)
 maps=data.column(inference(model,device),3);baseline=data.evaluate(maps,3,.5)['root'];best=baseline['Dice'];bad=0;hist=[]
 config=dict(arm=arm,initial_SHA256=p.sha(initial),seed=p.SEED,epochs_max=epochs,samples_per_epoch=32,batch_size=2,encoder_lr=5e-6,decoder_lr=5e-5,
  device=str(device),CUDA_available=torch.cuda.is_available(),threads=threads,only_training_sampler_differs=True,objective='unchanged v6 hard-negative BCE+Dice+0.15clDice',
  validation_region='unchanged C3, all original pixels',C4_used_for_selection=False,labels_edited=False,
  screening_protocol_SHA256=p.sha(run/'screening_protocol.json'),visual_review_SHA256=p.sha(run/'visual_review.json'))
 torch.save(dict(state_dict=model.state_dict(),epoch=0,config=config,validation_Dice=best),folder/'best.pt')
 p.js(folder/'initial_validation.json',baseline);start=time.monotonic()
 for epoch in range(1,epochs+1):
  model.train();losses=[];factor=.2+.8*.5*(1+np.cos(np.pi*(epoch-1)/epochs))
  for group,lr in zip(opt.param_groups,[5e-6,5e-5]):group['lr']=lr*factor
  for i in range(0,32,2):
   # Pair members use the same native size for background batches.
   index=i+32*(epoch-1);ss=[sampler.sample(index+j) for j in range(2)]
   x=torch.from_numpy(np.stack([s[0] for s in ss]).transpose(0,3,1,2).copy()).to(device)
   y=torch.from_numpy(np.stack([s[1] for s in ss])[:,None]).to(device);v=torch.from_numpy(np.stack([s[2] for s in ss])[:,None]).to(device)
   opt.zero_grad(set_to_none=True);loss=objective(model(x),y,v);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()))
  rec=dict(epoch=epoch,loss=float(np.mean(losses)),validation_Dice=None,validation_Precision=None,validation_Recall=None)
  if epoch%5==0 or epoch==epochs:
   maps=data.column(inference(model,device),3);m=data.evaluate(maps,3,.5)['root'];rec.update(validation_Dice=m['Dice'],validation_Precision=m['Precision'],validation_Recall=m['Recall'])
   if m['Dice']>best+1e-5:
    best=m['Dice'];bad=0;torch.save(dict(state_dict=model.state_dict(),epoch=epoch,config=config,validation_Dice=best),folder/'best.pt')
   else:bad+=1
   print('CLEAR_PATCH_EPOCH',arm,epoch,'VAL_DICE',round(m['Dice'],5),flush=True)
  hist.append(rec);csvout(folder/'history.csv',hist)
 ck=torch.load(folder/'best.pt',map_location=device,weights_only=True);model.load_state_dict(ck['state_dict']);choices=[]
 for tta in [False,True]:
  maps=data.column(inference(model,device,tta),3);chosen,grid=p.calibrate(data,maps);p.save_maps(folder/('val_tta' if tta else 'val_identity'),maps)
  csvout(folder/('thresholds_tta.csv' if tta else 'thresholds_identity.csv'),grid);choices.append(dict(tta=tta,threshold=chosen['threshold'],metrics={k:v for k,v in chosen.items() if k!='threshold'}))
 selected=max(choices,key=lambda r:r['metrics']['Dice']);p.js(folder/'validation_selection.json',dict(selected=selected,trials=choices,C4_used_for_selection=False))
 p.js(folder/'training_run.json',dict(**config,epochs_completed=len(hist),best_epoch=ck['epoch'],seconds=time.monotonic()-start,checkpoint_SHA256=p.sha(folder/'best.pt')))
 print('CLEAR_PATCH_TRAIN_DONE',arm,selected,flush=True)

def main():
 a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True);a.add_argument('--run',type=Path,required=True);a.add_argument('--prepare',action='store_true')
 a.add_argument('--initial',type=Path);a.add_argument('--arm',choices=['control','screened']);a.add_argument('--epochs',type=int,default=20);a.add_argument('--threads',type=int,default=3);args=a.parse_args()
 data=p.Data(args.data)
 if args.prepare:prepare(data,args.run)
 else:
  assert args.arm and args.initial and (args.run/'visual_review.json').exists()
  train_arm(data,args.run,args.initial,args.arm,args.epochs,args.threads)
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Export the exact native crops and augmentation recipe used by v9 training."""
import argparse,random,sys,shutil
from pathlib import Path
import numpy as np
from PIL import Image
import protocol as p
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'root_background_v5'))
import common as c

def replay(data,index):
 t=random.choice(c.PERIODS);col=random.randrange(3);x0=c.COLS[col]
 if index%2==0:dx,y=random.choice(c.COORDS)
 else:
  yy,xx=data.rootcoords[t,col][random.randrange(len(data.rootcoords[t,col]))];dx=int(np.clip(xx-random.randrange(256),0,192));y=int(np.clip(yy-random.randrange(256),0,852))
 im=data.rgb[t][y:y+256,x0+dx:x0+dx+256].copy();gt=data.gt[t][y:y+256,x0+dx:x0+dx+256].copy();k=random.randrange(4);im=np.rot90(im,k);gt=np.rot90(gt,k)
 flipx=random.random()<.5
 if flipx:im=np.fliplr(im);gt=np.fliplr(gt)
 flipy=random.random()<.5
 if flipy:im=np.flipud(im);gt=np.flipud(gt)
 brightness=random.uniform(.9,1.1);gamma=random.uniform(.9,1.1);im=np.clip(im.astype(np.float32)/255*brightness,0,1)**gamma
 return (im.copy(),(gt==255).astype(np.float32),(gt!=128).astype(np.float32)),dict(period=t,column=col,x=x0+dx,y=y,side=256,rotation_quarters=k,flipx=flipx,flipy=flipy,brightness=brightness,gamma=gamma)

def main():
 a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True);a.add_argument('--run',type=Path,required=True);args=a.parse_args();p.Data(args.data);data=c.Data(args.data);out=args.run/'data';out.mkdir(parents=True,exist_ok=True)
 shutil.copytree(args.data/'dataset/upper',out/'dataset/upper',dirs_exist_ok=True);shutil.copyfile(args.data/'original_human_progress.json',out/'original_human_progress.json')
 random.seed(p.SEED);records=[];checks=0
 for epoch in range(1,9):
  for i in range(32):
   state=random.getstate();expected=data.sample(i);random.setstate(state);actual,record=replay(data,i);assert all(np.array_equal(a,b) for a,b in zip(expected,actual));checks+=1
   records.append(dict(id='E%02d_S%02d'%(epoch,i),epoch=epoch,sample=i,partition='train',**record))
 # All periods always inherit the same spatial column; no split by date.
 for side,coords in [(256,c.COORDS),(384,p.XY)]:
  for split,col in [('val',3),('test',4)]:
   for t in p.PERIODS:
    for j,(dx,y) in enumerate(coords):records.append(dict(id='%s_C%d_S%d_W%02d'%(t,col,side,j),epoch=0,sample=j,partition=split,period=t,column=col,x=p.COLS[col]+dx,y=y,side=side,rotation_quarters=0,flipx=False,flipy=False,brightness=1.,gamma=1.))
 hashes={};manifest=[]
 for r in records:
  stem=r['id'];x,y,side=r['x'],r['y'],r['side'];t=r['period'];entry=dict(r,spatial_group='C'+str(r['column']))
  assert x>=p.COLS[r['column']] and x+side<=p.COLS[r['column']]+448
  for layer in ['RGB','mask','color','temporal','viability']:
   src=np.array(Image.open(args.data/'dataset/upper'/(t+'_'+layer+'.png')));native=src[y:y+side,x:x+side];f=out/'dataset/patches'/r['partition']/(stem+'_'+layer+'.png');f.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(native).save(f);assert np.array_equal(np.array(Image.open(f)),native);entry[layer]=str(f.relative_to(out));hashes[str(f.relative_to(out))]=p.sha(f)
  manifest.append(entry)
 p.csvout(out/'dataset/manifest.csv',manifest)
 for f in sorted((out/'dataset/upper').glob('*.png')):hashes[str(f.relative_to(out))]=p.sha(f)
 hashes['dataset/manifest.csv']=p.sha(out/'dataset/manifest.csv');p.js(out/'dataset/provenance.json',dict(source_progress_sha256=p.sha(out/'original_human_progress.json'),files=hashes))
 p.js(out/'dataset/summary.json',dict(source_images=4,positions=1,training_region_images=12,validation_region_images=4,repeated_C4_region_images=4,upper_pixels=[2541,1108],training_draws_per_model=256,training_native_size=256,epochs=8,samples_per_epoch=32,
  training_unique_period_windows=len({(r['period'],r['x'],r['y']) for r in records if r['partition']=='train'}),
  validation_patches_per_model=40,repeated_C4_patches_per_model=40,U_Net_inference_patch_size=384,other_models_inference_patch_size=256,
  exported_validation_patches=80,exported_repeated_C4_patches=80,external_independent_test_images=0,training_pixels_reused_across_models=True,
  annotation_SHA256=p.sha(out/'original_human_progress.json'),labels_edited=False,death_not_used=True,temporal_attributes_preserved_not_trained=True,
  grouping='All four periods share train C0-C2 / validation C3 / repeated benchmark C4;64px guards. Augmented or overlapping crops are not independent source images.',
  training_recipe='Native RGB crops are saved before photometric/geometric augmentation; exact replay parameters are in manifest. Root/background only.',
  background='One class for soil/droplets/debris; no background subtype annotation exists. Unpainted upper pixels accepted as background under user declaration.'))
 p.js(out/'dataset/verification.json',dict(passed=True,exact_training_replay_checks=checks,lossless_crop_checks=len(records)*5,all_periods_share_spatial_split=True,all_source_hashes_verified=True))
 print('V9_DATA_READY',len(records),checks,flush=True)
if __name__=='__main__':main()

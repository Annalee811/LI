#!/usr/bin/env python3
"""RoMa refinement guided by Z2 soil crack structures; all output RGB is sampled from originals."""
from pathlib import Path
import os,sys,json,gc,shutil
os.environ.setdefault('TORCH_HOME','/workspace/.cache/torch')
import cv2,numpy as np,torch
from PIL import Image
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
import register_roma as reg
import evaluate_continuous_registration as v
from run_0353_1_root_priority import postprocess
ROOT=Path.cwd()/'results/microtube_registration';POSITION='A017-1-Z2';RAW=ROOT/'input/raw/V-变化'/POSITION
OUT=ROOT/'A017-1_Z2_soil_shapes';cv2.setNumThreads(4);torch.set_num_threads(4);torch.manual_seed(20261013)

def expand_crop_prediction(warp,certainty,roi,wh):
 """Convert cropped normalized coordinates and their query domains to the full inference canvas."""
 batch=warp.ndim==4
 raw=warp.detach().cpu().numpy();conf=certainty.detach().cpu().numpy()
 if batch:raw,conf=raw[0],conf[0]
 hh,ww=raw.shape[0],raw.shape[1]//2;fw,fh=wh;x0,y0,x1,y1=roi;rw,rh=x1-x0,y1-y0
 y,x=np.mgrid[:hh,:ww].astype(np.float32);px=(x+.5)*fw/ww-.5;py=(y+.5)*fh/hh-.5
 qx=((px-x0+.5)*ww/rw-.5).astype(np.float32);qy=((py-y0+.5)*hh/rh-.5).astype(np.float32)
 domain=(px>=x0)&(px<=x1-1)&(py>=y0)&(py<=y1-1)
 parts=[];cs=[]
 for j in [0,1]:
  a=raw[:,j*ww:(j+1)*ww].copy()
  for channel in [0,2]:a[...,channel]=(a[...,channel]+1)*rw/fw+2*x0/fw-1
  for channel in [1,3]:a[...,channel]=(a[...,channel]+1)*rh/fh+2*y0/fh-1
  a=cv2.remap(a,qx,qy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(-3,-3,-3,-3));a[~domain]=-3
  c=cv2.remap(conf[:,j*ww:(j+1)*ww],qx,qy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0);c[~domain]=0
  parts.append(a);cs.append(c)
 a=np.concatenate(parts,axis=1);c=np.concatenate(cs,axis=1)
 if batch:a,c=a[None],c[None]
 return torch.as_tensor(a,device=warp.device,dtype=warp.dtype),torch.as_tensor(c,device=certainty.device,dtype=certainty.dtype)

class SharedROIModel:
 def __init__(self,base,roi):self.base=base;self.roi=roi
 def normalize(self,value):
  im=value if isinstance(value,Image.Image) else Image.open(value).convert('RGB')
  a=np.array(im.crop(tuple(self.roi)).convert('RGB'));lab=cv2.cvtColor(a,cv2.COLOR_RGB2LAB)
  lab[...,0]=cv2.createCLAHE(2,(12,12)).apply(lab[...,0]);return Image.fromarray(cv2.cvtColor(lab,cv2.COLOR_LAB2RGB))
 def match(self,a,b,**kw):
  wh=a.size if isinstance(a,Image.Image) else Image.open(a).size
  warp,c=self.base.match(self.normalize(a),self.normalize(b),**kw)
  return expand_crop_prediction(warp,c,self.roi,wh)
 def sample(self,*a,**kw):return self.base.sample(*a,**kw)

def shared_roi(matrix,source_wh,target_wh,soil_band=False):
 source_mask=np.ones(source_wh[::-1],np.uint8)
 if soil_band:source_mask[:round(source_wh[1]*.58)]=0
 mask=cv2.warpPerspective(source_mask,matrix,target_wh,flags=cv2.INTER_NEAREST)>0
 ys=np.flatnonzero(mask.any(1));xs=np.flatnonzero(mask.any(0));x0,x1=int(xs[0]),int(xs[-1])+1;y0,y1=int(ys[0]),int(ys[-1])+1
 rect=v.rectangle(mask[y0:y1,x0:x1]);rect=[rect[0]+x0,rect[1]+y0,rect[2]+x0,rect[3]+y0]
 assert mask[rect[1]:rect[3],rect[0]:rect[2]].all();return rect

def coordinate_check():
 # An identity matcher on a cropped canvas must remain identity at full-canvas coordinates.
 h,w=48,64;y,x=np.mgrid[:h,:w].astype(np.float32);xy=np.stack([(x+.5)*2/w-1,(y+.5)*2/h-1],-1)
 raw=np.concatenate([np.dstack([xy,xy]),np.dstack([xy,xy])],axis=1)
 a,c=expand_crop_prediction(torch.from_numpy(raw),torch.ones(h,2*w),[40,20,280,180],(320,240));a=a.numpy();c=c.numpy()
 yy,xx=np.mgrid[:h,:w];px=(xx+.5)*320/w-.5;py=(yy+.5)*240/h-.5
 good=(px>45)&(px<275)&(py>25)&(py<175)&(c[:,:w]>.99)
 predicted=(a[:,:w,:2]+1)*np.array([160,120])-.5
 error=np.linalg.norm(predicted-np.stack([px,py],-1),axis=-1)
 assert error[good].max()<.2,float(error[good].max());print('PASS cropped-to-full identity coordinate check',float(error[good].max()),flush=True)

def main():
 import argparse
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pair',choices=['T1_to_T2','T2_to_T4']);parser.add_argument('--soil-band-only',action='store_true');args_cli=parser.parse_args()
 coordinate_check()
 rows=json.loads((OUT/'soil_shape_candidates.json').read_text())
 h14=np.array(next(d['matrix_native'] for d in rows if d['mode']=='clahe' and d['pair']=='T1_to_T4'))
 h24=np.array(next(d['matrix_native'] for d in rows if d['mode']=='clahe' and d['pair']=='T2_to_T4'))
 matrices={'T1_to_T2':np.linalg.inv(h24)@h14,'T2_to_T4':h24}
 # T4 is a coarse localization bridge only; final dense fields still follow T1 -> T2 -> T4.
 (OUT/'selected_soil_coarse_priors.json').write_text(json.dumps({k:m.tolist() for k,m in matrices.items()},indent=2))
 from romatch import roma_outdoor
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_float32_matmul_precision('highest')
 model=roma_outdoor(device=device,coarse_res=560,upsample_res=864,symmetric=True,use_custom_corr=False,upsample_preds=True)
 selected=[]
 for name,mat in matrices.items():
  if args_cli.pair and name!=args_cli.pair:continue
  ta,tb=[int(s[1:]) for s in name.split('_to_')];src=RAW/f'T{ta}.jpg';tgt=RAW/f'T{tb}.jpg'
  roi=shared_roi(mat,Image.open(src).size,Image.open(tgt).size,args_cli.soil_band_only);folder=OUT/('soil_band_pairs' if args_cli.soil_band_only else 'refined_pairs')/name
  print('SOIL ROI',name,roi,'matrix',mat.tolist(),flush=True)
  args=SimpleNamespace(prealignments={name:mat},device=device,confidence=.5,cycle=3.,samples=5000,seed=20261013,ransac=3.,coarse=560,upsample=864,tile=128)
  b,c,valid,m=reg.pair(SharedROIModel(model,roi),src,tgt,folder,args);del b,c,valid;gc.collect()
  m.update(inference_shared_roi_xyxy=roi,inference_source_lower_soil_band=args_cli.soil_band_only,inference_only_normalization='Both inference crops use identical L-channel CLAHE; final RGB sampled once from original full-size RGB.',coarse_method='Stable soil crack descriptors; T4 bridge localizes T1/T2; final dense sequence T1->T2->T4',calendar_gap_periods=tb-ta-1)
  if tb-ta>1:m['kind']='consecutive_available_periods_with_missing_intermediate'
  postprocess(folder,ROOT/'A017-1_root_priority_configuration'/POSITION,0,m)
  baseline=ROOT/POSITION/name;old=json.loads((baseline/'metrics.json').read_text());new=json.loads((folder/'metrics.json').read_text())
  usable=new['effective_area_ratio']>=.01 and new['reliable_sampled_matches']>=100 and new['cycle_error_effective_target_px']['p95'] is not None and new['cycle_error_effective_target_px']['p95']<=3
  if usable and new['effective_pixel_count']>old['effective_pixel_count']:
   backup=OUT/'baseline_pairs'/name
   if not backup.exists():shutil.copytree(baseline,backup)
   for f in folder.iterdir():
    if f.is_file():shutil.copy2(f,baseline/f.name)
   selected.append(name);print('SELECTED_SOIL_ROI',name,new['effective_area_ratio'],flush=True)
  else:print('KEPT_BASELINE',name,new['effective_area_ratio'],flush=True)
 p=ROOT/POSITION/'manifest.json';d=json.loads(p.read_text());selected=sorted(set(selected+d.get('soil_shape_refinement',{}).get('selected_pairs',[])));d['soil_shape_refinement']=dict(selected_pairs=selected,coarse_priors={k:m.tolist() for k,m in matrices.items()},model_inference='Shared geometry ROI + L-channel CLAHE, inference only; fields embedded in native full canvases, outside ROI unknown/confidence0',manual_landmark_TRE=None)
 p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
 print('FINISHED',selected,flush=True)

if __name__=='__main__':main()

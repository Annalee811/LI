#!/usr/bin/env python3
"""Targeted T1/T2 retry: conservative coarse localization and inference-only illumination normalization."""
from pathlib import Path
import os,sys,json,gc,shutil
os.environ.setdefault('TORCH_HOME','/workspace/.cache/torch');os.environ.setdefault('MPLCONFIGDIR','/workspace/.cache/matplotlib')
import numpy as np,cv2,torch
from PIL import Image
from scipy.signal import fftconvolve
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
import register_roma as reg
from run_0353_1_root_priority import support,postprocess,read,weighted_similarity,map_points
import argparse
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--position",nargs="+",required=True);args_cli=parser.parse_args()
ROOT=Path.cwd()/'results/microtube_registration';INPUT=ROOT/'input/raw/V-变化';family='-'.join(args_cli.position[0].split('-')[:2]);CACHE=ROOT/f'{family}_root_priority_configuration';OUT=ROOT/f'{family}_root_priority_refinements'
cv2.setNumThreads(2);torch.set_num_threads(2);torch.manual_seed(20261013);cv2.setRNGSeed(20261013)


def coarse(paths,position):
 prepared=[]
 for f in paths:
  a=read(f);h,w=a.shape[:2];nh=round(1400*h/w);a=cv2.resize(a,(1400,nh),interpolation=cv2.INTER_AREA if w>1400 else cv2.INTER_CUBIC);g=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
  sift=cv2.SIFT_create(nfeatures=0,contrastThreshold=.015,edgeThreshold=12);keys,d=sift.detectAndCompute(g,None);d=np.sqrt(d/(d.sum(1,keepdims=True)+1e-7));prepared.append((keys,d,np.array([w/1400,h/nh])))
 ka,da,sa=prepared[0];kb,db,sb=prepared[1];mat=None;note={}
 matcher=cv2.BFMatcher();ab=matcher.knnMatch(da,db,2);ba=matcher.knnMatch(db,da,2)
 reverse={m.queryIdx:m.trainIdx for m,n in ba if m.distance<.75*n.distance};ms=[m for m,n in ab if m.distance<.75*n.distance and reverse.get(m.trainIdx)==m.queryIdx]
 if len(ms)>=6:
  pa=np.array([ka[m.queryIdx].pt for m in ms],np.float32);pb=np.array([kb[m.trainIdx].pt for m in ms],np.float32)
  candidate,mask=cv2.estimateAffinePartial2D(pa,pb,method=cv2.RANSAC,ransacReprojThreshold=2.5,maxIters=20000,confidence=.999)
  ok=mask.ravel()>0 if mask is not None else np.zeros(len(ms),bool)
  if candidate is not None and ok.sum()>=8 and .8<np.linalg.norm(candidate[0,:2])<1.2:
   ra,_=support(read(paths[0]));rb,_=support(read(paths[1]));roots=map_points(ra,pa)&map_points(rb,pb)
   candidate=weighted_similarity(pa[ok],pb[ok],np.where(roots[ok],3.,1.))
   mat=np.diag([*sb,1])@np.vstack([candidate,[0,0,1]])@np.diag([*(1/sa),1])
   note=dict(method='RootSIFT low-contrast detection, same ratio .75; root-weighted consensus',feature_matches=len(ms),inliers=int(ok.sum()),root_inliers=int((roots&ok).sum()),root_weight=3,soil_weight=1)
 if mat is None:
  hs=[]
  for f in paths:
   a=cv2.resize(read(f),(700,624),interpolation=cv2.INTER_AREA);g=cv2.createCLAHE(2,(10,10)).apply(cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)).astype(np.float32)
   a=np.clip(np.maximum(g-cv2.GaussianBlur(g,(0,0),3),0),0,35);a[:20]=0;a[-20:]=0;a[:,:20]=0;a[:,-20:]=0;hs.append(a)
  a,b=hs;corr=fftconvolve(b,a[::-1,::-1],mode='full');ones=np.ones_like(a);count=fftconvolve(ones,ones,mode='full');ea=fftconvolve(ones,a[::-1,::-1]**2,mode='full');eb=fftconvolve(b**2,ones,mode='full');ncc=corr/np.sqrt(np.maximum(ea*eb,1));ncc[count<a.size*.12]=-1
  xx=np.arange(ncc.shape[1])-(a.shape[1]-1);ncc[:,np.abs(xx)>105]=-1
  y,x=np.unravel_index(ncc.argmax(),ncc.shape);dx=x-(a.shape[1]-1);dy=y-(a.shape[0]-1)
  swh=Image.open(paths[0]).size;twh=Image.open(paths[1]).size
  mat=np.diag([twh[0]/700,twh[1]/624,1])@np.array([[1,0,dx],[0,1,dy],[0,0,1]])@np.diag([700/swh[0],624/swh[1],1])
  note=dict(method='Bright-line highpass correlation coarse hypothesis; not independent evidence',correlation_peak=float(ncc[y,x]),analysis_translation_xy=[int(dx),int(dy)],analysis_size=[700,624])
 note['source_to_target_native']=mat.tolist();return mat,note


class IlluminationNormalizedModel:
 def __init__(self,base):self.base=base
 def normalize(self,value):
  a=np.array(value.convert('RGB')) if isinstance(value,Image.Image) else read(value)
  # Same L-channel contrast transform on both inference inputs; output originals are untouched.
  lab=cv2.cvtColor(a,cv2.COLOR_RGB2LAB);lab[...,0]=cv2.createCLAHE(2,(12,12)).apply(lab[...,0]);a=cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)
  return Image.fromarray(a)
 def match(self,a,b,**kw):return self.base.match(self.normalize(a),self.normalize(b),**kw)
 def sample(self,*args,**kw):return self.base.sample(*args,**kw)


def main():
 from romatch import roma_outdoor
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_float32_matmul_precision('highest')
 model=roma_outdoor(device=device,coarse_res=560,upsample_res=864,symmetric=True,use_custom_corr=False,upsample_preds=True)
 for position in args_cli.position:
  raw=INPUT/position;paths=[next(f for f in raw.iterdir() if f.stem==f'T{i}') for i in [1,2]];matrix,note=coarse(paths,position);folder=OUT/position/'T1_to_T2';folder.mkdir(parents=True,exist_ok=True)
  print(position,'RETRY_CONFIG',json.dumps(note),flush=True)
  args=SimpleNamespace(prealignments={'T1_to_T2':matrix},device=device,confidence=.5,cycle=3.,samples=5000,seed=20261013,ransac=3.,coarse=560,upsample=864,tile=128)
  b,c,v,m=reg.pair(IlluminationNormalizedModel(model),paths[0],paths[1],folder,args);del b,c,v;gc.collect()
  m['inference_only_normalization']='identical fixed L-channel CLAHE clip=2, grid=12x12; original RGB sampled once for final outputs'
  m['root_priority_retry']=note;postprocess(folder,CACHE/position,0,m)
  (folder/'retry_configuration.json').write_text(json.dumps(note,indent=2))
  baseline=ROOT/position/'T1_to_T2';bm=json.loads((baseline/'metrics.json').read_text());nm=json.loads((folder/'metrics.json').read_text())
  if nm['effective_area_ratio']>=.01 and nm['reliable_sampled_matches']>=100 and (nm['cycle_error_effective_target_px']['p95'] or float('inf'))<=3 and nm['effective_pixel_count']>bm['effective_pixel_count'] and (nm.get('root_candidate_effective_area_ratio') or 0)>=(bm.get('root_candidate_effective_area_ratio') or 0):
   backup=OUT/position/'baseline_T1_to_T2';shutil.copytree(baseline,backup,dirs_exist_ok=True)
   for f in folder.iterdir():
    if f.is_file():shutil.copy2(f,baseline/f.name)
   manifest=ROOT/position/'manifest.json';d=json.loads(manifest.read_text());d['first_pair_refinement']=dict(configuration=note,baseline_effective_ratio=bm['effective_area_ratio'],selected_effective_ratio=nm['effective_area_ratio'],inference_normalization=m['inference_only_normalization']);manifest.write_text(json.dumps(d,ensure_ascii=False,indent=2))
   print('SELECTED_RETRY',position,nm['effective_area_ratio'],flush=True)
  else:print('KEPT_BASELINE',position,bm['effective_area_ratio'],nm['effective_area_ratio'],flush=True)

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Full RoMa for 0353-1 with root-prioritized coarse anchors and artifact rejection.

Root support is a conservative bright-ridge geometry proxy, not validated root
segmentation. Raw RGB is used for dense RoMa and output RGB is sampled from originals.
"""
from pathlib import Path
import os,sys,json,hashlib,gc,time
os.environ.setdefault('TORCH_HOME','/workspace/.cache/torch')
os.environ.setdefault('MPLCONFIGDIR','/workspace/.cache/matplotlib')
import cv2,numpy as np,torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
import register_roma as registration
from evaluate_adjacent_registration import features

REPO=Path.cwd().resolve();ROOT=REPO/'results/microtube_registration'
INPUT=ROOT/'input/raw/V-变化';POSITIONS=['0353-1-B3','0353-1-Z2']
CACHE=ROOT/'0353-1_root_priority_configuration';CACHE.mkdir(exist_ok=True)
cv2.setNumThreads(4);torch.set_num_threads(4);torch.manual_seed(20261013);cv2.setRNGSeed(20261013)


def read(path):return np.array(Image.open(path).convert('RGB'))


def support(rgb):
 h,w=rgb.shape[:2];nw=1400;nh=round(nw*h/w)
 a=cv2.resize(rgb,(nw,nh),interpolation=cv2.INTER_AREA if w>nw else cv2.INTER_CUBIC)
 gray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
 gray=cv2.createCLAHE(1.4,(12,12)).apply(gray).astype(np.float32)/255
 response=np.zeros_like(gray)
 for sigma in [1.,1.6,2.5,3.8]:
  xx=gaussian_filter(gray,sigma,order=(0,2))*sigma**2
  yy=gaussian_filter(gray,sigma,order=(2,0))*sigma**2
  xy=gaussian_filter(gray,sigma,order=(1,1))*sigma**2
  delta=np.sqrt((xx-yy)**2+4*xy**2)
  l1=(xx+yy-delta)/2;l2=(xx+yy+delta)/2
  swap=np.abs(l1)>np.abs(l2);small=np.where(swap,l2,l1);large=np.where(swap,l1,l2)
  ratio=np.abs(small)/(np.abs(large)+1e-7)
  strength=np.sqrt(small**2+large**2);scale=max(float(np.percentile(strength,98)),.008)
  vessel=np.exp(-ratio**2/(2*.45**2))*(1-np.exp(-strength**2/(2*scale**2)))
  vessel[large>=0]=0;response=np.maximum(response,vessel)
 binary=(response>=.18).astype(np.uint8)
 n,lab,stats,_=cv2.connectedComponentsWithStats(binary,8);mask=np.zeros_like(binary)
 for j in range(1,n):
  x,y,ww,hh,area=stats[j]
  if area<18 or max(ww,hh)<15:continue
  yy,xx=np.where(lab==j);ev=np.linalg.eigvalsh(np.cov(np.stack([xx,yy])))
  elongation=(ev[-1]+1)/(ev[0]+1)
  if elongation>=3.5 or (area>=200 and max(ww,hh)>=45):mask[lab==j]=1
 mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))>0
 # Compact near-white specular components; no dark root-shaped soil cracks removed.
 rawgray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
 bright=(rawgray>=245).astype(np.uint8)
 contours,_=cv2.findContours(bright,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 artifact=np.zeros_like(bright)
 for contour in contours:
  area=cv2.contourArea(contour);per=cv2.arcLength(contour,True)
  x,y,ww,hh=cv2.boundingRect(contour)
  circularity=4*np.pi*area/(per**2+1e-6)
  if area>=24 and min(ww,hh)>=6 and max(ww,hh)/max(min(ww,hh),1)<1.8 and circularity>=.6:
   cv2.drawContours(artifact,[contour],-1,1,-1)
 artifact=cv2.dilate(artifact,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))>0
 mask&=~artifact
 return mask,artifact


def map_points(mask,points):
 return cv2.remap(mask.astype(np.uint8),points[None,:,0].astype(np.float32),points[None,:,1].astype(np.float32),cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)[0]>0


def weighted_similarity(pa,pb,weights):
 # x'=a*x-b*y+tx; y'=b*x+a*y+ty. Root-supported anchors weight 3x.
 n=len(pa);A=np.zeros((2*n,4),np.float64);b=pb.reshape(-1).astype(np.float64)
 A[0::2,0]=pa[:,0];A[0::2,1]=-pa[:,1];A[0::2,2]=1
 A[1::2,0]=pa[:,1];A[1::2,1]=pa[:,0];A[1::2,3]=1
 sqrtw=np.repeat(np.sqrt(weights),2)
 x=np.linalg.lstsq(A*sqrtw[:,None],b*sqrtw,rcond=None)[0]
 return np.array([[x[0],-x[1],x[2]],[x[1],x[0],x[3]]])


def configure(position):
 raw=INPUT/position;paths=sorted((f for f in raw.iterdir() if f.stem.upper() in ['T1','T2','T3','T4']),key=lambda f:int(f.stem[1:]))
 period_ids=[int(f.stem[1:]) for f in paths]
 out=CACHE/position;out.mkdir(exist_ok=True)
 masks=[];artifacts=[];feature_data=[]
 for i,path in enumerate(paths):
  a=read(path);r,bad=support(a);masks.append(r);artifacts.append(bad)
  native_r=cv2.resize(r.astype(np.uint8),tuple(reversed(a.shape[:2])),interpolation=cv2.INTER_NEAREST)>0
  native_bad=cv2.resize(bad.astype(np.uint8),tuple(reversed(a.shape[:2])),interpolation=cv2.INTER_NEAREST)>0
  Image.fromarray(native_r.astype(np.uint8)*255).save(out/f'T{period_ids[i]}_root_support_candidates.png')
  Image.fromarray(native_bad.astype(np.uint8)*255).save(out/f'T{period_ids[i]}_compact_specular_mask.png')
  feature_data.append(features(path))
 prealign={};rows=[]
 for i in range(len(paths)-1):
  ka,da,sa,_=feature_data[i];kb,db,sb,_=feature_data[i+1]
  matcher=cv2.BFMatcher(cv2.NORM_L2);ab=matcher.knnMatch(da,db,2);ba=matcher.knnMatch(db,da,2)
  reverse={m.queryIdx:m.trainIdx for m,n in ba if m.distance<.75*n.distance}
  matches=[m for m,n in ab if m.distance<.75*n.distance and reverse.get(m.trainIdx)==m.queryIdx]
  pa=np.array([ka[m.queryIdx].pt for m in matches],np.float32);pb=np.array([kb[m.trainIdx].pt for m in matches],np.float32)
  name=f'T{period_ids[i]}_to_T{period_ids[i+1]}';row=dict(pair=name,mutual_feature_matches=len(matches),root_candidate_anchors=0,coarse_alignment_used=False,calendar_gap_periods=period_ids[i+1]-period_ids[i]-1)
  if len(matches)>=6:
   ok=~map_points(artifacts[i],pa)&~map_points(artifacts[i+1],pb);pa,pb=pa[ok],pb[ok]
   root=map_points(masks[i],pa)&map_points(masks[i+1],pb);row['root_candidate_anchors']=int(root.sum())
   model,inlier=None,None;selection='root-prioritized joint robust fit'
   if root.sum()>=8:
    rm,ri=cv2.estimateAffinePartial2D(pa[root],pb[root],method=cv2.RANSAC,ransacReprojThreshold=2.5,maxIters=20000,confidence=.999)
    if rm is not None:
     residual=np.linalg.norm(pa@rm[:,:2].T+rm[:,2]-pb,axis=1);consensus=residual<=2.5
     if (consensus&root).sum()>=6 and consensus.sum()>=10 and np.ptp(pa[consensus],axis=0).max()>=100:
      model,inlier=rm,consensus;selection='root-first RANSAC + weighted root/soil refinement'
   if model is None:
    model,im=cv2.estimateAffinePartial2D(pa,pb,method=cv2.RANSAC,ransacReprojThreshold=2.5,maxIters=20000,confidence=.999)
    if im is not None:inlier=im.ravel().astype(bool)
   if model is not None and inlier is not None and inlier.sum()>=8:
    model=weighted_similarity(pa[inlier],pb[inlier],np.where(root[inlier],3.,1.))
    scale=np.linalg.norm(model[0,:2]);angle=np.degrees(np.arctan2(model[1,0],model[0,0]))
    if .8<=scale<=1.2 and abs(angle)<=20:
     hm=np.vstack([model,[0,0,1]])
     hm=np.diag([*sb,1])@hm@np.diag([*(1/sa),1]);prealign[name]=hm.tolist()
     row.update(coarse_alignment_used=True,coarse_method=selection,coarse_inliers=int(inlier.sum()),coarse_root_inliers=int((inlier&root).sum()),root_anchor_weight=3.,soil_anchor_weight=1.,homography_native_source_to_target=hm.tolist())
  rows.append(row);print(position,'COARSE',json.dumps(row,ensure_ascii=False),flush=True)
 (out/'prealign.json').write_text(json.dumps(prealign,indent=2));(out/'root_priority_configuration.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
 return paths,prealign,out,rows


def postprocess(folder,config,i,meta):
 source_period=Path(meta['source_file']).stem.upper();target_period=Path(meta['target_file']).stem.upper()
 with np.load(folder/'dense_transform.npz') as z:fields={k:z[k] for k in z.files}
 source_root=np.array(Image.open(config/f'{source_period}_root_support_candidates.png'))>0
 target_root=np.array(Image.open(config/f'{target_period}_root_support_candidates.png'))>0
 source_art=np.array(Image.open(config/f'{source_period}_compact_specular_mask.png'))>0
 target_art=np.array(Image.open(config/f'{target_period}_compact_specular_mask.png'))>0
 warp_art=registration.remap(source_art.astype(np.uint8),fields['target_to_source_xy_px'],nearest=True)>0
 artifact=target_art|warp_art
 old=fields['valid_target'];valid=old&~artifact;fields['valid_target']=valid
 Image.fromarray(valid.astype(np.uint8)*255).save(folder/'valid_overlap_mask.png')
 Image.fromarray(artifact.astype(np.uint8)*255).save(folder/'artifact_exclusion_mask.png')
 np.savez_compressed(folder/'dense_transform.npz',**fields)
 meta['artifact_excluded_effective_pixels']=int((old&artifact).sum());meta['effective_area_ratio']=float(valid.mean());meta['effective_pixel_count']=int(valid.sum())
 meta['root_candidate_target_area_ratio']=float(target_root.mean())
 meta['root_candidate_effective_area_ratio']=float(valid[target_root].mean()) if target_root.any() else None
 meta['root_priority_configuration_file']=str(config/'root_priority_configuration.json')
 meta['root_semantics']='Conservative ridge/length proxy used to prioritize coarse anchors. Not verified root segmentation.'
 (folder/'metrics.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
 print('ROOT_QC',folder.parent.name,meta['pair'],'valid',meta['effective_area_ratio'],'root supported valid',meta['root_candidate_effective_area_ratio'],flush=True)
 del fields;gc.collect()


def main():
 configs=[]
 for position in POSITIONS:configs.append((position,*configure(position)))
 torch.set_float32_matmul_precision('highest');device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 from romatch import roma_outdoor
 print('LOADING FULL ROMA',device,flush=True)
 model=roma_outdoor(device=device,coarse_res=560,upsample_res=864,symmetric=True,use_custom_corr=False,upsample_preds=True)
 for position,paths,prealign,config,rows in configs:
  period_ids=[int(f.stem[1:]) for f in paths]
  dest=ROOT/position;dest.mkdir(exist_ok=True)
  args=SimpleNamespace(prealignments=prealign,device=device,confidence=.5,cycle=3.,samples=5000,seed=20261013,ransac=3.,coarse=560,upsample=864,tile=128)
  manifest=dict(position=position,period_ids=period_ids,missing_period_ids=sorted(set(range(1,5))-set(period_ids)),input_sha256={str(f.relative_to(REPO)):hashlib.file_digest(f.open('rb'),'sha256').hexdigest() for f in paths},
   args=dict(device=str(device),coarse=560,upsample=864,confidence=.5,cycle_native_target_px=3.,seed=20261013),
   model='full roma_outdoor',root_priority=rows,method='Root-first geometric prealignment from conservative root ridge support; 3:1 root/soil weight; full RoMa dense residuals on RGB; artifact rejection. RGB output sampled once from originals.')
  (dest/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
  for i in range(len(paths)-1):
   folder=dest/f'T{period_ids[i]}_to_T{period_ids[i+1]}'
   if (folder/'root_priority_completed.json').exists():print('REUSE',folder,flush=True);continue
   b,c,v,m=registration.pair(model,paths[i],paths[i+1],folder,args)
   m['calendar_gap_periods']=period_ids[i+1]-period_ids[i]-1
   if m['calendar_gap_periods']:m['kind']='consecutive_available_periods_with_missing_intermediate'
   del b,c,v;gc.collect();postprocess(folder,config,i,m)
   (folder/'root_priority_completed.json').write_text(json.dumps(dict(status='completed',input_hashes=manifest['input_sha256'],root_configuration=rows[i]),ensure_ascii=False,indent=2))
  print('POSITION DONE',position,flush=True)

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Bounded camera-orientation diagnostic; original RGB is not changed or rendered.
RoMa internal candidate confidence and global consensus are not ground-truth accuracy.
"""
from pathlib import Path
import argparse,os,json,time,gc,hashlib
os.environ.setdefault('TORCH_HOME','/workspace/.cache/torch')
import numpy as np,cv2,torch
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--position',required=True);p.add_argument('--source-period',type=int,default=3);p.add_argument('--target-period',type=int,default=4);p.add_argument('--threads',type=int,default=2);p.add_argument('--coarse',type=int,default=336);p.add_argument('--upsample',type=int,default=448);args=p.parse_args();root=Path(__file__).resolve().parent;raw=root/'input/raw/V-变化'/args.position;files={int(f.stem[1:]):f for f in raw.glob('T*')};out=root/(args.position+'_orientation_trials');out.mkdir(exist_ok=True);cv2.setNumThreads(args.threads);torch.set_num_threads(args.threads);torch.manual_seed(20261013);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from romatch import roma_outdoor
model=roma_outdoor(device=device,coarse_res=args.coarse,upsample_res=args.upsample,symmetric=True,use_custom_corr=False,upsample_preds=True);A=Image.open(files[args.source_period]).convert('RGB');original_B=Image.open(files[args.target_period]).convert('RGB');wa,ha=A.size;wb0,hb0=original_B.size;rows=[]
def sample(a,p):return cv2.remap(a,p[None,:,0].astype('float32'),p[None,:,1].astype('float32'),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)[0]
for angle in [0,90,180,270]:
 f=out/f'T{args.source_period}_to_T{args.target_period}_rot{angle}';f.mkdir(exist_ok=True)
 if (f/'metrics.json').exists():rows.append(json.loads((f/'metrics.json').read_text()));continue
 B=original_B if angle==0 else original_B.transpose({90:Image.Transpose.ROTATE_90,180:Image.Transpose.ROTATE_180,270:Image.Transpose.ROTATE_270}[angle]);wb,hb=B.size;start=time.time();print('ROTATION_TRIAL_START',args.position,angle,flush=True)
 with torch.inference_mode():warp,c=model.match(A,B,device=device)
 if warp.ndim==4:warp,c=warp[0],c[0]
 data=warp.cpu().float().numpy();conf=c.cpu().float().numpy();mid=data.shape[1]//2;h=data.shape[0];torch.manual_seed(20261013);m,scores=model.sample(warp,c,num=5000);m=m.cpu().numpy();scores=scores.cpu().numpy();ia=(m[:,:2]+1)*[mid/2,h/2]-.5;ib=(m[:,2:]+1)*[mid/2,h/2]-.5;sc=sample(conf[:,:mid],ia);tc=sample(conf[:,mid:],ib);back=sample(data[:,mid:,:2],ib);forward=sample(data[:,:mid,2:],ia);sa=np.linalg.norm((back-m[:,:2])*[800,ha*800/wa],axis=1);sb=np.linalg.norm((forward-m[:,2:])*[800,hb*800/wb],axis=1);pa=(m[:,:2]+1)*[wa/2,ha/2]-.5;pb=(m[:,2:]+1)*[wb/2,hb/2]-.5
 if angle==90:pb=np.stack([wb0-1-pb[:,1],pb[:,0]],axis=-1)
 elif angle==180:pb=np.stack([wb0-1-pb[:,0],hb0-1-pb[:,1]],axis=-1)
 elif angle==270:pb=np.stack([pb[:,1],hb0-1-pb[:,0]],axis=-1)
 valid=(pa[:,0]>=0)&(pa[:,0]<=wa-1)&(pa[:,1]>=0)&(pa[:,1]<=ha-1)&(pb[:,0]>=0)&(pb[:,0]<=wb0-1)&(pb[:,1]>=0)&(pb[:,1]<=hb0-1)&(sc>=.5)&(tc>=.5)&(sa<=5)&(sb<=5);H=None;k=None
 if valid.sum()>=6:
  H,k=cv2.estimateAffinePartial2D((pa[valid]*1600/wa).astype('float32'),(pb[valid]*1600/wb0).astype('float32'),method=cv2.RANSAC,ransacReprojThreshold=10,maxIters=30000,confidence=.999)
 d=dict(position=args.position,pair=f'T{args.source_period}_to_T{args.target_period}',inference_target_rotation_CCW_deg=angle,device=str(device),coarse_res=args.coarse,upsample_res=args.upsample,internally_supported_candidates=int(valid.sum()),global_similarity_consensus=None if k is None else int(k.sum()),global_matrix_analysis1600=None if H is None else np.vstack([H,[0,0,1]]).tolist(),elapsed_seconds=round(time.time()-start,2),input_sha256={f'T{t}':hashlib.sha256(files[t].read_bytes()).hexdigest() for t in [args.source_period,args.target_period]},interpretation='Target rotation is for orientation testing only. Coordinates converted back to original target pixel centres. Confidence/cycle/consensus are internal evidence, not verified region correctness. No original RGB colors or pixels changed.');rows.append(d);np.savez_compressed(f/'sampled_matches.npz',source_xy=pa,target_xy=pb,reliable=valid,source_confidence=sc,target_confidence=tc,roma_sample_score=scores);(f/'metrics.json').write_text(json.dumps(d,indent=2));print('ROTATION_TRIAL_DONE',json.dumps(d),flush=True);del warp,c,data,conf,m;gc.collect()
(out/'orientation_trial_summary.json').write_text(json.dumps(rows,indent=2))

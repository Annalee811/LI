#!/usr/bin/env python3
"""Compact full-RoMa region proposals; original RGB is never rendered with dense warps.
Save native-coordinate candidates and honest bidirectional model evidence.
"""
from pathlib import Path
import argparse,json,hashlib,time,gc,os
os.environ.setdefault('TORCH_HOME','/workspace/.cache/torch')
os.environ.setdefault('MPLCONFIGDIR','/workspace/.cache/matplotlib')
import torch,cv2,numpy as np
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--positions',nargs='+',required=True);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--threads',type=int,default=2);p.add_argument('--extra-pair');p.add_argument('--prealignment',type=Path);args=p.parse_args();root=args.root.resolve();cv2.setNumThreads(args.threads);torch.set_num_threads(args.threads);torch.manual_seed(20261013);cv2.setRNGSeed(20261013);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_float32_matmul_precision('highest')
from romatch import roma_outdoor
print('LOAD_FULL_ROMA',device,'threads',args.threads,flush=True);model=roma_outdoor(device=device,coarse_res=560,upsample_res=864,symmetric=True,use_custom_corr=False,upsample_preds=True)
def sample(a,coords):
 return cv2.remap(a,coords[None,:,0].astype('float32'),coords[None,:,1].astype('float32'),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)[0]
def project(a,H):return (a@H[:2,:2].T+H[:2,2]).astype('float32')
for pos in args.positions:
 raw=root/'input/raw/V-变化'/pos;files={int(f.stem[1:]):f for f in raw.glob('T*') if f.is_file()};periods=sorted(files)
 if len(periods)<=2:print('SKIP',pos,'<=2 periods',flush=True);continue
 dest=root/pos;dest.mkdir(exist_ok=True);hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files.values()};manifest=dest/'manifest.json'
 if manifest.exists():
  old=json.loads(manifest.read_text());expected={Path(k).stem:(v['sha256'] if isinstance(v,dict) else v) for k,v in old['input_sha256'].items()};assert all(expected.get(Path(k).stem)==v for k,v in hashes.items())
 else:manifest.write_text(json.dumps(dict(position=pos,period_ids=periods,missing_period_ids=sorted(set(range(1,5))-set(periods)),input_sha256=hashes,model='full roma_outdoor',method='RoMa region candidate evidence only; dense maps never applied to final RGB',args=dict(device=str(device),coarse=560,upsample=864,seed=20261013)),ensure_ascii=False,indent=2))
 pairs=[tuple(int(v[1:]) for v in args.extra_pair.split('_to_'))] if args.extra_pair else list(zip(periods,periods[1:]))
 for a,b in pairs:
  out=dest/f'T{a}_to_T{b}';out.mkdir(exist_ok=True);done=out/'compact_candidate_completed.json'
  if done.exists():assert json.loads(done.read_text())['input_sha256']==hashes;print('REUSE',pos,out.name,flush=True);continue
  if (out/'sampled_matches.npz').exists():print('PRESERVE_EXISTING_CACHE',pos,out.name,flush=True);continue
  start=time.time();A=Image.open(files[a]).convert('RGB');B=Image.open(files[b]).convert('RGB');wa,ha=A.size;wb,hb=B.size;infer_size=(wa,ha);H=None
  if args.prealignment:
   H=np.array(json.loads(args.prealignment.read_text())['matrices_native'][out.name]);scaled=np.diag([1600/wb]*2+[1])@H;align=cv2.warpAffine(np.array(A),scaled[:2],(1600,round(hb*1600/wb)),flags=cv2.INTER_LINEAR);A=Image.fromarray(align);B=B.resize(A.size,Image.Resampling.LANCZOS);infer_size=A.size
  print('ROMA_START',pos,out.name,'original WH',[wa,ha],[wb,hb],flush=True)
  with torch.inference_mode():warp,c=model.match(A,B,device=device)
  if warp.ndim==4:warp,c=warp[0],c[0]
  raw=warp.detach().cpu().float().numpy();confidence=c.detach().cpu().float().numpy();height=raw.shape[0];mid=raw.shape[1]//2;torch.manual_seed(20261013);matches,scores=model.sample(warp,c,num=5000);matches=matches.detach().cpu().numpy();scores=scores.detach().cpu().numpy();ia=(matches[:,:2]+1)*[mid/2,height/2]-.5;ib=(matches[:,2:]+1)*[mid/2,height/2]-.5;sc=sample(confidence[:,:mid],ia);tc=sample(confidence[:,mid:],ib);back=sample(raw[:,mid:,:2],ib);forward=sample(raw[:,:mid,2:],ia)
  source_cycle=np.linalg.norm((back-matches[:,:2])*[800,infer_size[1]*800/infer_size[0]],axis=1);target_cycle=np.linalg.norm((forward-matches[:,2:])*[800,hb*800/wb],axis=1);pa=(matches[:,:2]+1)*[infer_size[0]/2,infer_size[1]/2]-.5;pb=(matches[:,2:]+1)*[wb/2,hb/2]-.5
  if H is not None:
   # Match target B was uniformly resized to the inference canvas; normalized target coordinates remain valid.
   pa=project(pa,np.linalg.inv(scaled))
  inside=(pa[:,0]>=0)&(pa[:,0]<=wa-1)&(pa[:,1]>=0)&(pa[:,1]<=ha-1)&(pb[:,0]>=0)&(pb[:,0]<=wb-1)&(pb[:,1]>=0)&(pb[:,1]<=hb-1);ok=inside&(sc>=.5)&(tc>=.5)&(source_cycle<=5)&(target_cycle<=5)
  np.savez_compressed(out/'sampled_matches.npz',source_xy=pa,target_xy=pb,roma_sample_score=scores,source_confidence=sc,target_confidence=tc,reliable=ok,source_cycle_analysis1600_px=source_cycle,target_cycle_analysis1600_px=target_cycle)
  np.savez_compressed(out/'roma_raw_normalized.npz',warp=raw,confidence=confidence,inference_source_size_wh=infer_size,source_prealignment_to_inference=np.eye(3) if H is None else scaled)
  met=dict(pair=out.name,source_size_wh=[wa,ha],target_size_wh=[wb,hb],device=str(device),model='full roma_outdoor',coarse_res=560,upsample_res=864,sample_count=len(pa),bidirectionally_supported_candidates=int(ok.sum()),confidence_threshold=.5,cycle_threshold_analysis1600_px=5,prealignment_native=None if H is None else H.tolist(),elapsed_seconds=round(time.time()-start,2),interpretation='RoMa confidence/cycle are internal candidate evidence, not independent accuracy. Pixel-precision continuous success is not asserted. Dense RoMa is not used to render final RGB.')
  (out/'metrics.json').write_text(json.dumps(met,indent=2));done.write_text(json.dumps(dict(input_sha256=hashes,metrics=met),indent=2));print('ROMA_DONE',pos,json.dumps(met),flush=True);del A,B,warp,c,raw,confidence,matches,scores;gc.collect()
  if torch.cuda.is_available():torch.cuda.empty_cache()

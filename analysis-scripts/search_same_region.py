#!/usr/bin/env python3
"""Full-frame region verification for a selected microtube position.
RoMa already supplies dense localization hints; CLAHE/blur RootSIFT independently
checks complete RGB frames. Detection preprocessing never changes saved RGB.
"""
from pathlib import Path
import argparse,json,cv2,numpy as np
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--position',required=True);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--width',type=int,default=1000);p.add_argument('--threads',type=int,default=1);p.add_argument('--append',action='store_true');args=p.parse_args()
ROOT=args.root.resolve();RAW=ROOT/'input/raw/V-变化'/args.position;OUT=ROOT/(args.position+'_region_localization');OUT.mkdir(exist_ok=True);cv2.setNumThreads(args.threads);cv2.setRNGSeed(20261013)
paths={int(f.stem[1:]):f for f in RAW.glob('T*') if f.is_file()};ids=sorted(paths)
if len(ids)<=2:print('SKIPPED <=2 periods',args.position,flush=True);raise SystemExit(0)
images={t:np.array(Image.open(f).convert('RGB')) for t,f in paths.items()};width=args.width;small={t:cv2.resize(a,(width,round(a.shape[0]*width/a.shape[1])),interpolation=cv2.INTER_AREA if a.shape[1]>width else cv2.INTER_CUBIC) for t,a in images.items()};rows=json.loads((OUT/'full_canvas_feature_search.json').read_text()) if args.append and (OUT/'full_canvas_feature_search.json').exists() else []
for mode in ['clahe','blur']:
 desc={}
 for t,a in small.items():
  g=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
  if mode=='blur':g=cv2.GaussianBlur(g,(0,0),1.6*width/1600)
  g=cv2.createCLAHE(2,(12,12)).apply(g);k,d=cv2.SIFT_create(nfeatures=22000,contrastThreshold=.006,edgeThreshold=18).detectAndCompute(g,None);d=np.sqrt(d/(d.sum(1,keepdims=True)+1e-8));desc[t]=(k,d)
 for i,a in enumerate(ids):
  for b in ids[i+1:]:
   ka,da=desc[a];kb,db=desc[b];bf=cv2.BFMatcher();ab=bf.knnMatch(da,db,k=2);ba=bf.knnMatch(db,da,k=2);rev={m.queryIdx:m.trainIdx for m,n in ba if m.distance<.85*n.distance};ms=[m for m,n in ab if m.distance<.85*n.distance and rev.get(m.trainIdx)==m.queryIdx]
   pa=np.array([ka[m.queryIdx].pt for m in ms],np.float32)*1600/width;pb=np.array([kb[m.trainIdx].pt for m in ms],np.float32)*1600/width;left=np.arange(len(ms));hypotheses=[]
   for j in range(3):
    if len(left)<6:break
    H,k=cv2.estimateAffinePartial2D(pa[left],pb[left],method=cv2.RANSAC,ransacReprojThreshold=10,maxIters=30000,confidence=.999,refineIters=20)
    if H is None:break
    inliers=left[k.ravel()>0];H=np.vstack([H,[0,0,1]]);hypotheses.append(dict(matrix_analysis_1600=H.tolist(),inliers=len(inliers),normalized_scale=float(np.hypot(H[0,0],H[1,0])),angle_deg=float(np.degrees(np.arctan2(H[1,0],H[0,0])))));left=np.setdiff1d(left,inliers)
   row=dict(pair=f'T{a}_to_T{b}',mode=mode,analysis_width=width,mutual_matches=len(ms),source_points_analysis_1600=pa.tolist(),target_points_analysis_1600=pb.tolist(),hypotheses=hypotheses);rows.append(row);print('SEARCH',args.position,mode,row['pair'],len(ms),[(h['inliers'],round(h['matrix_analysis_1600'][0][2]),round(h['matrix_analysis_1600'][1][2])) for h in hypotheses],flush=True)
(OUT/'full_canvas_feature_search.json').write_text(json.dumps(rows,indent=2));print('SEARCH_DONE',args.position,flush=True)

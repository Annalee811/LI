import os
from pathlib import Path
import json,cv2,numpy as np
r=Path(os.environ['LI_Q1_RECHECK_WORK']);m=json.loads((r/'search_priors.json').read_text());z=np.load(r/'T2_to_T3/matches.npz');pa=z['pa'];pb=z['pb'];c=z['confidence'];cy=z['cycle_analysis1600'];v=[]
for t,p in [(2,pa),(3,pb)]:
 a=np.load(r/f'T{t}_vessels.npy').astype('float32');box=m['native_boxes'][str(t)];q=p-np.array(box[:2]);q*=a.shape[1]/(box[2]-box[0]);v.append(cv2.remap(a,q[:,0].astype('float32').reshape(216,-1),q[:,1].astype('float32').reshape(216,-1),cv2.INTER_LINEAR).ravel())
rows=[];cv2.setRNGSeed(20261019)
check=np.load(Path(__file__).parent/'reference_evidence/T2_to_T3_descriptor_matches.npz');vp=check['source_xy_analysis1600']*1328/1600;vq=check['target_xy_analysis1600']*1328/1600
# Independent RootSIFT consensuses fixed before this RoMa refit.
prior=json.loads((Path(__file__).parent/'reference_evidence/descriptor_search.json').read_text())['combined_hypotheses']['T2_to_T3'][0];P=np.array(prior['matrix_analysis1600']);P[:2,2]*=1328/1600;eval_ok=np.linalg.norm(vp@P[:2,:2].T+P[:2,2]-vq,axis=1)<5*1328/1600
for rt in [.1,.2,.3,.4]:
 ok=(c>=.3)&(cy<=10)&(v[0]>=rt)&(v[1]>=rt)
 H,kk=cv2.estimateAffinePartial2D(pa[ok].astype('float32'),pb[ok].astype('float32'),method=cv2.RANSAC,ransacReprojThreshold=4,maxIters=20000,confidence=.999,refineIters=20)
 if H is None:continue
 ix=np.flatnonzero(ok)[kk.ravel()>0];e=np.linalg.norm(pa[ix]@H[:,:2].T+H[:,2]-pb[ix],axis=1);ve=np.linalg.norm(vp[eval_ok]@H[:,:2].T+H[:,2]-vq[eval_ok],axis=1)
 row=dict(root_ridge_threshold=rt,anchors=len(ix),span_native=np.ptp(pa[ix],axis=0).tolist(),median_fit_error_native_px=float(np.median(e)),external_descriptor_median_px=float(np.median(ve)),external_descriptor_n=int(eval_ok.sum()),matrix_native=np.vstack([H,[0,0,1]]).tolist());rows.append(row);print(row)
np.savez_compressed(r/'T2_to_T3/independent_descriptor_check.npz',pa=vp[eval_ok],pb=vq[eval_ok]);(r/'root_pose_candidates.json').write_text(json.dumps(rows,indent=2))

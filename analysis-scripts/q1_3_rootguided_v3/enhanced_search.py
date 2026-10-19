import os,time,json
os.environ.setdefault('TORCH_HOME','/workspace/.cache/torch')
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
from romatch import roma_outdoor
r=Path(os.environ['LI_Q1_RECHECK_WORK']);m=json.loads((r/'search_priors.json').read_text());boxes={int(k):v for k,v in m['native_boxes'].items()};wh={int(k):v for k,v in m['original_wh'].items()};torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(20261019);dev='cuda' if torch.cuda.is_available() else 'cpu';model=roma_outdoor(device=dev,coarse_res=560,upsample_res=864,symmetric=True,use_custom_corr=False)
for a,b in [(1,2),(1,3)]:
 out=r/f'enhanced_T{a}_to_T{b}';out.mkdir(exist_ok=True)
 if (out/'matches.npz').exists():continue
 A=Image.open(r/f'T{a}_match_CLAHE.png');B=Image.open(r/f'T{b}_match_CLAHE.png');start=time.time();print('ENHANCED_MATCH',a,b,flush=True)
 with torch.inference_mode():w,c=model.match(A,B,device=dev)
 w=w.cpu().float().numpy();c=c.cpu().float().numpy()
 if w.ndim==4:w,c=w[0],c[0]
 mid=w.shape[1]//2;np.savez_compressed(out/'dense.npz',forward=w[:,:mid,2:].astype('float16'),backward=w[:,mid:,:2].astype('float16'),ca=c[:,:mid].astype('float16'),cb=c[:,mid:].astype('float16'),source_wh=A.size,target_wh=B.size)
 yy,xx=np.mgrid[:w.shape[0]:4,:mid:4];pa=np.stack([(xx+.5)*A.width/mid-.5,(yy+.5)*A.height/w.shape[0]-.5],-1);pb=(w[yy,xx,2:]+1)*[B.width/2,B.height/2]-.5;qb=(pb+.5)*[mid/B.width,w.shape[0]/B.height]-.5;rem=lambda ar:cv2.remap(ar.astype('float32'),qb[...,0].astype('float32'),qb[...,1].astype('float32'),cv2.INTER_LINEAR);cc=np.minimum(c[yy,xx],rem(c[:,mid:]));back=(rem(w[:,mid:,:2])+1)*[A.width/2,A.height/2]-.5;cy=np.linalg.norm(back-pa,axis=-1)*1600/wh[a][0]
 np.savez_compressed(out/'matches.npz',pa=(pa+boxes[a][:2]).reshape(-1,2),pb=(pb+boxes[b][:2]).reshape(-1,2),confidence=cc.ravel(),cycle_analysis1600=cy.ravel());print('DONE',a,b,round(time.time()-start,1),'confmax',cc.max(),'good',int(((cc>.3)&(cy<10)).sum()),flush=True)

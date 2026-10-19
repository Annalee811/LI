import os,time
from pathlib import Path
os.environ.setdefault('TORCH_HOME','/workspace/.cache/torch')
import cv2,numpy as np,torch
from PIL import Image
from romatch import roma_outdoor
r=Path(os.environ.get('LI_Q1_RECHECK_WORK','/tmp/LI_Q1_3_rootguided_v3'));out=r/'posed_T2_to_T3';out.mkdir(exist_ok=True);torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(20261019);dev='cuda' if torch.cuda.is_available() else 'cpu'
if not (out/'dense.npz').exists():
 model=roma_outdoor(device=dev,coarse_res=560,upsample_res=864,symmetric=True,use_custom_corr=False);A=Image.open(r/'deliverable/RGB/T2_confirmed_common_RGB.png');B=Image.open(r/'deliverable/RGB/T3_aligned_to_T2_common_RGB.png');print('POSED_MATCH',A.size,B.size,flush=True)
 with torch.inference_mode():w,c=model.match(A,B,device=dev)
 w=w.cpu().float().numpy();c=c.cpu().float().numpy()
 if w.ndim==4:w,c=w[0],c[0]
 mid=w.shape[1]//2;np.savez_compressed(out/'dense.npz',forward=w[:,:mid,2:].astype('float16'),backward=w[:,mid:,:2].astype('float16'),ca=c[:,:mid].astype('float16'),cb=c[:,mid:].astype('float16'),source_wh=A.size,target_wh=B.size);print('POSED_DONE',flush=True)

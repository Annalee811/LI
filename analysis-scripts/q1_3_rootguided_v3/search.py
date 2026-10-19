import os,json,time,sys,itertools
from pathlib import Path
os.environ.setdefault('TORCH_HOME','/workspace/.cache/torch')
import cv2,numpy as np,torch
from PIL import Image,ImageDraw
from romatch import roma_outdoor
root=Path(os.environ['LI_Q1_RECHECK_WORK']); raw=Path(os.environ['LI_Q1_RECHECK_RAW']);cv2.setNumThreads(1);torch.set_num_threads(4);torch.manual_seed(20261019);cv2.setRNGSeed(20261019)
fractions={1:[.02,.39,.99,.86],2:[.02,.25,.99,.88],3:[.04,.57,.99,1.]}; images={};boxes={};wh={}
for t,f in fractions.items():
 im=Image.open(next(raw.glob(f'T{t}.*'))).convert('RGB');wh[t]=im.size;b=[round(v*im.size[i%2]) for i,v in enumerate(f)];boxes[t]=b;images[t]=im.crop(b);images[t].save(root/f'T{t}_search_RGB.png')
(root/'search_priors.json').write_text(json.dumps(dict(fraction_boxes=fractions,native_boxes=boxes,original_wh=wh,source='User supplied root/layer region guidance; broad search prior, not point correspondence truth.'),indent=2))
model=roma_outdoor(device='cuda' if torch.cuda.is_available() else 'cpu',coarse_res=560,upsample_res=864,symmetric=True,use_custom_corr=False); device='cuda' if torch.cuda.is_available() else 'cpu'
for a,b in [(1,2),(1,3),(2,3)]:
 dest=root/f'T{a}_to_T{b}';dest.mkdir(exist_ok=True)
 if (dest/'matches.npz').exists():continue
 start=time.time();print('MATCH',a,b,flush=True)
 with torch.inference_mode():w,c=model.match(images[a],images[b],device=device)
 w=w.cpu().float().numpy();c=c.cpu().float().numpy()
 if w.ndim==4:w,c=w[0],c[0]
 mid=w.shape[1]//2
 np.savez_compressed(dest/'dense.npz',forward=w[:,:mid,2:].astype('float16'),backward=w[:,mid:,:2].astype('float16'),ca=c[:,:mid].astype('float16'),cb=c[:,mid:].astype('float16'),source_wh=images[a].size,target_wh=images[b].size)
 yy,xx=np.mgrid[:w.shape[0]:4,:mid:4];pa=np.stack([(xx+.5)*images[a].width/mid-.5,(yy+.5)*images[a].height/w.shape[0]-.5],-1);pb=(w[yy,xx,2:]+1)*[images[b].width/2,images[b].height/2]-.5
 qb=(pb+.5)*[mid/images[b].width,w.shape[0]/images[b].height]-.5
 rem=lambda ar:cv2.remap(ar.astype('float32'),qb[...,0].astype('float32'),qb[...,1].astype('float32'),cv2.INTER_LINEAR)
 cc=np.minimum(c[yy,xx],rem(c[:,mid:]));back=(rem(w[:,mid:,:2])+1)*[images[a].width/2,images[a].height/2]-.5;cycle=np.linalg.norm(back-pa,axis=-1)*1600/wh[a][0]
 pa=pa+boxes[a][:2];pb=pb+boxes[b][:2]
 np.savez_compressed(dest/'matches.npz',pa=pa.reshape(-1,2),pb=pb.reshape(-1,2),confidence=cc.ravel(),cycle_analysis1600=cycle.ravel())
 print('DONE',a,b,round(time.time()-start,1),'good',int(((cc>.3)&(cycle<10)).sum()),flush=True)

"""High-detail RoMa: overlapping context tiles, predetermined query ownership.
No confidence maximization, no elastic RGB rendering. Coordinates tested explicitly.
"""
import cv2,numpy as np,torch
from PIL import Image
class DetailedRoMa:
 def __init__(self,base):self.base=base;self.tiles=[]
 def sample(self,*args,**kwargs):return self.base.sample(*args,**kwargs)
 def match(self,ia,ib,**kwargs):
  w,h=ia.size;tw=min(1000,w);starts=sorted(set([0,(w-tw)//2,w-tw]));centers=np.array(starts)+tw/2;boundaries=[0]+[round((a+b)/2) for a,b in zip(centers[:-1],centers[1:])]+[w]
  ys,xs=np.mgrid[:h,:w].astype('float32');canon=np.stack([(xs+.5)*2/w-1,(ys+.5)*2/h-1],-1);parts=[np.concatenate([canon,canon],-1).copy() for _ in range(2)];cs=[np.zeros((h,w),np.float32) for _ in range(2)];self.tiles=[]
  for i,start in enumerate(starts):
   roi=(start,0,start+tw,h);print('DETAIL_TILE',roi,'query_owner',boundaries[i:i+2],flush=True)
   warp,conf=self.base.match(ia.crop(roi),ib.crop(roi),**kwargs)
   if warp.ndim==4:warp,conf=warp[0],conf[0]
   raw=warp.detach().cpu().float().numpy();cc=conf.detach().cpu().float().numpy();mid=raw.shape[1]//2;self.tiles.append(dict(roi=roi,owner_x=boundaries[i:i+2]))
   q0,q1=boundaries[i:i+2]
   for j in range(2):
    pred=cv2.resize(raw[:,j*mid:(j+1)*mid],(tw,h))
    for channel in [0,2]:pred[...,channel]=(pred[...,channel]+1)*tw/w+2*start/w-1
    cp=cv2.resize(cc[:,j*mid:(j+1)*mid],(tw,h));parts[j][:,q0:q1]=pred[:,q0-start:q1-start];cs[j][:,q0:q1]=cp[:,q0-start:q1-start]
   del warp,conf
  return torch.from_numpy(np.concatenate(parts,1)),torch.from_numpy(np.concatenate(cs,1))

def coordinate_check():
 class Identity:
  def match(self,a,b,**kw):
   w,h=a.size;y,x=np.mgrid[:h,:w].astype('float32');p=np.stack([(x+.5)*2/w-1,(y+.5)*2/h-1],-1);raw=np.concatenate([p,p],-1);return torch.from_numpy(np.concatenate([raw,raw],1)),torch.ones(h,2*w)
 m=DetailedRoMa(Identity());warp,c=m.match(Image.new('RGB',(2307,702)),Image.new('RGB',(2307,702)));p=warp.numpy()[:,:2307,:2];y,x=np.mgrid[:702,:2307].astype('float32');expected=np.stack([x,y],-1);actual=(p+1)*[2307/2,702/2]-.5;assert np.max(np.abs(actual-expected))<.001;print('PASS tiled identity coordinates',np.max(np.abs(actual-expected)),flush=True)

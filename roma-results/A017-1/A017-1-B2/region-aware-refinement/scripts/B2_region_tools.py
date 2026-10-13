"""Geometry and contextual RoMa tools for seasonal REGION correspondence.
Geometric observational overlap is distinct from confident point correspondence.
"""
import cv2,numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
def grid(h,w):
 y,x=np.mgrid[:h,:w].astype('float32');return np.stack([x,y],-1)
def project(p,H):
 H=np.asarray(H);return (p@H[:2,:2].T+H[:2,2]).astype('float32')
def inside(p,wh):
 return np.isfinite(p).all(-1)&(p[...,0]>=0)&(p[...,0]<=wh[0]-1)&(p[...,1]>=0)&(p[...,1]<=wh[1]-1)
def remap(a,p,nearest=False):return cv2.remap(a,p[...,0].astype('float32'),p[...,1].astype('float32'),cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
def determinant(p):
 dx_y,dx_x=np.gradient(p[...,0]);dy_y,dy_x=np.gradient(p[...,1]);return dx_x*dy_y-dx_y*dy_x

def rectangle(mask):
 h,w=mask.shape;heights=np.zeros(w,np.int32);best=(0,0,0,0,0)
 for y,row in enumerate(mask):
  heights=np.where(row,heights+1,0);stack=[]
  for x in range(w+1):
   cur=int(heights[x]) if x<w else 0;start=x
   while stack and stack[-1][1]>cur:
    sx,hh=stack.pop();width=x-sx;start=sx
    if width>=hh and width*hh>best[0]:best=(width*hh,sx,y-hh+1,x,y+1)
   if not stack or stack[-1][1]<cur:stack.append((start,cur))
 if not best[0]:raise ValueError('No common horizontal rectangle')
 return list(best[1:])

def fit(pa,pb,weight):
 from scipy.optimize import least_squares
 n=len(pa);A=np.zeros((2*n,4));A[::2,0]=pa[:,0];A[::2,1]=-pa[:,1];A[::2,2]=1;A[1::2,0]=pa[:,1];A[1::2,1]=pa[:,0];A[1::2,3]=1;Y=pb.ravel();sw=np.repeat(np.sqrt(weight),2)
 x=np.linalg.lstsq(A*sw[:,None],Y*sw,rcond=None)[0];x=least_squares(lambda z:(A@z-Y)*sw,x,loss='soft_l1',f_scale=2.5,max_nfev=500).x
 a,b,tx,ty=x;return np.array([[a,-b,tx],[b,a,ty],[0,0,1]])

def montage(images,labels,path,subtitle=''):
 h,w=images[0].shape[:2];head=110;out=Image.new('RGB',(w*len(images),h+head),'white');d=ImageDraw.Draw(out);font=ImageFont.truetype(FONT,28);small=ImageFont.truetype(FONT,21)
 for i,(im,label) in enumerate(zip(images,labels)):
  out.paste(Image.fromarray(im),(i*w,head));d.text((i*w+14,10),label,font=font,fill='#24343b');d.text((i*w+14,58),subtitle,font=small,fill='#52616a')
 out.save(path,dpi=(600,600));return out

class ContextRoMa:
 """2-D overlapping context tiles; fixed query owners, no confidence-max selection."""
 def __init__(self,base):self.base=base;self.tiles=[]
 def match(self,a,b,device):
  w,h=a.size;tw,th=min(1000,w),min(900,h)
  startsx=sorted(set([0,(w-tw)//2,w-tw]));startsy=sorted(set([0,h-th]))
  def bounds(starts,size,total):return [0]+[round((x+y+size)/2) for x,y in zip(starts[:-1],starts[1:])]+[total]
  bx=bounds(startsx,tw,w);by=bounds(startsy,th,h);fwd=np.zeros((h,w,2),'float32');back=fwd.copy();cf=np.zeros((h,w),'float32');cb=cf.copy();self.tiles=[]
  for iy,y in enumerate(startsy):
   for ix,x in enumerate(startsx):
    roi=[x,y,x+tw,y+th];owner=[bx[ix],by[iy],bx[ix+1],by[iy+1]];print('CONTEXT_TILE',roi,'query_owner',owner,flush=True)
    with torch.inference_mode():warp,c=self.base.match(a.crop(roi),b.crop(roi),device=device)
    if warp.ndim==4:warp,c=warp[0],c[0]
    raw=warp.cpu().float().numpy();c=c.cpu().float().numpy();mid=raw.shape[1]//2;sl=np.s_[owner[1]:owner[3],owner[0]:owner[2]];local=np.s_[owner[1]-y:owner[3]-y,owner[0]-x:owner[2]-x]
    pf=(cv2.resize(raw[:,:mid,2:],(tw,th))+1)*[tw/2,th/2]-.5+[x,y]
    pb=(cv2.resize(raw[:,mid:,:2],(tw,th))+1)*[tw/2,th/2]-.5+[x,y]
    fwd[sl]=pf[local];back[sl]=pb[local];cf[sl]=cv2.resize(c[:,:mid],(tw,th))[local];cb[sl]=cv2.resize(c[:,mid:],(tw,th))[local]
    self.tiles.append(dict(context_xyxy=roi,query_owner_xyxy=owner));del warp,c,raw
  return fwd,back,cf,cb

def self_check():
 class Identity:
  def match(self,a,b,device):
   w,h=a.size;p=(grid(h,w)+.5)/[w/2,h/2]-1;v=np.concatenate([p,p],-1);return torch.from_numpy(np.concatenate([v,v],1).astype('float32')),torch.ones(h,w*2)
 f,b,cf,cb=ContextRoMa(Identity()).match(Image.new('RGB',(2307,1320)),Image.new('RGB',(2307,1320)),torch.device('cpu'))
 np.testing.assert_allclose(f,grid(1320,2307),atol=.001);np.testing.assert_allclose(b,grid(1320,2307),atol=.001);assert np.all(cf==1) and np.all(cb==1)
 # Translation changes point correspondence, while geometric observation is still valid.
 moved=grid(10,12)+[4,0];geo=np.ones((10,12),bool);support=inside(moved,(12,10));assert geo.sum()>support.sum();assert (np.linalg.norm(moved-grid(10,12),axis=-1)[support]>3).all()
 print('PASS context-tile coordinates and observation/correspondence distinction',flush=True)

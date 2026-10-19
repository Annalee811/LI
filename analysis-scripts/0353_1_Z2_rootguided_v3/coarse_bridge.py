"""User-constrained coarse translation from root-like ridge patterns.

This is an explicitly unverified localization fallback, never RoMa success.
"""
import argparse,json,sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image
from scipy.signal import fftconvolve
from skimage.filters import frangi

def main():
 p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args();r=a.work;pri=json.loads((r/'user_region_priors.json').read_text());W=800;arrays={};masks={};cv2.setNumThreads(1)
 for t in [2,3]:
  im=Image.open(next((r/'inputs/raw/0353-1-Z2').glob(f'T{t}.*'))).convert('RGB');s=W/im.width;rgb=np.array(im.resize((W,round(im.height*s)),Image.Resampling.LANCZOS));g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)/255.;v=frangi(g,sigmas=[.7,1.2,2],black_ridges=False);v=np.clip(v/max(np.quantile(v,.995),1e-8),0,1).astype('float32');mask=np.zeros(v.shape,np.float32);b=np.rint(np.array(pri[f'T{t}']['native_box'])*s).astype(int);mask[b[1]:b[3],b[0]:b[2]]=1
  # Keep coherent elongated components, not individual soil grains.
  binary=np.uint8(v>.16);n,labels,stats,_=cv2.connectedComponentsWithStats(binary,8);keep=np.zeros(n,bool)
  for k in range(1,n):
   x,y,w,h,area=stats[k]
   if area>=12 and max(w,h)>=18:keep[k]=True
  v*=keep[labels];arrays[t]=v*mask;masks[t]=mask;Image.fromarray(np.uint8(v*255)).save(r/f'T{t}_coarse_ridge_inspection.png')
 A=arrays[2];B=arrays[3];ma=masks[2];mb=masks[3];corr=lambda b,a:fftconvolve(b,a[::-1,::-1],mode='full');n=corr(mb,ma);sa=corr(mb,A);sb=corr(B,ma);aa=corr(mb,A*A);bb=corr(B*B,ma);ab=corr(B,A);den=np.sqrt(np.maximum(aa-sa*sa/np.maximum(n,1),0)*np.maximum(bb-sb*sb/np.maximum(n,1),0));score=(ab-sa*sb/np.maximum(n,1))/np.maximum(den,1e-9)
 yy,xx=np.indices(score.shape);dx=xx-(A.shape[1]-1);dy=yy-(A.shape[0]-1);seed=(np.mean(pri['T3']['native_box'][1::2])-np.mean(pri['T2']['native_box'][1::2]))*W/1328
 valid=(np.abs(dx)<=.15*W)&(np.abs(dy-seed)<=.22*A.shape[0])&(n>=.5*min(ma.sum(),mb.sum()));score[~valid]=-np.inf;working=score.copy();peaks=[]
 for i in range(5):
  y,x=np.unravel_index(np.argmax(working),working.shape);peaks.append(dict(dx_native=float(dx[y,x]*1328/W),dy_native=float(dy[y,x]*1328/W),ridge_NCC=float(score[y,x]),overlapping_search_pixels_analysis800=int(round(n[y,x]))));working[(dx-dx[y,x])**2+(dy-dy[y,x])**2<40**2]=-np.inf
 best=peaks[0];H=[[1.,0.,best['dx_native']],[0.,1.,best['dy_native']],[0.,0.,1.]]
 result=dict(pair='T2_to_T3',status='user_guided_coarse_translation_UNVERIFIED',native_matrix=H,method='Masked root-like ridge normalized correlation inside user supplied observation bands; translation only; no scale/shear/elastic deformation',source='User-region prior plus ridge-pattern coarse localization; not a successful RoMa point match',peaks=peaks,seed_dy_native=float(seed*1328/W),manual_TRE=None,automatic_point_TRE=None,mandatory_review=True,limitation='Ridge responses include some non-root elongated structures. Pattern correlation does not establish root identity or pixel-accurate registration.')
 (r/'coarse_bridge.json').write_text(json.dumps(result,indent=2));np.savez_compressed(r/'coarse_bridge_score.npz',score=score.astype('float32'),seed_dy_analysis800=seed);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()

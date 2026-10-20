"""Render the two user-identified details and individual putative root routes."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import common as c
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);a=p.parse_args();r=a.run;o=r/'paper_figures';o.mkdir(exist_ok=True);rois=[('A',(1495,930,1580,1035)),('B',(1595,915,1715,1015))];plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'savefig.facecolor':'white'})
def over(rgb,cl):
 z=rgb.copy();mask=(cl>0)&(cl<255);z[mask]=np.rint(.25*rgb[mask]+.75*c.PALETTE[cl[mask]]).astype('uint8');return z
fig,axes=plt.subplots(2,7,figsize=(16.54,5.1),squeeze=False);names=['T1 RGB','T2 RGB','V17 U-Net','Selective labels','Updated U-Net','Updated DeepLab','Updated SegFormer'];selected=[]
for i,(label,roi) in enumerate(rois):
 x,y,xx,yy=roi;sl=np.s_[y:yy,x:xx];rgb=np.array(Image.open(r/'data/upper/T2_RGB.png'))[sl];previous=np.array(Image.open(r/'data/upper/T1_RGB.png'))[sl];old=np.array(Image.open(a.previous/'unet/four_period_prediction/T2_temporal_classes.png'))[sl];gt=np.array(Image.open(r/'data/revised/T2_classes.png'))[sl];views=[previous,rgb,over(rgb,old),over(rgb,gt)]+[over(rgb,np.array(Image.open(r/m/'four_period_prediction/T2_temporal_classes.png'))[sl]) for m in c.MODELS]
 for j,im in enumerate(views):
  axes[i,j].imshow(im,interpolation='none');axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
  for sp in axes[i,j].spines.values():sp.set_visible(False)
  if not i:axes[i,j].set_title(names[j],fontsize=8)
  if j==0:axes[i,j].set_ylabel('Detail '+label,rotation=0,labelpad=23,fontweight='bold')
 selected.append(dict(label=label,ROI=list(roi),source='Fixed T2 regions corresponding to the user-reported P33 junction details, not chosen by a new score'))
fig.legend(handles=[Patch(color=c.PALETTE[k]/255,label=n) for k,n in [(1,'Persistent'),(2,'Newly observed'),(3,'Tip extension')]],loc='lower center',ncol=3,frameon=False);fig.subplots_adjust(left=.045,right=.995,top=.92,bottom=.09,wspace=.025,hspace=.03);fig.savefig(o/'FigS7_user_detail_before_after.tif',dpi=600,pil_kwargs={'compression':'tiff_adobe_deflate'});fig.savefig(o/'FigS7_user_detail_before_after.png',dpi=130);plt.close(fig)
# Show only one root's annotation at a time, rather than confusing crossing colors.
axis=c.load_axes(r/'axis_cache/unet/T2.npz');cl=np.array(Image.open(r/'unet/four_period_prediction/T2_temporal_classes.png'));rgb=np.array(Image.open(r/'data/upper/T2_RGB.png'));fig,axes=plt.subplots(2,4,figsize=(10.8,5.6),squeeze=False);instances=[]
for i,(label,roi) in enumerate(rois):
 x,y,xx,yy=roi;sl=np.s_[y:yy,x:xx];counts=np.bincount(axis['axis_map'][sl].ravel());counts[0]=0;ids=np.argsort(counts)[::-1][:4]
 for j,root in enumerate(ids):
  mask=axis['axis_map']==root;localmask=mask[sl];img=rgb[sl].copy();img[localmask]=np.rint(.18*img[localmask]+.82*c.PALETTE[cl[sl][localmask]]).astype('uint8');axes[i,j].imshow(img,interpolation='none');path=axis['paths'][root-1];pts=axis['xy'][path];inside=(pts[:,0]>=y)&(pts[:,0]<yy)&(pts[:,1]>=x)&(pts[:,1]<xx);points=pts[inside]
  if len(points)>10:
   mid=len(points)//2;a0=points[max(0,mid-5)];a1=points[min(len(points)-1,mid+5)];axes[i,j].annotate('',xy=(a1[1]-x,a1[0]-y),xytext=(a0[1]-x,a0[0]-y),arrowprops=dict(arrowstyle='<->',color='white',lw=1))
  axes[i,j].set_title(f'{label}: candidate R{root}',fontsize=8);axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
  for sp in axes[i,j].spines.values():sp.set_visible(False)
  instances.append(dict(detail=label,root_axis_id=int(root),root_pixels_in_detail=int(counts[root]),individual_class_values=np.unique(cl[mask]).tolist()))
fig.text(.5,.015,'One candidate root highlighted per panel. Arrows show orientation, not verified growth polarity.\nCandidate IDs are not independently validated biological root identities.',ha='center',fontsize=7);fig.subplots_adjust(left=.025,right=.995,top=.94,bottom=.10,wspace=.04,hspace=.13);fig.savefig(o/'FigS8_individual_root_routes.tif',dpi=600,pil_kwargs={'compression':'tiff_adobe_deflate'});fig.savefig(o/'FigS8_individual_root_routes.png',dpi=130);plt.close(fig);c.js(o/'user_detail_protocol.json',dict(regions=selected,isolated_root_examples=instances,root_IDs_are_candidates=True,RGB_original=True))
print('USER_DETAILS_AND_INDIVIDUAL_ROUTES_READY',flush=True)

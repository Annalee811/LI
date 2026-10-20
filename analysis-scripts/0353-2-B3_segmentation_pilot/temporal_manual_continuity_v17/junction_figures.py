"""Before/after crossing details selected from V14 conflicts, not new scores."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch,Rectangle
import common as c
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);a=p.parse_args();r=a.run;out=r/'paper_figures';out.mkdir(exist_ok=True);plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
def overlay(rgb,cl):
 z=rgb.copy();mask=(cl>0)&(cl<255);z[mask]=np.rint(.25*rgb[mask]+.75*c.PALETTE[cl[mask]]).astype('uint8');return z
def save(fig,name):
 fig.savefig(out/(name+'.tif'),dpi=600,pil_kwargs={'compression':'tiff_adobe_deflate'});fig.savefig(out/(name+'.png'),dpi=130);plt.close(fig)
rows=c.rows(r/'data/paired_manifest.csv');rois=[]
for t in c.PERIODS[1:]:
 axis=c.load_axes(r/'axis_cache/unet'/f'{t}.npz');old=np.array(Image.open(a.previous/'unet/four_period_prediction'/f'{t}_temporal_classes.png'));gt=np.array(Image.open(r/'data/revised'/f'{t}_classes.png'));options=[]
 for row in rows:
  if row['split']!='test' or row['current']!=t:continue
  x,y=int(row['x']),int(row['y']);sl=np.s_[y:y+256,x:x+256];classes=old[sl];known=(gt[sl]>0)&(gt[sl]<255);score=int(min((classes==2).sum(),(classes==3).sum()));options.append(dict(period=t,tile_id=row['tile_id'],ROI=[x,y,x+256,y+256],V14_new_growth_content_score=score,typed_reference_pixels=int(known.sum())))
 selected=max(options,key=lambda z:(z['V14_new_growth_content_score'],z['typed_reference_pixels'],z['tile_id']));rois.append(selected)
fig,axes=plt.subplots(3,8,figsize=(16.54,7.8),squeeze=False);headers=['Original RGB','Manual / U-Net width','V14 U-Net','Updated U-Net','V14 DeepLabV3+','Updated DeepLabV3+','V14 SegFormer','Updated SegFormer']
for i,rec in enumerate(rois):
 t=rec['period'];x,y,xx,yy=rec['ROI'];sl=np.s_[y:yy,x:xx];rgb=np.array(Image.open(r/'data/upper'/f'{t}_RGB.png'))[sl];gt=np.array(Image.open(r/'data/revised'/f'{t}_classes.png'))[sl];views=[rgb,overlay(rgb,gt)]
 for m in c.MODELS:
  for folder in [a.previous,r]:views.append(overlay(rgb,np.array(Image.open(folder/m/'four_period_prediction'/f'{t}_temporal_classes.png'))[sl]))
 for j,im in enumerate(views):
  ax=axes[i,j];ax.imshow(im,interpolation='none');ax.set_xticks([]);ax.set_yticks([])
  for sp in ax.spines.values():sp.set_visible(False)
  if not i:ax.set_title(headers[j],fontsize=8)
  if j==0:ax.set_ylabel(t,rotation=0,labelpad=12,fontweight='bold')
fig.legend(handles=[Patch(color=c.PALETTE[k]/255,label=n) for k,n in [(1,'Persistent'),(2,'Newly observed'),(3,'Tip growth')]],loc='lower center',ncol=3,frameon=False);fig.subplots_adjust(left=.025,right=.995,top=.94,bottom=.075,wspace=.035,hspace=.05);save(fig,'FigS7_crossing_before_after')
# Root-graph ambiguity is shown rather than claimed to be biological identity.
fig,axes=plt.subplots(1,3,figsize=(9.5,3.6));metric=c.rows(r/'tables/root_continuity_before_after.csv')
for j,m in enumerate(c.MODELS):
 old=[int(q['axes_with_new_and_growth']) for q in metric if q['model']==m and q['version']=='V14'];new=[int(q['axes_with_new_and_growth']) for q in metric if q['model']==m and q['version']=='manual_continuity_v17'];x=np.arange(3);axes[j].bar(x-.18,old,.34,label='V14',color='#D55E00');axes[j].bar(x+.18,new,.34,label='Updated',color='#0072B2');axes[j].set_xticks(x,['T2','T3','T4']);axes[j].set_title(c.MODEL_NAMES[m]);axes[j].set_ylabel('Axes containing NEW and GROWTH');axes[j].spines[['top','right']].set_visible(False);axes[j].grid(axis='y',alpha=.15)
axes[0].legend(frameon=False);fig.tight_layout();save(fig,'FigS8_root_axis_consistency')
c.js(out/'crossing_figure_protocol.json',dict(ROIs=rois,selection='Within each date, same six internal test windows; maximum V14 NEW/GROWTH content, no updated predictions used to choose',field_RGB_unchanged=True,unknown_reference_pixels_not_given_temporal_class=True,consistency_zero_is_enforced_not_independent_accuracy=True))
with (out/'CAPTIONS.md').open('a') as f:f.write('\n- S7: Identical local RGB and manual-class reference, V14 and updated outputs for all three models. Each date uses the internal test window with greatest V14 NEW/GROWTH coexistence content; the updated model score does not select the window. No individual biological root IDs are manually available.\n- S8: Counts of putative axes containing both NEW and GROWTH before and after on the same learned graph; zero is a grammar constraint, not independent anatomical accuracy. Full upper fields include training pixels.\n')
print('CROSSING_DETAIL_FIGURES_READY',flush=True)

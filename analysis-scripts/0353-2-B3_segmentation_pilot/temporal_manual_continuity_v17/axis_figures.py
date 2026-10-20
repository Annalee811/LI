"""Inspect original RGB, full-image root-axis routes and temporal decisions."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Patch
import common as c
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run;o=r/'paper_figures';o.mkdir(exist_ok=True);roi=(1200,220,1800,820);sl=np.s_[220:820,1200:1800];palette=np.array([[0,114,178],[213,94,0],[0,158,115],[136,86,167],[204,121,167],[180,151,0]],np.uint8)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
def save(fig,name,vector=False):
 fig.savefig(o/(name+'.tif'),dpi=600,pil_kwargs={'compression':'tiff_lzw'});fig.savefig(o/(name+'.png'),dpi=130)
 if vector:fig.savefig(o/(name+'.pdf'));fig.savefig(o/(name+'.svg'))
 plt.close(fig)
fig,aa=plt.subplots(4,3,figsize=(8.27,11.1));selected=[]
for i,t in enumerate(c.PERIODS):
 rgb=np.array(Image.open(r/'data/upper'/f'{t}_RGB.png'));axis=c.load_axes(r/'axis_cache/unet'/(t+'.npz'));route=rgb.copy();ids=axis['axis_map'];route[ids>0]=np.rint(.5*route[ids>0]+.5*np.array([210,210,210])).astype('uint8');choices=[]
 for ai,path in enumerate(axis['paths'],1):
  xy=axis['xy'][path];inside=(xy[:,0]>=220)&(xy[:,0]<820)&(xy[:,1]>=1200)&(xy[:,1]<1800)
  if inside.sum()>=55:choices.append((int(inside.sum()),ai,path,xy[inside]))
 choices.sort(key=lambda z:(-z[0],z[1]));choices=choices[:6]
 for j,(_,ai,path,points) in enumerate(choices):
  mask=ids==ai;route[mask]=np.rint(.15*rgb[mask]+.85*palette[j]).astype('uint8');mid=len(points)//2;p0=points[max(0,mid-8)]-[220,1200];p1=points[min(len(points)-1,mid+8)]-[220,1200];aa[i,1].annotate('',xy=p1[::-1],xytext=p0[::-1],arrowprops=dict(arrowstyle='<->',color='white',lw=.7));label=points[mid]-[220,1200];aa[i,1].text(label[1]+7,label[0]-5,f'A{ai}',fontsize=6,color='white',bbox=dict(facecolor='black',alpha=.65,pad=.8,edgecolor='none'));selected.append(dict(period=t,axis_id=ai,selection='Six longest visible traced axes in a fixed ROI, not selected by score'))
 unresolved=[d for d in axis['decisions'] if d.get('unresolved') and 'point' in d and 220<=d['point'][0]<820 and 1200<=d['point'][1]<1800]
 for d in unresolved[:8]:aa[i,1].add_patch(Circle((d['point'][1]-1200,d['point'][0]-220),7,fill=False,color='#E69F00',lw=.6))
 aa[i,0].imshow(rgb[sl],interpolation='none');aa[i,1].imshow(route[sl],interpolation='none');im=np.array(Image.open(r/'unet/four_period_prediction'/f'{t}_temporal_overlay.png'));aa[i,2].imshow(im[sl],interpolation='none')
 for j in range(3):
  aa[i,j].set_xticks([]);aa[i,j].set_yticks([])
  if i==0:aa[i,j].set_title(['Original RGB','Traced axes / local IDs','Whole-axis temporal output'][j],fontsize=9)
  if j==0:aa[i,j].set_ylabel(t,rotation=0,labelpad=13,fontweight='bold')
 fig.subplots_adjust(left=.055,right=.99,top=.96,bottom=.065,wspace=.03,hspace=.025)
fig.text(.5,.026,'Axis IDs are local candidates. Double arrows: orientation, not growth polarity.\nOrange circles: unresolved crossings. Colors in the middle column identify routes only.',ha='center',fontsize=7);save(fig,'Fig6_RGB_root_axis_continuity')
c.js(o/'axis_figure_protocol.json',dict(local_ROI=list(roi),root_model='unet',local_ids_not_confirmed_root_identity=True,original_RGB_unchanged=True,selected=selected))
# S4: all18test pairs retained, paired dates remain correlated spatial repeats.
rows=c.rows(r/'tables/class_metrics.csv')
per=[]
for m in c.MODELS:
 rr=c.rows(r/m/'evaluation/test/per_pair_class_metrics.csv')
 for pair in sorted(set(z['pair_id'] for z in rr)):
  z=[x for x in rr if x['pair_id']==pair and x['reference']=='revised' and x['Dice']!=''];per.append(dict(model=m,pair_id=pair,period=z[0]['current'],macro_Dice=float(np.mean([float(x['Dice']) for x in z])),classes_defined=len(z)))
c.csvout(r/'tables/test_pair_heterogeneity.csv',per);fig,ax=plt.subplots(figsize=(6.2,4));rng=np.random.default_rng(10);cols={'T2':'#0072B2','T3':'#009E73','T4':'#CC79A7'}
for j,m in enumerate(c.MODELS):
 z=[x for x in per if x['model']==m];values=[x['macro_Dice'] for x in z];box=ax.boxplot([values],positions=[j],widths=.46,showfliers=False,patch_artist=True);box['boxes'][0].set(facecolor='#E6E9ED',edgecolor='#777777')
 for t in ['T2','T3','T4']:
  yy=[x['macro_Dice'] for x in z if x['period']==t];ax.scatter(j+rng.uniform(-.15,.15,len(yy)),yy,s=20,color=cols[t],edgecolor='white',linewidth=.3,label=t if j==0 else None,zorder=3)
ax.set_xticks(range(3),['U-Net','DeepLabV3+','SegFormer-B0']);ax.set_ylim(0,1);ax.set_ylabel('Per-pair temporal macro Dice');ax.grid(axis='y',alpha=.2);ax.set_title('All internal test pairs | manual classes / U-Net morphology');ax.legend(frameon=False,ncol=3,loc='upper right');fig.tight_layout();save(fig,'FigS4_test_pair_heterogeneity',True)
with (o/'CAPTIONS.md').open('a') as f:f.write('\n- Fig6: Original RGB, candidate root axes and final U-Net temporal output in the fixed local window. The longest six visible axes per period are colored; other routes are gray. Local axis numbers do not establish biological identity across dates. Double arrows show orientation, not growth polarity. Orange circles mark up to eight unresolved local crossings.\n- S4 test-pair heterogeneity: All18same-field test pairs retained, grouped by model and colored by transition. MacroDice excludes undefined classes with no TP/FP/FN; dates at the same six locations are correlated, not independent plants.\n- The confusion matrix is S6. The active decoder allows an all-growth continuation trace only when its endpoint extends outward from a real previous predicted root tip with matching direction and learned growth evidence; a lateral born on a prior shaft cannot use this exception.\n')
print('AXIS_AND_HETEROGENEITY_FIGURES_READY',flush=True)

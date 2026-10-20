"""Visual evidence sheets for the largest edit of each type/date, without model-score selection."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import common as c
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run;o=r/'annotation_review';o.mkdir(exist_ok=True);edits=json.loads((r/'data/revision_log.json').read_text());chosen=[]
for t in c.PERIODS[1:]:
 for kind in ['Persistent','Entire newly']:
  group=[x for x in edits if x['period']==t and x['reason'].startswith(kind)]
  if group:chosen.append(max(group,key=lambda q:q['changed_pixels']))
fig,axes=plt.subplots(len(chosen),5,figsize=(12,2.3*len(chosen)),squeeze=False);plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7})
def overlay(rgb,cl):
 z=rgb.copy();mask=(cl>0)&(cl<255);z[mask]=np.rint(.25*rgb[mask]+.75*c.PALETTE[cl[mask]]).astype('uint8');return z
for i,rec in enumerate(chosen):
 t=rec['period'];prev=c.PERIODS[c.PERIODS.index(t)-1];x0,y0,x1,y1=rec['bbox_xyxy'];cx,cy=(x0+x1)//2,(y0+y1)//2;x=int(np.clip(cx-128,0,2541-256));y=int(np.clip(cy-128,0,1108-256));sl=np.s_[y:y+256,x:x+256];rgb=np.array(Image.open(r/'data/upper'/f'{t}_RGB.png'))[sl];previous=np.array(Image.open(r/'data/upper'/f'{prev}_RGB.png'))[sl];old=np.array(Image.open(r/'data/frozen'/f'{t}_classes.png'))[sl];new=np.array(Image.open(r/'data/revised'/f'{t}_classes.png'))[sl];mask=(old!=new);error=np.full_like(rgb,255);error[mask]=[213,94,0];views=[previous,rgb,overlay(rgb,old),overlay(rgb,new),error]
 for j,v in enumerate(views):
  axes[i,j].imshow(v,interpolation='none');axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
  for sp in axes[i,j].spines.values():sp.set_visible(False)
  if i==0:axes[i,j].set_title(['Previous RGB','Current RGB','Frozen V17 labels','Selective proposal','Changed pixels'][j],fontsize=8)
  if j==0:axes[i,j].set_ylabel(f'{t} R{rec["root_axis_id"]}\n{rec["changed_pixels"]} px',fontsize=7)
 rec['review_ROI']=[x,y,x+256,y+256]
fig.subplots_adjust(left=.065,right=.995,top=.96,bottom=.03,wspace=.02,hspace=.04);fig.savefig(o/'selective_revision_evidence.tif',dpi=600,pil_kwargs={'compression':'tiff_adobe_deflate'});fig.savefig(o/'selective_revision_evidence.png',dpi=130);plt.close(fig);c.js(o/'selected_revision_examples.json',chosen);print('REVISION_EVIDENCE_READY',len(chosen),flush=True)

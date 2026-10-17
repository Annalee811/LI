#!/usr/bin/env python3
"""Describe stroke geometry and strict-Dice sensitivity; not model performance."""
import argparse
from pathlib import Path
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import common as c

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();data=c.Data(a.run);rows=[]
    for t in c.PERIODS:
        gt=data.gt[t];root=gt==255;sk=skeletonize(root);width=2*distance_transform_edt(root)[sk]-1
        q=np.quantile(width,[.25,.5,.75,.95]);shifts={}
        for axis in ['x','y']:
            one,two=(gt[:,1:],gt[:,:-1]) if axis=='x' else (gt[1:,:],gt[:-1,:])
            shifts[axis]=c.metrics(c.counts(one,two==255))['Dice']
        rows.append(dict(period=t,root_pixels=int(root.sum()),estimated_skeleton_local_width_q25_q50_q75_q95=q.tolist(),
            manual_mask_vs_itself_shifted_one_pixel_Dice=shifts))
    c.js(a.run/'label_geometry_audit.json',dict(periods=rows,
        interpretation='Diagnostic of human stroke geometry and one-pixel metric sensitivity, not prediction scores or an anatomical accuracy ceiling',
        labels_changed=False,used_for_model_selection=False,used_to_relax_gate=False,
        width='2*Euclidean distance-to-background at skeleton minus1; approximate local stroke width, not physical root diameter'))
    folder=a.run/'figures';folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(8,3.2));x=np.arange(4)
    axes[0].bar(x,[r['estimated_skeleton_local_width_q25_q50_q75_q95'][1] for r in rows],color='#7b6e91')
    axes[0].set(xticks=x,xticklabels=c.PERIODS,ylabel='Approximate median stroke width (px)',ylim=(0,4))
    for off,axis,col in [(-.18,'x','#477d69'),(.18,'y','#be914d')]:
        axes[1].bar(x+off,[r['manual_mask_vs_itself_shifted_one_pixel_Dice'][axis] for r in rows],.36,label='1px '+axis+' shift',color=col)
    axes[1].set(xticks=x,xticklabels=c.PERIODS,ylabel='Manual self-shift strict Dice',ylim=(0,1));axes[1].legend(frameon=False,fontsize=8)
    fig.suptitle('Label geometry diagnostic — not model scores',fontsize=10);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(folder/('label_geometry_sensitivity.'+ext),dpi=300)
    plt.close(fig);print('LABEL_GEOMETRY_AUDITED',flush=True)
if __name__=='__main__':main()

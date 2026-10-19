"""Publication-quality area/point-support summary from actual result tables."""
import argparse,csv,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/LI_allseries_matplotlib')
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from render_results import aggregate

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args();rows,conf=aggregate(a.work);root=a.work/'deliverable';positions=sorted({r['position'] for r in rows});area=np.full((len(positions),4),np.nan);support=area.copy();review=[]
    for i,pos in enumerate(positions):
        rr=[r for r in rows if r['position']==pos];review.append(any(r['review_required_regions'] for r in rr))
        for r in rr:
            if r['common_area_percent'] is not None:area[i,int(r['period'][1:])-1]=r['common_area_percent']
        for r in conf:
            if r['position']==pos:support[i,int(r['target_period'][1:])-1]=r['effective_overlap_percent_of_common']
    plt.rcParams.update({'font.family':'Liberation Sans','font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axs=plt.subplots(1,2,figsize=(9,9.5));fig.subplots_adjust(left=.19,right=.97,bottom=.11,top=.93,wspace=.43)
    for j,(ax,data,title,cmap) in enumerate(zip(axs,[area,support],['(a) Common observation area / original (%)','(b) Effective support / common area (%)'],['YlGnBu','cividis'])):
        cmap=plt.get_cmap(cmap).copy();cmap.set_bad('#EEEEEE');im=ax.imshow(data,vmin=0,vmax=100,cmap=cmap,aspect='auto');ax.set_title(title,fontsize=10,pad=13)
        ax.set_xticks(range(4),['T1','T2','T3','T4']);ax.set_yticks(range(len(positions)),[pos+(' †' if pos=='0353-2-B3' else '')+(' *' if rev else '') for pos,rev in zip(positions,review)] if j==0 else ['']*len(positions));ax.tick_params(length=0)
        for y in range(len(positions)):
            for x in range(4):
                value=data[y,x];text='—' if not np.isfinite(value) else ('<0.1' if 0<value<.1 else f'{value:.1f}')
                color='#777777' if not np.isfinite(value) else ('white' if (value>55 if j==0 else value<40) else '#192D35');ax.text(x,y,text,ha='center',va='center',color=color,fontsize=8.8)
        ax.set_xticks(np.arange(-.5,4,.999999),minor=True);ax.set_yticks(np.arange(-.5,len(positions),1),minor=True);ax.grid(which='minor',color='white',lw=.8);ax.tick_params(which='minor',bottom=False,left=False)
        for spine in ax.spines.values():spine.set_visible(False)
        fig.colorbar(im,ax=ax,orientation='horizontal',pad=.045,shrink=.95,fraction=.025,ticks=[0,25,50,75,100])
    fig.text(.19,.025,'* At least one observation band needs review. Gray: missing date, reference-only support, or unresolved localization.\n† 0353-2-B3: upper observation band only; denominator remains the full original image area.\nAreas exclude separators; effective support requires RoMa confidence and geometric consistency after localization.',fontsize=8.1,color='#596B75',va='bottom')
    stem=root/'all_series_area_and_confidence_summary'
    for ext in ['png','pdf','svg']:fig.savefig(str(stem)+'.'+ext,dpi=600)
    with Image.open(str(stem)+'.png') as im:im.save(str(stem)+'.tif',compression='tiff_lzw',dpi=(600,600))
    plt.close(fig)
if __name__=='__main__':main()

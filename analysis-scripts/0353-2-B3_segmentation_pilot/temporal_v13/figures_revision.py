"""Show every accepted label correction on unchanged original RGB, no model map."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch,Rectangle
import common as c

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--original-data',type=Path,required=True);a=p.parse_args();run=a.run;out=run/'figures';log=json.loads((run/'data/correction_log.json').read_text());caps=json.loads((out/'captions.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axs=plt.subplots(len(log['records']),4,figsize=(9.5,10),layout='constrained')
    for i,r in enumerate(log['records']):
        t=r['current'];prev=r['previous'];x0,y0,x1,y1=r['bbox'];pad=35;x0=max(0,x0-pad);y0=max(0,y0-pad);x1=min(2541,x1+pad);y1=min(c.HEIGHT,y1+pad);sl=(slice(y0,y1),slice(x0,x1))
        rgb=np.array(Image.open(run/'data/upper'/f'{t}_RGB.png'));previous=np.array(Image.open(run/'data/upper'/f'{prev}_RGB.png'));root=np.array(Image.open(run/'data/upper'/f'{t}_mask.png'))==255;old=np.array(Image.open(a.original_data/'targets'/f'{t}_temporal_target.png'));new=np.array(Image.open(run/'data/targets'/f'{t}_temporal_target.png'))
        overlays=[]
        for target in [old,new]:
            image=rgb.copy();typed=root&(target>0)&(target<255);image[typed]=np.rint(.3*rgb[typed]+.7*c.PALETTE[target[typed]]).astype(np.uint8);overlays.append(image)
        for ax,im in zip(axs[i],[previous,rgb,*overlays]):
            ax.imshow(im[sl],interpolation='nearest');ax.set_xticks([]);ax.set_yticks([]);b=r['bbox'];ax.add_patch(Rectangle((b[0]-x0-2,b[1]-y0-2),b[2]-b[0]+4,b[3]-b[1]+4,fill=False,color='#222222',lw=.6))
        axs[i,0].set_ylabel(r['review_id']+' '+prev+' → '+t+'\n'+str(r['pixels'])+' pixels')
    for ax,title in zip(axs[0],['Previous original RGB','Current original RGB','Original user classes','Reviewed time classes']):ax.set_title(title)
    fig.legend(handles=[Patch(color=c.PALETTE[1]/255,label='Persistent'),Patch(color=c.PALETTE[2]/255,label='Newly observed')],loc='outside lower center',ncol=2,frameon=False);fig.suptitle('Four image-reviewed label corrections · no model predictions used')
    name='Fig13_all_annotation_revisions';fig.savefig(out/(name+'.png'),dpi=600,bbox_inches='tight');fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
    with Image.open(out/(name+'.png')) as im:im.save(out/(name+'.tif'),compression='tiff_lzw',dpi=(600,600))
    plt.close(fig);caps[name]='All4 accepted assistant visual revisions of user annotations: T2 203px, T3 91px, T4 34px, newly observed to persistent. The prior date already shows the corresponding root stem. Root masks and RGB are unchanged. Boxes identify the original labelled segment, not a model prediction. These are authorized annotation revisions, not independent expert adjudication. Five ambiguous candidate regions remain unchanged.';c.js(out/'captions.json',caps);(out/'Figure_captions_EN.md').write_text('\n\n'.join(f'{k}: {v}' for k,v in caps.items())+'\n');print('ANNOTATION_REVISION_FIGURE_READY',flush=True)
if __name__=='__main__':main()

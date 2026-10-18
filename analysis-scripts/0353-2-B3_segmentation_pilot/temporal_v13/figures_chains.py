"""All-model root-chain coherence diagnostics, fixed before/after fields."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import common as c
from coherence import chains

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);a=p.parse_args();out=a.run/'figures';caps=json.loads((out/'captions.json').read_text());rows=json.loads((a.run/'chain_coherence_summary.json').read_text())['records']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    def save(fig,name,caption,raster=False):
        fig.savefig(out/(name+'.png'),dpi=600,bbox_inches='tight');fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
        if not raster:fig.savefig(out/(name+'.svg'),bbox_inches='tight')
        with Image.open(out/(name+'.png')) as im:im.save(out/(name+'.tif'),compression='tiff_lzw',dpi=(600,600))
        plt.close(fig);caps[name]=caption
    fig,axs=plt.subplots(2,3,figsize=(10,6),layout='constrained')
    for i,split in enumerate(['validation','repeated_benchmark']):
        for j,(field,title) in enumerate([('switches_per_1000_edges','Class switches / 1000 chain edges'),('runs_shorter_than_5px','Class runs shorter than 5 px'),('false_switches_on_same_human_class','False switches on same human class')]):
            ax=axs[i,j]
            for version,offset,col in [('v12',-.18,'#B0B8BF'),('v13',.18,'#0072B2')]:
                vals=[next(r[field] for r in rows if r['model']==n and r['split']==split and r['version']==version) for n in c.MODELS];ax.bar(np.arange(3)+offset,vals,.34,color=col,label=version)
            ax.set_xticks(range(3),['U-Net','DeepLabV3+','SegFormer']);ax.set_title(title);ax.set_ylabel(split.replace('_',' '));ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axs[0,0].legend(frameon=False);fig.suptitle('Root-segment coherence · shared frozen root support')
    save(fig,'Fig11_root_chain_coherence','All models and both regions, with no outcome filtering. Reduced class switching is a coherence diagnostic, not independent proof of correctness: real class boundaries also count. Right panels count only switches between adjacent predicted skeleton points lying on the same typed human class. Graph junctions split paths. Original pixel metrics must be read alongside these plots.')
    # Field selection uses ONLY prior v12 predictions, not improvements or new GT.
    t='T2';old=np.array(Image.open(a.previous/'unet/evaluation/validation'/f'{t}_classes.png'));_,nodes,paths,_=chains(old>0);points=np.zeros(old.shape,np.float32)
    for path in paths:
        xy=nodes[path];v=old[tuple(xy.T)];xy=xy[1:][v[1:]!=v[:-1]]
        if len(xy):points[tuple(xy.T)]=1
    density=ndi.uniform_filter(points,192,mode='constant');y,x=np.unravel_index(density.argmax(),old.shape);s=320;x=int(np.clip(x-s//2,0,old.shape[1]-s));y=int(np.clip(y-s//2,0,old.shape[0]-s));sl=(slice(y,y+s),slice(x,x+s))
    c.js(a.run/'chain_visual_ROI.json',dict(period=t,column=3,x=x,y=y,side=s,selection='Maximum192px v12 U-Net class-switch density only; new outputs not used'))
    fig,axs=plt.subplots(3,4,figsize=(11,8.6),layout='constrained')
    rgb=np.array(Image.open(a.run/'data/upper'/f'{t}_RGB.png'))[:,c.COLS[3]:c.COLS[3]+448]
    for i,n in enumerate(c.MODELS):
        gt=np.array(Image.open(a.run/n/'evaluation/validation'/f'{t}_human_target.png'));human=np.full((*gt.shape,3),180,np.uint8);valid=gt!=255;human[valid]=c.PALETTE[gt[valid]]
        ims=[rgb[sl],human[sl]]+[c.PALETTE[np.array(Image.open(root/n/'evaluation/validation'/f'{t}_classes.png'))[sl]] for root in [a.previous,a.run]]
        for ax,im in zip(axs[i],ims):ax.imshow(im,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
        axs[i,0].set_ylabel(c.MODEL_NAMES[n])
    for ax,title in zip(axs[0],['Original RGB','Human labels','v12','v13']):ax.set_title(title)
    fig.legend(handles=[Patch(color=c.PALETTE[k]/255,label=label) for k,label in [(1,'Persistent'),(2,'New'),(3,'Tip growth')]],loc='outside lower center',ncol=3,frameon=False);fig.suptitle('Fixed C3 / T1 → T2 field · root-segment label switching')
    save(fig,'Fig12_fixed_chain_detail','One native320px field for all models, human reference, and both versions. Selected by old v12 U-Net switch density alone. No class recoloring or resampling of native outputs. Fewer switches do not establish identity through a crossing. Persisting parent portions and terminal extensions may legitimately differ.',True)
    c.js(out/'captions.json',caps);(out/'Figure_captions_EN.md').write_text('\n\n'.join(f'{k}: {v}' for k,v in caps.items())+'\n');print('CHAIN_FIGURES_READY',flush=True)
if __name__=='__main__':main()

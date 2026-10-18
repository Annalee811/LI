"""Old/new plots for all3 models and all3 classes, with fixed junction ROIs."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import common as c

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);a=p.parse_args();run=a.run;out=run/'figures';rows=list(csv.DictReader((run/'before_after_class_metrics.csv').open(encoding='utf-8-sig')));caps=json.loads((out/'captions.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name,caption):
        fig.savefig(out/(name+'.png'),dpi=600,bbox_inches='tight');fig.savefig(out/(name+'.pdf'),bbox_inches='tight');fig.savefig(out/(name+'.svg'),bbox_inches='tight')
        with Image.open(out/(name+'.png')) as im:im.save(out/(name+'.tif'),compression='tiff_lzw',dpi=(600,600))
        plt.close(fig);caps[name]=caption
    for split in ['validation','repeated_benchmark']:
        fig,axs=plt.subplots(3,3,figsize=(10,8.4),layout='constrained')
        for i,region in enumerate(['all','junction','previous_tip']):
            for j,n in enumerate(c.MODELS):
                ax=axs[i,j]
                for version,offset,col in [('v10',-.18,'#B0B8BF'),('v11',.18,'#0072B2')]:
                    rr=[r for r in rows if r['model']==n and r['region']==region and r['split']==split and r['version']==version];values=[float(r['Dice']) if r['Dice'] else 0 for r in rr];ax.bar(np.arange(3)+offset,values,.34,color=col,label=version)
                ax.set_ylim(0,1);ax.set_xticks(range(3),['Persistent','New','Growth']);ax.set_title(c.MODEL_NAMES[n]+' · '+region.replace('_',' '));ax.set_ylabel('Strict temporal Dice');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        axs[0,0].legend(frameon=False);fig.suptitle('Before / after · '+split.replace('_',' '))
        save(fig,'Fig9_before_after_'+split,'Same reference and frozen first-stage root masks for both versions. All three classes and all three models are shown, globally, within12px of a diagonal-corrected skeleton junction and within20px of a previous predicted tip. Fixed0–1 scales; reduced scores are retained. C3 selects checkpoints; C4 is a repeated same-position benchmark. Neither is an independent tube test.')
    roi=json.loads((run/'junction_visual_ROI.json').read_text());t=roi['period'];x,y,s=[roi[k] for k in ['x','y','side']]
    fig,axs=plt.subplots(3,3,figsize=(8.4,8.8),layout='constrained')
    for i,n in enumerate(c.MODELS):
        gt=np.array(Image.open(run/n/'evaluation/validation'/f'{t}_human_target.png'))[y:y+s,x:x+s];v=np.full((*gt.shape,3),180,np.uint8);valid=gt!=255;v[valid]=c.PALETTE[gt[valid]];axs[i,0].imshow(v)
        for j,base in [(1,a.previous),(2,run)]:
            pred=np.array(Image.open(base/n/'evaluation/validation'/f'{t}_classes.png'))[y:y+s,x:x+s];axs[i,j].imshow(c.PALETTE[pred])
        for j in range(3):axs[i,j].set_xticks([]);axs[i,j].set_yticks([])
        axs[i,0].set_ylabel(c.MODEL_NAMES[n])
    for ax,title in zip(axs[0],['Human labels','v10','v11']):ax.set_title(title)
    fig.suptitle('Fixed C3 / T2 root-class error region')
    save(fig,'Fig10_fixed_junction_region','Same native256px C3 T2 region for all models and both versions. ROI chosen by maximum old-v10 U-Net error density only, before inspecting v11 output; coordinates and selection rule saved. These panels are error analysis, not an independent performance estimate. Persistent purple, new green, growth gold; excluded reference gray.')
    c.js(out/'captions.json',caps);(out/'Figure_captions_EN.md').write_text('\n\n'.join(f'{k}: {v}' for k,v in caps.items())+'\n')
    print('BEFORE_AFTER_FIGURES_READY',flush=True)
if __name__=='__main__':main()

"""Save completed model monitors immediately; never display unfinished epochs."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--model',required=True);a=p.parse_args();meta=json.loads((a.run/a.model/'training_run.json').read_text());rr=list(csv.DictReader((a.run/a.model/'history.csv').open(encoding='utf-8-sig')));assert len(rr)==meta['epochs_completed'];xx=[int(r['epoch']) for r in rr];out=a.run/'completed_model_figures';out.mkdir(exist_ok=True);name={'unet':'U-Net','deeplab':'DeepLabV3+','segformer':'SegFormer-B0'}[a.model]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'xtick.labelsize':7,'ytick.labelsize':7})
    for kind in ['loss','LR']:
        fig,ax=plt.subplots(figsize=(3.65,3.0));fig.subplots_adjust(left=.18,right=.97,top=.86,bottom=.26);ax.axvspan(1,20.5,color='#F2F4F6',zorder=0);ax.axvline(20.5,ls=':',color='#8B929A',lw=.8)
        if kind=='loss':
            ax.plot(xx,[float(r['train_loss']) for r in rr],color='#0072B2',lw=1.2,label='Training Loss');ax.plot(xx,[float(r['validation_loss']) for r in rr],color='#D55E00',lw=1.2,label='Validation Loss');ax.set_ylabel('Loss');ax.set_ylim(0,max(2.5,max(float(r['validation_loss']) for r in rr)*1.1));ax.legend(frameon=False,fontsize=7)
        else:ax.plot(xx,[float(r['learning_rate']) for r in rr],color='#009E73',lw=1.2);ax.set_ylabel('Learning rate');ax.ticklabel_format(axis='y',style='sci',scilimits=(-5,-5),useMathText=True)
        ax.set(xlim=(1,xx[-1]+.5),xlabel='Epoch',title=name+' · all measured epochs');ax.grid(color='#E5E9ED',lw=.5);ax.set_axisbelow(True);fig.text(.56,.075,'1–20: replay measurements',ha='center',fontsize=7,color='#5F6873');fig.text(.56,.03,'21 onward: final-weight continuation',ha='center',fontsize=7,color='#5F6873');png=out/(a.model+'_'+kind+'_preview.png');fig.savefig(png,dpi=180);temp=out/(a.model+'_'+kind+'_600.png');fig.savefig(temp,dpi=600);plt.close(fig)
        with Image.open(temp) as im:im.convert('RGB').save(out/(a.model+'_'+kind+'.tif'),compression='tiff_lzw',dpi=(600,600))
        temp.unlink()
    print('COMPLETE_MODEL_MONITORS_SAVED',a.model,len(rr),flush=True)

if __name__=='__main__':main()

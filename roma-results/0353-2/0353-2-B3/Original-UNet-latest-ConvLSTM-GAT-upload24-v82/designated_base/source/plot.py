"""Plot recorded values only; unfinished figures are explicitly marked as partial."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text
from PIL import Image


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    models=['unet','deeplab','segformer']; names=['U-Net branch','DeepLabV3+ branch','SegFormer-B0 branch']
    histories={}
    for m in models:
        path=a.run/m/'history.csv'
        if path.exists():
            with path.open(encoding='utf-8-sig') as f:histories[m]=list(csv.DictReader(f))
    if not histories:return
    complete=all((a.run/m/'completion.json').exists() for m in models)
    dest=a.run/'figures';dest.mkdir(exist_ok=True)
    font_path=a.run.parents[0]/'B3_fixed_100_epochs_v35_20261023/assets/TimesNewRoman-Bold.ttf'
    assert font_path.exists(),font_path
    plt.rcParams.update({'font.size':11,'axes.labelweight':'bold','axes.titleweight':'bold','axes.linewidth':1.1,'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black'})
    for kind in ['loss','learning_rate','validation_metrics']:
        fig,axes=plt.subplots(1,3,figsize=(12,3.65),layout='constrained')
        for i,(m,name,ax) in enumerate(zip(models,names,axes)):
            data=histories.get(m,[])
            ax.set_title(name);ax.text(-.12,1.08,'abc'[i],transform=ax.transAxes,fontweight='bold',fontsize=15)
            ax.set_xlabel('Temporal-head training epoch');ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
            if not data:ax.text(.5,.5,'Not started',transform=ax.transAxes,ha='center');continue
            if kind=='loss':
                rows=data
                for field,label,color in [('training_loss','Training loss','#0072B2'),('validation_loss','Validation loss','#D55E00')]:
                    ax.plot([int(r['epoch']) for r in rows],[float(r[field]) for r in rows],label=label,color=color,lw=1.5)
                ax.set_ylabel('Temporal composite loss')
            elif kind=='learning_rate':
                rows=[r for r in data if int(r['epoch'])>0]
                ax.plot([int(r['epoch']) for r in rows],[float(r['learning_rate']) for r in rows],color='#0072B2',lw=1.6,label='AdamW learning rate')
                ax.set_ylabel('Learning rate');ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
            else:
                for field,label,color in [('validation_macro_Dice','Macro Dice','#0072B2'),('validation_macro_IoU','Macro IoU','#D55E00'),('validation_root_accuracy','Root pixel accuracy','#009E73')]:
                    ax.plot([int(r['epoch']) for r in data],[float(r[field]) for r in data],label=label,color=color,lw=1.5)
                ax.set_ylabel('Agreement with uploaded reference');ax.set_ylim(0,1)
            ax.legend(frameon=False,fontsize=8)
        if not complete:fig.suptitle('Training in progress — recorded epochs only',fontsize=12)
        fig.canvas.draw()
        for item in fig.findobj(Text):
            item.set_color('black');item.set_fontproperties(FontProperties(fname=str(font_path),size=item.get_fontsize(),weight='bold'))
        filename=('Fig_' if complete else 'Preview_')+'Three_temporal_branches_'+kind+'.tif'
        temporary=dest/(filename+'.png')
        fig.savefig(temporary,dpi=600,facecolor='white')
        with Image.open(temporary) as im:im.convert('RGB').save(dest/filename,format='TIFF',compression='tiff_adobe_deflate',dpi=(600,600))
        temporary.unlink();plt.close(fig)
    (dest/'plot_status.json').write_text(json.dumps(dict(complete=complete,last_recorded_epoch={m:int(v[-1]['epoch']) for m,v in histories.items()},smoothing=False,interpolation=False,reference='Uploaded (18) pixel labels; internal agreement only'),indent=2)+'\n')


if __name__=='__main__':main()

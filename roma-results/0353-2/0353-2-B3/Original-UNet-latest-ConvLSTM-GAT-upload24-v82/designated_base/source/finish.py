"""Actual selected-head inference, separate three-class metrics, no forced labels."""
import argparse, json, os, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import common as c

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    prep=json.loads((a.run/'preparation.json').read_text());metrics=[]
    for m in c.MODELS:
        folder=a.run/m;data=c.Data(a.run,prep['roots'],m)
        ck=torch.load(folder/'best_temporal_head.pt',weights_only=True,map_location='cpu')
        model=c.AxisHead().to(memory_format=torch.channels_last);model.load_state_dict(ck['state_dict'])
        full=folder/'full';full.mkdir(exist_ok=True)
        for t in c.PERIODS:
            dest=full/f'{t}_classes.png'
            if not dest.exists():
                if t=='T1':pred=np.where(data.gate[t],4,0).astype(np.uint8)
                else:
                    q=c.predict_probability(model,data.features[t],torch.device('cpu'))
                    assert np.isfinite(q).all() and np.allclose(q.sum(0),1,atol=1e-5)
                    tmp=full/f'{t}_probability.tmp'
                    with tmp.open('wb') as f:np.savez_compressed(f,probability=q.astype(np.float16))
                    os.replace(tmp,full/f'{t}_probability.npz')
                    pred=np.where(data.gate[t],q.argmax(0)+1,0).astype(np.uint8)
                Image.fromarray(pred).save(dest)
            pred=np.array(Image.open(dest));assert np.array_equal(pred>0,data.gate[t])
            print('FULL',m,t,'binary support unchanged',flush=True)
        measured=json.loads((folder/'test_metrics.json').read_text())
        for row in measured['classes']:metrics.append(dict(model=c.MODEL_NAMES[m],selected_epoch=ck['epoch'],evaluation='inner spatial holdout, annotated pixels only',**row))
        h=c.rows(folder/'history.csv');assert [int(r['epoch']) for r in h]==list(range(101))
        initial=torch.load(folder/'initial_head.pt',weights_only=True,map_location='cpu')['state_dict']
        assert any(not torch.equal(initial[k],ck['state_dict'][k]) for k in initial)
        assert ck['config']['GAT'] is False and ck['config']['old_temporal_checkpoint_loaded'] is False
    c.csvout(a.run/'Three_models_temporal_class_metrics.csv',metrics)
    # Quantitative class panels are independent, so no rare-class score is hidden in a macro mean.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.text import Text
    font=Path('/workspace/LI/results/B3_fixed_100_epochs_v35_20261023/assets/TimesNewRoman-Bold.ttf')
    figdir=a.run/'figures';figdir.mkdir(exist_ok=True)
    for cid,title,stem in [(1,'Persistent roots','Persistent'),(2,'Newly observed roots','New'),(3,'Tip extension','Growing')]:
        fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
        for m,col in zip(c.MODELS,['#0072B2','#D55E00','#009E73']):
            h=c.rows(a.run/m/'history.csv')
            for ax,field in zip(axes,['Dice','IoU']):
                vals=[float(r[f'validation_{cid}_{field}']) if r[f'validation_{cid}_{field}'] else float('nan') for r in h]
                ax.plot([int(r['epoch']) for r in h],vals,color=col,label=c.MODEL_NAMES[m],lw=1.5)
                ax.set_xlabel('Temporal-head training epoch');ax.set_ylabel('Validation '+field);ax.set_ylim(0,1);ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False)
        fig.suptitle(title+' — internal reference agreement')
        fig.canvas.draw()
        for tx in fig.findobj(Text):tx.set_color('black');tx.set_fontproperties(FontProperties(fname=str(font),size=tx.get_fontsize(),weight='bold'))
        tmp=figdir/(stem+'.png');fig.savefig(tmp,dpi=600,facecolor='white')
        with Image.open(tmp) as im:im.convert('RGB').save(figdir/f'Fig_{stem}_roots_metrics.tif',compression='tiff_adobe_deflate',dpi=(600,600))
        tmp.unlink();plt.close(fig)
    for item in prep['immutable_inputs']:assert c.sha(item['path'])==item['sha256']
    c.js(a.run/'final_verification.json',dict(passed=True,epochs_each=100,history_epochs=list(range(101)),all_three_classes_saved=True,full_binary_support_unchanged=True,all_source_hashes_unchanged=True,GAT=False,old_temporal_state_loaded=False))
    import subprocess, sys
    subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('render.py')),'--run',str(a.run)],check=True)
    c.js(a.run/'completion.json',dict(state='complete',epochs_each=100,GAT=False,source_SHA256=prep['uploaded_reference_SHA256'],finished_unix=time.time(),models={m:json.loads((a.run/m/'completion.json').read_text()) for m in c.MODELS}))

if __name__=='__main__':main()

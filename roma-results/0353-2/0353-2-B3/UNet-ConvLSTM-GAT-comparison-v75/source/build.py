"""Completed U-Net V74 preview against immutable V66, latest20 reference only."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from scipy import ndimage as ndi
ROOT=Path('/workspace/LI');RUN=ROOT/'results/B3_ConvLSTM_GAT_upload20_v74_20261028';BASE=ROOT/'results/B3_temporal_upload18_noGAT_v66_20261028'
OUT=ROOT/'results/B3_ConvLSTM_GAT_UNet_preview_v75_20261028';HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(RUN/'source'))
from model import TemporalGraph,hierarchical_prediction
from train import measures
from prepare import js,sha,csvout,PERIODS
FONT=ROOT/'results/B3_fixed_100_epochs_v35_20261023/assets/TimesNewRoman-Bold.ttf'
PAL=np.array([[255,255,255],[136,86,167],[0,158,115],[230,171,2],[0,114,178]],np.uint8)
def isolated(c,mask=None):
    im=np.full((*c.shape,3),255,np.uint8);fg=np.isin(c,[1,2,3,4]);im[fg]=PAL[c[fg]]
    if mask is not None:im[mask&~fg]=[148,148,148]
    return im
def overlay(rgb,c,mask=None):
    im=rgb.copy();fg=np.isin(c,[1,2,3,4]);im[fg]=np.rint(.25*rgb[fg]+.75*PAL[c[fg]]).astype(np.uint8)
    return im
def cut(a,r):x,y,x1,y1=r;return a[y:y1,x:x1]
class Canvas:
    def __init__(self,w,h):self.im=Image.new('RGB',(w,h),'white');self.d=ImageDraw.Draw(self.im);self.labels=[]
    def text(self,x,y,s,size=44,anchor='mt'):
        font=ImageFont.truetype(str(FONT),size);box=self.d.textbbox((x,y),s,font=font,anchor=anchor);assert min(box[:2])>=0 and box[2]<=self.im.width and box[3]<=self.im.height,(s,box)
        self.d.text((x,y),s,font=font,fill='black',anchor=anchor);self.labels.append(dict(text=s,bounds=box))
    def panel(self,a,x,y,w,h,cat=False):self.im.paste(Image.fromarray(a).resize((w,h),Image.Resampling.NEAREST if cat else Image.Resampling.LANCZOS),(x,y));self.d.rectangle((x,y,x+w-1,y+h-1),outline='#DFE4E9',width=2)
    def legend(self,y):
        for j,(s,k) in enumerate([('T1 baseline',4),('Persistent',1),('Newly observed',2),('Tip extension',3)]):
            x=200+j*(self.im.width-300)//4;self.d.rectangle((x,y,x+36,y+36),fill=tuple(PAL[k]));self.text(x+52,y,s,36,'lt')
    def save(self,stem):
        p=OUT/(stem+'.tif');self.im.save(p,compression='tiff_adobe_deflate',dpi=(600,600));prev=self.im.copy();prev.thumbnail((1800,1900));prev.save(OUT/(stem+'_preview.png'))
        return dict(file=p.name,sha256=sha(p),dimensions=list(self.im.size),typography=self.labels)

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);OUT.mkdir(exist_ok=True);(OUT/'optimized/full').mkdir(parents=True,exist_ok=True)
    folder=RUN/'unet_convlstm_gat';done=json.loads((folder/'completion.json').read_text());assert done['epochs_completed']==100
    ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);assert sha(folder/'best.pt')==done['checkpoint_sha256'];model=TemporalGraph('convlstm_gat');model.load_state_dict(ck['state_dict']);model.eval()
    cached=dict(np.load(folder/'inference.npz'));h=torch.from_numpy(cached['h']);group=cached['groups'];state=cached['group_new_probability'];owners=dict(np.load(RUN/'data/unet/ownership.npz'))
    pred={};before={};ref={};rgb={};masks={}
    for t in PERIODS:
        gate=owners[t]>=0;px=dict(np.load(RUN/'data/unet'/(t+'_pixels.npz')));before[t]=np.array(Image.open(BASE/'unet/full'/f'{t}_classes.png'))
        if t=='T1':p=np.where(gate,4,0).astype(np.uint8)
        else:
            values=[]
            with torch.inference_mode():
                for lo in range(0,len(px['node']),16384):
                    ix=slice(lo,lo+16384);logits=model.pixel_logits(h,torch.from_numpy(px['node'][ix]),torch.from_numpy(px['pf'][ix])).numpy();values.append(hierarchical_prediction(logits,px['node'][ix],group,state,done['new_state_threshold']))
            p=np.zeros((1108,2541),np.uint8);p.ravel()[px['pixel']]=np.concatenate(values)+1
        assert np.array_equal(p>0,gate) and np.array_equal(before[t]>0,gate)
        pred[t]=p;Image.fromarray(p).save(OUT/'optimized/full'/f'{t}_classes.png');ref[t]=np.array(Image.open(RUN/'data/reference'/f'{t}_classes.png'));masks[t]=np.array(Image.open(RUN/'data/reference'/f'{t}_mask.png'))==255;rgb[t]=np.array(Image.open(BASE/'data/upper'/f'{t}_RGB.png'))
    test=dict(np.load(RUN/'data/unet/test.npz'));met={};metricrows=[]
    for name,maps in [('original',before),('ConvLSTM_GAT_root_state',pred)]:
        vals=np.zeros(len(test['node']),np.int64)
        for ti,t in enumerate(PERIODS):
            select=test['phase']==ti
            if select.any():vals[select]=maps[t].ravel()[test['pixel'][select]]-1
        met[name]=measures(test['target'],vals)
        for r in met[name]['classes']:metricrows.append(dict(model='U-Net',variant=name,reference='Latest20; fixed-root internal spatial holdout',**r))
    assert met['ConvLSTM_GAT_root_state']['confusion']==done['test']['confusion']
    for r in done['raw_auxiliary']['classes']:metricrows.append(dict(model='U-Net',variant='ConvLSTM_GAT_flat_auxiliary',reference='Latest20; auxiliary pre-root-state output, not final structured result',**r))
    csvout(OUT/'UNet_before_after_latest20_metrics.csv',metricrows);js(OUT/'metrics.json',met)
    # Reference-only common local window. No model accuracy enters this selection.
    best=None
    for y in range(0,1108-384+1,48):
        for x in range(0,2541-384+1,48):
            cc=[int((ref[t][y:y+384,x:x+384]==k).sum()) for t in PERIODS[1:] for k in [2,3]];score=sum(np.log1p(cc))
            if best is None or score>best[0]:best=(score,[x,y,x+384,y+384],cc)
    roi=best[1];outputs=[]
    cv=Canvas(4900,4200);cv.text(2450,35,'U-Net temporal classification: original vs ConvLSTM + GAT',76);cv.text(2450,145,'Same revised reference (20) | 100 optimization epochs | Selected epoch '+str(done['selected_epoch']),38)
    xs=[170,1350,2550,3750];w=1000;hh=436
    for x,label in zip(xs,['Original RGB','Revised reference','Original U-Net branch','U-Net + ConvLSTM + GAT']):cv.text(x+w//2,245,label,40)
    for i,t in enumerate(PERIODS):
        y=310+i*925;cv.text(75,y+440,t,60);cv.panel(rgb[t],xs[0],y+220,w,hh)
        x0,y0,x1,y1=roi;cv.d.rectangle((xs[0]+round(x0/2541*w),y+220+round(y0/1108*hh),xs[0]+round(x1/2541*w),y+220+round(y1/1108*hh)),outline='#B88A00',width=5)
        for x,a,mask in [(xs[1],ref[t],masks[t]),(xs[2],before[t],None),(xs[3],pred[t],None)]:cv.panel(overlay(rgb[t],a,mask),x,y,w,hh);cv.panel(isolated(a,mask),x,y+hh+15,w,hh,True)
    cv.legend(4090);outputs.append(cv.save('Fig1_UNet_ConvLSTM_GAT_four_period_comparison'))
    cv=Canvas(5680,3420);cv.text(2840,35,'Matched local temporal detail: original vs ConvLSTM + GAT',78);cv.text(2840,145,'Same384px window in all four periods; no binary shape, width or length changes',40)
    for i,t in enumerate(PERIODS):
        bx=70+(i%2)*2820;by=240+(i//2)*1490;cv.text(bx+10,by,t,60,'lt');xx=[bx+30,bx+720,bx+1400,bx+2080];ww=600
        for x,label in zip(xx,['RGB','Reference','Original','ConvLSTM + GAT']):cv.text(x+ww//2,by+80,label,40)
        raw=cut(rgb[t],roi);cv.panel(raw,xx[0],by+450,ww,ww)
        for x,a,mask in [(xx[1],cut(ref[t],roi),cut(masks[t],roi)),(xx[2],cut(before[t],roi),None),(xx[3],cut(pred[t],roi),None)]:cv.panel(overlay(raw,a,mask),x,by+145,ww,ww);cv.panel(isolated(a,mask),x,by+770,ww,ww,True)
    cv.legend(3310);outputs.append(cv.save('Fig2_UNet_ConvLSTM_GAT_matched_local'))
    # True extension reference windows, including T4, selected without model accuracy.
    cv=Canvas(4900,3370);cv.text(2450,35,'Tip-extension detail across follow-up periods',78);cv.text(2450,145,'Reference-only extension windows | Original and optimized predictions kept unchanged',38);tiprois=[]
    for i,t in enumerate(PERIODS[1:]):
        lab,n=ndi.label(ref[t]==3);sizes=np.bincount(lab.ravel());sizes[0]=0;yy,xx=np.where(lab==sizes.argmax());x=int(np.clip(xx.mean()-128,0,2541-256));y=int(np.clip(yy.mean()-128,0,1108-256));r=[x,y,x+256,y+256];tiprois.append(dict(period=t,ROI=r))
        top=260+i*970;cv.text(80,top+440,t,58);raw=cut(rgb[t],r);xxs=[200,1400,2600,3800];ww=850
        for px,label in zip(xxs,['RGB','Reference','Original','ConvLSTM + GAT']):cv.text(px+ww//2,top,label,42)
        cv.panel(raw,xxs[0],top+70,ww,ww)
        for px,a,mask in [(xxs[1],cut(ref[t],r),cut(masks[t],r)),(xxs[2],cut(before[t],r),None),(xxs[3],cut(pred[t],r),None)]:cv.panel(isolated(a,mask),px,top+70,ww,ww,True)
    cv.legend(3260);outputs.append(cv.save('Fig3_UNet_ConvLSTM_GAT_extension_detail'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.text import Text
    with (folder/'history.csv').open(encoding='utf-8-sig') as f:history=list(csv.DictReader(f))
    zero=json.loads((folder/'epoch0_metrics.json').read_text());assert [int(v['epoch']) for v in history]==list(range(1,101));epoch=[0]+[int(v['epoch']) for v in history]
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    axes[0].plot(epoch,[zero['training_monitoring_CE']]+[float(v['training_monitoring_CE']) for v in history],label='Training CE',color='#0072B2');axes[0].plot(epoch,[zero['validation_monitoring_CE']]+[float(v['validation_pixel_CE']) for v in history],label='Validation CE',color='#D55E00');axes[0].set_ylabel('Weighted pixel cross entropy');axes[0].set_title('100 real optimization epochs')
    for k,field,label,col in [(0,'validation_persistent_Dice','Persistent','#8856A7'),(1,'validation_new_Dice','Newly observed','#009E73'),(2,'validation_extension_Dice','Tip extension','#B47E00')]:axes[1].plot(epoch,[zero['validation']['classes'][k]['Dice']]+[float(v[field]) for v in history],label=label,color=col)
    axes[1].set_ylabel('Validation structured Dice');axes[1].set_ylim(0,1);axes[1].set_title('Three temporal categories')
    axes[2].plot([int(v['epoch']) for v in history],[float(v['learning_rate']) for v in history],label='AdamW learning rate',color='#0072B2');axes[2].set_ylabel('Learning rate');axes[2].ticklabel_format(axis='y',style='sci',scilimits=(0,0));axes[2].set_title('Learning-rate schedule')
    for ax in axes:ax.set_xlabel('Optimization epoch');ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=8)
    fig.canvas.draw()
    for tx in fig.findobj(Text):tx.set_color('black');tx.set_fontproperties(FontProperties(fname=str(FONT),size=tx.get_fontsize(),weight='bold'))
    fig.savefig(OUT/'Fig4_UNet_ConvLSTM_GAT_100_epochs_preview.png',dpi=160,facecolor='white');tmp=OUT/'training_full.png';fig.savefig(tmp,dpi=600,facecolor='white');plt.close(fig)
    tif=OUT/'Fig4_UNet_ConvLSTM_GAT_100_epochs.tif'
    with Image.open(tmp) as im:im.convert('RGB').save(tif,compression='tiff_adobe_deflate',dpi=(600,600))
    tmp.unlink();outputs.append(dict(file=tif.name,sha256=sha(tif)))
    js(OUT/'verification.json',dict(passed=True,source='Latest20',source_sha256=json.loads((RUN/'preparation.json').read_text())['upload20_sha256'],epochs=100,selected_epoch=done['selected_epoch'],checkpoint_sha256=sha(folder/'best.pt'),original_binary_geometry_preserved_all_periods=True,heldout_predictions_match_training_evaluation=True,reference_ROI=roi,reference_ROI_class_counts=best[2],extension_ROIs=tiprois,no_smoothing=True,illustrative_not_independent_holdout=True,figures=outputs))
    (OUT/'RESULTS.md').write_text('''# U-Net + ConvLSTM + GAT completed comparison

V74 completed100 real epochs, validation selected epoch45. Original V66 predictions
and optimized structured predictions are evaluated against exactly the same latest20
reference pixels and frozen binary supports. Historical V66 curves were trained on18
and are preserved separately; this file does not relabel their historical losses.

Current conditional Dice (persistent/new/tip): original .9711/.8229/.0276;
ConvLSTM+GAT root-state output .9270/.7154/.5757. Macro .6072 -> .7394.
Tip recognition improved, while new/persistent Dice declined. This is not an all-class
improvement. New labels are revised/model-assisted, and all data are one B3 field;
scores are internal consistency, not independent generalization. Same-label controls
are retained in the full experiment; a before/after comparison alone does not isolate
the effect of the two added modules from the updated supervision.

Figures show native original/reference/optimized pixels, white isolated-root background,
the same reference-selected ROI across dates and extra true-extension windows. ROI
views are illustrative and may include development pixels. No label/shape edit, color
majority voting, synthetic error or invented epoch is used. Flat auxiliary pixel scores
are separately labeled in CSV and are not mixed into the final structured predictions.
''')
    print(json.dumps(dict(original=met['original']['macro_Dice'],optimized=met['ConvLSTM_GAT_root_state']['macro_Dice'],figures=len(outputs)),indent=2),flush=True)
if __name__=='__main__':main()

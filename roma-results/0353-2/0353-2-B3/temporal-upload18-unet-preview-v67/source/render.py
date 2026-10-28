"""Native temporal class figures and unsmoothed actual U-Net epoch0..100 curves."""
import csv,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path('/workspace/LI');RUN=ROOT/'results/B3_temporal_upload18_noGAT_v66_20261028'
OUT=ROOT/'results/B3_temporal_upload18_UNet_preview_v67_20261028'
FONT=ROOT/'results/B3_fixed_100_epochs_v35_20261023/assets/TimesNewRoman-Bold.ttf'
PERIODS=['T1','T2','T3','T4'];COLORS=np.array([[255,255,255],[136,86,167],[0,158,115],[230,171,2],[0,114,178]],np.uint8)
ROI=[1472,704,1856,1088]
TIP_ROI=[2180,430,2436,686]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

class Canvas:
    def __init__(self,w,h):self.im=Image.new('RGB',(w,h),'white');self.d=ImageDraw.Draw(self.im);self.texts=[];self.panels=[]
    def text(self,x,y,v,size=46,anchor='mt'):
        font=ImageFont.truetype(str(FONT),size);b=self.d.textbbox((x,y),v,font=font,anchor=anchor)
        assert b[0]>=0 and b[1]>=0 and b[2]<=self.im.width and b[3]<=self.im.height,(v,b,self.im.size)
        self.d.text((x,y),v,font=font,anchor=anchor,fill='black');self.texts.append(dict(text=v,box=b,color='black',font='Times New Roman Bold'))
    def panel(self,a,box,tag,category=False):
        x,y,w,h=box
        self.im.paste(Image.fromarray(a).resize((w,h),Image.Resampling.NEAREST if category else Image.Resampling.LANCZOS),(x,y))
        self.d.rectangle((x,y,x+w-1,y+h-1),outline='#DDE3E8',width=2)
        self.panels.append(dict(tag=tag,box=[x,y,x+w,y+h],native_shape=list(a.shape),category_resampling='nearest' if category else None))
    def scale(self,x,y,w,h,nativew,px):
        length=round(w*px/nativew);bx=x+18;by=y+h-65
        self.d.rectangle((bx-8,by-5,bx+length+14,by+56),fill='white')
        self.d.rectangle((bx,by+39,bx+length,by+44),fill='black');self.text(bx+length/2,by,str(px)+' px',30)
    def legend(self,y):
        for i,(label,k) in enumerate([('T1 baseline',4),('Persistent',1),('Newly observed',2),('Tip extension',3)]):
            x=200+i*(self.im.width-300)//4
            self.d.rectangle((x,y,x+38,y+38),fill=tuple(COLORS[k]));self.text(x+56,y,label,38,'lt')
    def save(self,stem):
        path=OUT/(stem+'.tif');self.im.save(path,compression='tiff_adobe_deflate',dpi=(600,600))
        preview=self.im.copy();preview.thumbnail((1800,2000));preview.save(OUT/(stem+'_preview.png'))
        with Image.open(path) as im:assert im.mode=='RGB' and im.info['dpi']==(600,600)
        return dict(file=path.name,sha256=sha(path),dimensions=list(self.im.size),panels=self.panels,texts=self.texts)

def keep(a):return np.isin(a,[1,2,3,4])
def isolated(a,mask=None):
    image=np.full((*a.shape,3),255,np.uint8);valid=keep(a);image[valid]=COLORS[a[valid]]
    if mask is not None:image[mask&~valid]=[148,148,148]
    return image
def overlay(rgb,a,mask=None):
    image=rgb.copy();valid=keep(a);image[valid]=np.rint(.25*rgb[valid]+.75*COLORS[a[valid]]).astype(np.uint8)
    if mask is not None:
        unknown=mask&~valid;image[unknown]=np.rint(.25*rgb[unknown]+.75*np.array([148,148,148])).astype(np.uint8)
    return image
def cut(a,roi):x0,y0,x1,y1=roi;return a[y0:y1,x0:x1]

def main():
    verification=json.loads((OUT/'inference_verification.json').read_text());assert verification['passed']
    rgb={t:np.array(Image.open(RUN/'data/upper'/f'{t}_RGB.png')) for t in PERIODS}
    ref={t:np.array(Image.open(RUN/'data/current'/f'{t}_classes.png')) for t in PERIODS}
    refmask={t:np.array(Image.open(RUN/'data/full_annotation'/f'{t}_mask.png'))[:1108]==255 for t in PERIODS}
    pred={t:np.array(Image.open(OUT/'unet/full'/f'{t}_classes.png')) for t in PERIODS}
    sources=[dict(path=str(p),sha256=sha(p)) for t in PERIODS for p in [RUN/'data/upper'/f'{t}_RGB.png',RUN/'data/current'/f'{t}_classes.png',OUT/'unet/full'/f'{t}_classes.png']]
    # Full registered main field, using the exact pixel masks and white categorical background.
    cv=Canvas(4200,4680);cv.text(2100,35,'U-Net temporal classification across four periods',76)
    cv.text(2100,140,'Uploaded (18) reference | 100 training epochs | Validation-selected epoch 37 | No GAT',38)
    cv.text(2100,200,'Each reference/model column: RGB overlay above; classified roots on white below',36)
    xs=[240,1540,2840];w=1150;h=501
    for x,name in zip(xs,['Original RGB','Reference annotation','U-Net prediction']):cv.text(x+w//2,265,name,51)
    for i,t in enumerate(PERIODS):
        y=340+i*1030;cv.text(100,y+470,t,70,'mm')
        cv.panel(rgb[t],[xs[0],y+260,w,h],t+'/original');cv.scale(xs[0],y+260,w,h,2541,500)
        x0,y0,x1,y1=ROI;cv.d.rectangle((xs[0]+round(x0/2541*w),y+260+round(y0/1108*h),xs[0]+round(x1/2541*w),y+260+round(y1/1108*h)),outline='#BB8C00',width=5)
        for x,a,mask,name in [(xs[1],ref[t],refmask[t],'reference'),(xs[2],pred[t],None,'unet')]:
            cv.panel(overlay(rgb[t],a,mask),[x,y,w,h],t+'/'+name+'/overlay')
            cv.panel(isolated(a,mask),[x,y+h+12,w,h],t+'/'+name+'/roots',True)
    cv.legend(4530);cv.text(2100,4610,'Fixed binary root geometry; no dilation or forced temporal matches. Gray = untyped reference root.',34)
    outputs=[cv.save('Fig1_UNet_four_period_temporal_comparison')]
    # Same 384px reference-selected ROI in all four dates; T4 extension gets a separate honest inset.
    cv=Canvas(5000,4240);cv.text(2500,35,'Local temporal detail: reference annotation and U-Net',78)
    cv.text(2500,145,'Same 384 x 384 px window in T1-T4; additional T4 extension inset below',40)
    for i,t in enumerate(PERIODS):
        bx=70+(i%2)*2470;by=240+(i//2)*1370
        cv.text(bx+20,by,t,62,'lt')
        col=[bx+70,bx+870,bx+1650];w=700
        for x,name in zip(col,['Original RGB','Reference','U-Net']):cv.text(x+w//2,by+75,name,45)
        raw=cut(rgb[t],ROI);ar=cut(ref[t],ROI);ap=cut(pred[t],ROI);mr=cut(refmask[t],ROI)
        cv.panel(raw,[col[0]+65,by+430,570,570],t+'/ROI/original');cv.scale(col[0]+65,by+430,570,570,384,50)
        for x,a,mask,name in [(col[1],ar,mr,'reference'),(col[2],ap,None,'unet')]:
            cv.panel(overlay(raw,a,mask),[x+65,by+145,570,570],t+'/ROI/'+name+'/overlay')
            cv.panel(isolated(a,mask),[x+65,by+735,570,570],t+'/ROI/'+name+'/roots',True)
    y=3060;cv.d.line((80,y-30,4920,y-30),fill='#DAE2E8',width=3)
    cv.text(130,y,'T4: separate tip-extension window (256 x 256 px)',50,'lt')
    raw=cut(rgb['T4'],TIP_ROI);ar=cut(ref['T4'],TIP_ROI);ap=cut(pred['T4'],TIP_ROI);mr=cut(refmask['T4'],TIP_ROI)
    arrays=[('Original RGB',raw),('Reference overlay',overlay(raw,ar,mr)),('U-Net overlay',overlay(raw,ap)),('Reference roots',isolated(ar,mr)),('U-Net roots',isolated(ap))]
    for j,(name,a) in enumerate(arrays):
        x=135+j*970;cv.text(x+425,y+100,name,42);cv.panel(a,[x,y+165,850,850],'T4/tip_inset/'+name,j>=3)
    # Mark only the true reference extension location. Prediction panels remain unannotated.
    yy,xx=np.where(ar==3);assert len(xx)==48
    point=(135+970+round(xx.mean()/256*850),y+165+round(yy.mean()/256*850))
    cv.d.ellipse((point[0]-45,point[1]-32,point[0]+45,point[1]+32),outline='black',width=4)
    cv.legend(4125)
    outputs.append(cv.save('Fig2_UNet_local_temporal_detail'))
    # Recorded U-Net epoch0..100 curves, without smoothing or artificial completion.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.text import Text
    with (RUN/'unet/history.csv').open(encoding='utf-8-sig') as f:hist=list(csv.DictReader(f))
    assert [int(v['epoch']) for v in hist]==list(range(101))
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    epoch=[int(v['epoch']) for v in hist]
    for field,label,color in [('training_loss','Training loss','#0072B2'),('validation_loss','Validation loss','#D55E00')]:axes[0].plot(epoch,[float(v[field]) for v in hist],label=label,color=color,lw=1.5)
    axes[0].set_ylabel('Temporal composite loss');axes[0].set_title('U-Net training and validation')
    for k,label,col in [(1,'Persistent','#8856A7'),(2,'Newly observed','#009E73'),(3,'Tip extension','#B47E00')]:axes[1].plot(epoch,[float(v[f'validation_{k}_Dice']) for v in hist],label=label,color=col,lw=1.5)
    axes[1].set_ylabel('Validation Dice');axes[1].set_ylim(0,1);axes[1].set_title('Three temporal classes')
    axes[2].plot(epoch,[float(v['learning_rate']) for v in hist],color='#0072B2',lw=1.5,label='AdamW learning rate');axes[2].set_ylabel('Learning rate');axes[2].set_title('Learning-rate schedule');axes[2].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    for i,ax in enumerate(axes):
        ax.set_xlabel('Temporal-head training epoch');ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=9);ax.text(-.10,1.08,'abc'[i],transform=ax.transAxes,fontsize=16,fontweight='bold')
    fig.canvas.draw()
    for tx in fig.findobj(Text):tx.set_color('black');tx.set_fontproperties(FontProperties(fname=str(FONT),size=tx.get_fontsize(),weight='bold'))
    png=OUT/'Fig3_UNet_100_epoch_training_curves_preview.png';fig.savefig(png,dpi=180,facecolor='white')
    tmp=OUT/'training_full_resolution.png';fig.savefig(tmp,dpi=600,facecolor='white');plt.close(fig)
    path=OUT/'Fig3_UNet_100_epoch_training_curves.tif'
    with Image.open(tmp) as im:im.convert('RGB').save(path,compression='tiff_adobe_deflate',dpi=(600,600))
    tmp.unlink();outputs.append(dict(file=path.name,sha256=sha(path),epochs=101,smoothing=False))
    shutil.copy2(RUN/'unet/history.csv',OUT/'UNet_history_0_to_100.csv');shutil.copy2(RUN/'unet/test_metrics.json',OUT/'UNet_internal_holdout_metrics.json')
    for item in sources:assert sha(item['path'])==item['sha256']
    js(OUT/'figure_verification.json',dict(passed=True,outputs=outputs,sources=sources,ROI=ROI,T4_extension_ROI=TIP_ROI,
        native_predicted_geometry_unchanged=True,reference_classes_unchanged=True,source_input='upload18',
        local_ROI_counts=[dict(period=t,new=int((cut(ref[t],ROI)==2).sum()),extension=int((cut(ref[t],ROI)==3).sum())) for t in PERIODS],
        separate_T4_extension_reference_pixels=48,ROI_scope='Illustrative shared main field, not held-out-only; selection uses reference labels only',GAT=False))
    (OUT/'README.md').write_text('''# U-Net upload18 temporal preview, V67

U-Net completed100 real epochs in V66; validation selected epoch37. New inference
uses that exact checkpoint. No new training, GAT, learned routing, temporal class
constraints, root dilation or binary shape/width/length editing is applied.
The source reference is uploaded root_annotation_progress (18).json, preserved.
Reference roots without temporal typing are gray. Unreviewed background is not
shown as gray roots. T1 is baseline, not a learned change target.

Fig1: full four-period RGB/reference/U-Net overlays and white-background roots.
Fig2: same384px ROI in all dates; since T4 extension is not in this window, a
separate256px T4 inset shows all48 reference extension pixels, unaltered. Native
categorical roots use nearest-neighbor display. No missing class is fabricated.
ROIs are illustrative and reference-selected, not independent held-out test images.
Fig3: actual epoch0..100 loss, class Dice and learning-rate curves; no smoothing.
All labels Times New Roman Bold, black; 600dpi TIFF originals, PNG screen previews.

The reported JSON holdout metrics use predeclared within-field spatial masks;
they do not establish external tube/position generalization. Frozen binary misses
count as FN in end-to-end scores, and conditional scores are separately saved.
Other temporal branches may still be running; these figures are U-Net only.
''')
    print(json.dumps([dict(file=o['file'],sha256=o['sha256']) for o in outputs],indent=2),flush=True)
if __name__=='__main__':main()

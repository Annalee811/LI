"""Read-only native prediction figures; class pixels are never dilated or edited."""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import common as c

FONT=Path('/workspace/LI/results/B3_fixed_100_epochs_v35_20261023/assets/TimesNewRoman-Bold.ttf')
def label(draw,xy,text,size=42,anchor='mt'):
    draw.text(xy,text,font=ImageFont.truetype(str(FONT),size),fill='black',anchor=anchor)

def roi_from_reference(targets):
    # Illustration chosen using the reference only, before examining any prediction.
    # Balance new and extension visibility across all follow-up dates; not used for scoring.
    best=None
    for y in range(0,1108-384+1,64):
        for x in range(0,2541-384+1,64):
            counts=[int((targets[t][y:y+384,x:x+384]==k).sum()) for t in ['T2','T3','T4'] for k in [2,3]]
            score=sum(np.log1p(counts))
            if best is None or score>best[0]:best=(score,x,y,counts)
    _,x,y,counts=best
    return [x,y,x+384,y+384],counts

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    rgb={t:np.array(Image.open(a.run/'data/upper'/f'{t}_RGB.png')) for t in c.PERIODS}
    reference={t:np.array(Image.open(a.run/'data/current'/f'{t}_classes.png')) for t in c.PERIODS}
    roi,counts=roi_from_reference(reference);x0,y0,x1,y1=roi
    preds={m:{t:np.array(Image.open(a.run/m/'full'/f'{t}_classes.png')) for t in c.PERIODS} for m in c.MODELS}
    names=['Original RGB','Reference','U-Net','DeepLabV3+','SegFormer-B0']
    dest=a.run/'figures';dest.mkdir(exist_ok=True);files=[]
    for kind,title,periods,selected in [
        ('Four_period','Temporal root classes across four periods',c.PERIODS,None),
        ('Persistent','Persistent roots',['T2','T3','T4'],1),
        ('New','Newly observed roots',['T2','T3','T4'],2),
        ('Growing','Tip extension',['T2','T3','T4'],3)]:
        panelw=960;gap=30;left=190;top=240;rowh=1090
        canvas=Image.new('RGB',(left+5*(panelw+gap)+60,top+len(periods)*rowh+160),'white');draw=ImageDraw.Draw(canvas)
        label(draw,(canvas.width//2,30),title,72)
        label(draw,(canvas.width//2,125),'Uploaded (18) reference | Frozen binary supports | New temporal heads | No GAT',38)
        for j,name in enumerate(names):label(draw,(left+j*(panelw+gap)+panelw//2,185),name,46)
        for i,t in enumerate(periods):
            y=top+i*rowh; label(draw,(95,y+260),t,64,'mm')
            for j in range(5):
                x=left+j*(panelw+gap)
                if j==0:
                    arr=rgb[t]
                    full=Image.fromarray(arr).resize((panelw,419),Image.Resampling.LANCZOS)
                    canvas.paste(full,(x,y))
                    draw.rectangle((x+round(x0/2541*panelw),y+round(y0/1108*419),x+round(x1/2541*panelw),y+round(y1/1108*419)),outline='#B88A00',width=5)
                    crop=Image.fromarray(arr[y0:y1,x0:x1]).resize((600,600),Image.Resampling.LANCZOS)
                    canvas.paste(crop,(x+(panelw-600)//2,y+440))
                else:
                    a0=reference[t] if j==1 else preds[c.MODELS[j-2]][t]
                    keep=np.isin(a0,[1,2,3,4]) if selected is None else a0==selected
                    color=c.PALETTE[a0]
                    # Overlay full field and categorical local roots use the exact same native class masks.
                    blended=rgb[t].copy();blended[keep]=np.rint(.28*rgb[t][keep]+.72*color[keep]).astype(np.uint8)
                    canvas.paste(Image.fromarray(blended).resize((panelw,419),Image.Resampling.LANCZOS),(x,y))
                    local=np.full((y1-y0,x1-x0,3),255,np.uint8);support=keep[y0:y1,x0:x1]
                    local[support]=color[y0:y1,x0:x1][support]
                    # Nearest neighbor for categorical pixels; no interpolated or widened root labels.
                    canvas.paste(Image.fromarray(local).resize((600,600),Image.Resampling.NEAREST),(x+(panelw-600)//2,y+440))
            draw.line((left,y+1060,canvas.width-40,y+1060),fill='#E2E6EB',width=2)
        legend='Blue: T1 baseline    Purple: persistent    Green: newly observed    Gold: tip extension'
        label(draw,(canvas.width//2,canvas.height-110),legend,38)
        label(draw,(canvas.width//2,canvas.height-55),'Illustrative registered main field and shared ROI; internal agreement only; untyped reference pixels remain untyped',30)
        path=dest/f'Fig_{kind}_temporal_comparison.tif';canvas.save(path,compression='tiff_adobe_deflate',dpi=(600,600))
        files.append(dict(file=path.name,sha256=c.sha(path),dimensions=list(canvas.size)))
    c.js(a.run/'qualitative_figure_manifest.json',dict(files=files,ROI=roi,ROI_reference_counts_T2_new_extension_T3_new_extension_T4_new_extension=counts,
        ROI_selection='Reference-only coverage, no model comparison; visualization not an independent test',root_label_resampling='nearest neighbor; no dilation',source_SHA256=c.sha(a.run/'data/active_progress.json')))

if __name__=='__main__':main()

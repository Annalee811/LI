import sys,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path('/workspace/LI');OUT=ROOT/'results/B3_root_axis_junction_v83_20261028';OUT.mkdir(exist_ok=True);BASE=ROOT/'results/B3_temporal_upload18_noGAT_v66_20261028';LAST=ROOT/'results/B3_original_UNet_latestCG_upload24_v82_20261028';FONT=ROOT/'results/B3_fixed_100_epochs_v35_20261023/assets/TimesNewRoman-Bold.ttf'
PAL=np.array([[255,255,255],[136,86,167],[0,158,115],[230,171,2],[0,114,178]],np.uint8)
def iso(a):r=np.full((*a.shape,3),255,np.uint8);take=np.isin(a,[1,2,3,4]);r[take]=PAL[a[take]];return r
r=[1024,704,1280,960];x,y,x1,y1=r;cv=Image.new('RGB',(2050,1240),'white');dr=ImageDraw.Draw(cv);font=ImageFont.truetype(str(FONT),30)
for row,t in enumerate(['T1','T2','T3','T4']):
 rgb=np.array(Image.open(BASE/'data/upper'/f'{t}_RGB.png'));ref=np.array(Image.open(LAST/'data/reference'/f'{t}_classes.png'));orig=np.array(Image.open(BASE/'unet/full'/f'{t}_classes.png'));last=np.array(Image.open(LAST/'full'/f'{t}_classes.png'))
 for col,(label,a) in enumerate([('RGB',rgb),('Reference24',iso(ref)),('Original U-Net',iso(orig)),('Previous correction',iso(last))]):
  dr.text((col*510+10,row*310+2),t+' '+label,font=font,fill='black');cv.paste(Image.fromarray(a[y:y1,x:x1]).resize((500,260),Image.Resampling.NEAREST),(col*510+10,row*310+45))
cv.save(OUT/'case_four_period_inspection.png');(OUT/'review_ROI.json').write_text(json.dumps(dict(ROI=r,selection='User-identified unresolved case; diagnostic/regression illustration, not independent validation')))

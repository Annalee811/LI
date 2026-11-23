"""Render actual encoded predictions on registered RGB; does not calculate accuracy."""
from pathlib import Path
import csv,io,json
import numpy as np
from scipy import ndimage as ndi
from PIL import Image,ImageDraw,ImageFont
import fitz
W=Path(__file__).resolve().parents[1]
F=Path('/workspace/work/root_workflow_real_assets_20261109/fonts')
def font(size,bold=False):return ImageFont.truetype(str(F/('Times New Roman Bold.ttf' if bold else 'Times New Roman.ttf')),size)
def main():
 rows=list(csv.DictReader((W/'reports/annotation_checklist.csv').open(encoding='utf-8-sig')))
 positions=list(dict.fromkeys(r['position_id'] for r in rows));doc=fitz.open()
 dest=W/'reports/independent_preview';dest.mkdir(exist_ok=True)
 for pos in positions:
  rr={r['period']:r for r in rows if r['position_id']==pos}
  canvas=Image.new('RGB',(2600,1500),'white');draw=ImageDraw.Draw(canvas)
  draw.text((35,22),pos+' | Cross-tube prediction comparison',font=font(43,True),fill='black')
  draw.text((35,80),'Human temporal ground truth unavailable; these panels do not establish accuracy.',font=font(29),fill='black')
  labels=[('Original RGB',None),('Frozen V66',W/'independent_predictions/V66_pairwise_frozen/frozen'),('New full model',W/'independent_predictions/full/20261123'),('V90 transfer reference',W/'independent_predictions/V90_transfer_reference/frozen')]
  cols={1:(53,111,176),2:(231,168,53),3:(174,64,130),4:(235,235,235)}
  for j,t in enumerate(['T1','T2','T3','T4']):draw.text((290+j*570+240,126),t,font=font(32,True),fill='black')
  for i,(label,root) in enumerate(labels):
   y=185+i*286;draw.text((22,y+95),label,font=font(26,True),fill='black')
   for j,t in enumerate(['T1','T2','T3','T4']):
    x=290+j*570;box=(x,y,x+552,y+260)
    if t not in rr:
     draw.rectangle(box,outline='#DDDDDD');draw.text((x+155,y+105),'Not acquired',font=font(28),fill='black');continue
    rgb=np.array(Image.open(rr[t]['RGB_path']).convert('RGB'))
    if root is not None:
     m=np.array(Image.open(root/pos/(t+'_classes.png')));assert m.shape==rgb.shape[:2]
     # A one-pixel expansion only makes the plotted strokes legible; raw masks remain unchanged.
     mm=m.copy();empty=m==0
     for k in [4,1,2,3]: mm[empty & ndi.maximum_filter((m==k).astype(np.uint8),size=3).astype(bool)]=k
     for k,c in cols.items():rgb[mm==k]=(np.array(c)*.92+rgb[mm==k]*.08).astype(np.uint8)
    tile=Image.fromarray(rgb);tile.thumbnail((552,260),Image.Resampling.LANCZOS);canvas.paste(tile,(x+(552-tile.width)//2,y+(260-tile.height)//2))
  y=1350
  for x,(k,label) in zip([320,780,1260,1730],[(4,'First-observation root'),(1,'Persistent'),(2,'Newly observed'),(3,'Extending')]):
   draw.rectangle((x,y,x+32,y+22),fill=cols[k],outline='#555555');draw.text((x+46,y-7),label,font=font(28),fill='black')
  draw.text((35,1405),'New full model: fixed seed 20261123 (no test selection). V90 reference: documented transfer adapter.',font=font(26),fill='black')
  draw.text((35,1443),'Same registered RGB within each date; native masks are provided separately. Missing dates remain absent.',font=font(26),fill='black')
  canvas.save(dest/(pos+'.jpg'),quality=96,dpi=(300,300))
  b=io.BytesIO();canvas.save(b,format='JPEG',quality=96);page=doc.new_page(width=780,height=450);page.insert_image(page.rect,stream=b.getvalue())
 doc.save(W/'reports/Independent_prediction_preview.pdf',deflate=True)
 (dest/'preview_protocol.json').write_text(json.dumps(dict(positions=len(positions),frames=len(rows),model_seed=20261123,selection='first fixed seed, not selected by test quality',human_GT_available=False,metrics=None,stroke_display_expansion_pixels=1,raw_masks_unchanged=True),indent=2))
 print('PREVIEWS_COMPLETE',len(positions),flush=True)
if __name__=='__main__':main()

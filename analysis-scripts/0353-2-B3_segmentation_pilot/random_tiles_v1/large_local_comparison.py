"""Larger native central field for visual inspection, clearly labelled qualitative."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import common as c
from comparison import overlay,root_only

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();out=a.run/'comparison_RGB'
    # Fixed geometric center, not chosen from any model's performance.
    side=768;x=(2541-side)//2;y=(1108-side)//2;left=96;top=128;stride=side+24
    label=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',56)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',36)
    names=['Human']+c.MODELS
    for style in ['overlay','RGB','black_root']:
        ncols=5 if style=='overlay' else 4
        sheet=Image.new('RGB',(left+ncols*side+(ncols-1)*24,top+4*side+3*24),'white');dd=ImageDraw.Draw(sheet)
        headings=['Original RGB','Human annotation']+[c.NAMES[m] for m in c.MODELS] if style=='overlay' else ['Human annotation']+[c.NAMES[m] for m in c.MODELS]
        for j,title in enumerate(headings):dd.text((left+j*stride+9,12),title,fill='#222222',font=label)
        dd.text((left,78),'Central 768 px field | QUALITATIVE: contains fitted pixels | original coordinates and colors',fill='#333333',font=small)
        for i,t in enumerate(c.PERIODS):
            rgb=np.array(Image.open(a.run/'data/upper'/(t+'_RGB.png')).convert('RGB'))[y:y+side,x:x+side]
            gt=np.array(Image.open(a.run/'data/upper'/(t+'_mask.png')))[y:y+side,x:x+side]
            masks=[gt==255]+[np.array(Image.open(a.run/'native_predictions'/m/'upper'/(t+'_root_mask.png')))[y:y+side,x:x+side]>0 for m in c.MODELS]
            ims=([rgb]+[overlay(rgb,mask) for mask in masks]) if style=='overlay' else [root_only(rgb,mask) if style=='RGB' else np.repeat((~mask).astype(np.uint8)[:,:,None]*255,3,axis=2) for mask in masks]
            dd.text((7,top+i*stride+10),t,fill='#222222',font=label)
            native=out/('large_local' if style=='overlay' else 'background_removed_large_local')/'native';native.mkdir(parents=True,exist_ok=True)
            for j,im in enumerate(ims):
                sheet.paste(Image.fromarray(im),(left+j*stride,top+i*stride))
                column=('Original_RGB' if j==0 else names[j-1]) if style=='overlay' else names[j]
                Image.fromarray(im).save(native/(t+'_'+column+'_'+style+'.png'))
        filename='Center768_qualitative_four_period_RGB_annotated_comparison.png' if style=='overlay' else 'Center768_qualitative_four_period_background_removed_'+style+'.png'
        sheet.save(out/filename,dpi=(600,600))
    c.js(out/'large_local/protocol.json',dict(source_ROI_xyxy=[x,y,x+side,y+side],native_size=[768,768],periods=c.PERIODS,
        selection='Fixed geometric center before final model evaluation; not selected for appearance or scores',
        evaluation_role='Qualitative only: includes training/validation/test pixels; not part of held-out score computation',
        no_resizing=True,no_color_enhancement=True,mask_dilation=False,root_overlay_alpha=.75))
    print('LARGE_LOCAL_COMPARISONS_DONE',x,y,side,flush=True)

if __name__=='__main__':main()

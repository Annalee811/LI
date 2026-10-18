"""Keep real RGB backgrounds and overlay revised-reference/final-model foreground masks."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import common as c

ROOT_COLOR=np.array([0,158,115],np.float32)
def root_only(rgb,mask):
    out=rgb.copy();out[~mask]=255;return out

def overlay(rgb,mask,alpha=.75):
    out=rgb.copy();out[mask]=np.rint((1-alpha)*rgb[mask]+alpha*ROOT_COLOR).astype(np.uint8);return out

def errors(rgb,gt,pred):
    out=rgb.copy();g=gt==255;v=gt!=128
    for mask,color in [(g&pred&v,[0,158,115]),(~g&pred&v,[204,121,167]),(g&~pred&v,[230,159,0])]:
        out[mask]=np.rint(.15*rgb[mask]+.85*np.array(color)).astype(np.uint8)
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();data=c.Data(a.run)
    out=a.run/'comparison_RGB';out.mkdir(parents=True,exist_ok=True)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    for tile in sorted({r['tile_id'] for r in data.rows if r['split']=='test'}):
        folder=out/'native_local'/tile;folder.mkdir(parents=True,exist_ok=True)
        sheet=Image.new('RGB',(5*256+52,4*256+72),'white');draw=ImageDraw.Draw(sheet)
        root_sheets={style:Image.new('RGB',(4*256+52,4*256+72),'white') for style in ['RGB','black_root']}
        for style,rs in root_sheets.items():
            dd=ImageDraw.Draw(rs)
            for j,label in enumerate(['Reference annotation','U-Net','DeepLabV3+','SegFormer-B0']):dd.text((52+j*256+9,11),label,fill='#222222',font=font)
            dd.text((52,42),tile+' | '+('Original root RGB retained' if style=='RGB' else 'Black roots, white background')+' | Native pixels',fill='#333333',font=small)
        for j,label in enumerate(['Original RGB','Reference annotation','U-Net','DeepLabV3+','SegFormer-B0']):draw.text((52+j*256+9,11),label,fill='#222222',font=font)
        draw.text((52,42),tile+' | Native 256 x 256 px | Green: root; soil/droplets retained | identical coordinates',fill='#333333',font=small)
        for i,t in enumerate(c.PERIODS):
            r=next(r for r in data.rows if r['tile_id']==tile and r['period']==t);rgb,gt=data.load(r);ims=[rgb,overlay(rgb,gt==255)];masks=[gt==255]
            Image.fromarray(rgb).save(folder/(t+'_RGB.png'));Image.fromarray(ims[1]).save(folder/(t+'_Reference_overlay.png'))
            for model in c.MODELS:
                pred=np.array(Image.open(a.run/'native_predictions'/model/'test'/(r['image_id']+'_prediction.png')))>0
                im=overlay(rgb,pred);ims.append(im);masks.append(pred);Image.fromarray(im).save(folder/(t+'_'+model+'_overlay.png'))
                # Also place each model's local RGB outputs inside its independent result package.
                dest=a.run/'native_predictions'/model/'test_RGB_overlays';dest.mkdir(parents=True,exist_ok=True)
                Image.fromarray(im).save(dest/(r['image_id']+'_root_overlay.png'))
                Image.fromarray(errors(rgb,gt,pred)).save(dest/(r['image_id']+'_error_overlay.png'))
            draw.text((6,72+i*256+10),t,fill='#222222',font=font)
            for j,im in enumerate(ims):sheet.paste(Image.fromarray(im),(52+j*256,72+i*256))
            for style,rs in root_sheets.items():
                dd=ImageDraw.Draw(rs);dd.text((6,72+i*256+10),t,fill='#222222',font=font)
                for j,mask in enumerate(masks):
                    im=root_only(rgb,mask) if style=='RGB' else np.repeat((~mask).astype(np.uint8)[:,:,None]*255,3,axis=2)
                    rs.paste(Image.fromarray(im),(52+j*256,72+i*256))
                    model='Reference' if j==0 else c.MODELS[j-1]
                    dest=out/'background_removed_local'/tile;dest.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(im).save(dest/(t+'_'+model+'_'+style+'.png'))
        sheet.save(out/(tile+'_four_period_RGB_annotated_comparison.png'),dpi=(300,300))
        for style,rs in root_sheets.items():rs.save(out/(tile+'_four_period_background_removed_'+style+'.png'),dpi=(300,300))
    dest=out/'native_upper_reference';dest.mkdir(parents=True,exist_ok=True)
    for t in c.PERIODS:
        rgb=np.array(Image.open(a.run/'data/upper'/(t+'_RGB.png')).convert('RGB'));gt=np.array(Image.open(a.run/'data/upper'/(t+'_mask.png')))
        Image.fromarray(overlay(rgb,gt==255)).save(dest/(t+'_Reference_overlay.png'))
        masks=[gt==255]+[np.array(Image.open(a.run/'native_predictions'/m/'upper'/(t+'_root_mask.png')))>0 for m in c.MODELS]
        h,w=gt.shape;largefont=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',36)
        for style in ['RGB','black_root']:
            sheet=Image.new('RGB',(w+22,4*(h+56)+50),'white');dd=ImageDraw.Draw(sheet)
            dd.text((11,7),t+' | Full annotated upper field | Background removed | Native pixels',fill='#222222',font=largefont)
            for j,mask in enumerate(masks):
                model='Reference' if j==0 else c.MODELS[j-1];label='Reference annotation' if j==0 else c.NAMES[model]
                im=root_only(rgb,mask) if style=='RGB' else np.repeat((~mask).astype(np.uint8)[:,:,None]*255,3,axis=2)
                dd.text((11,50+j*(h+56)+6),label,fill='#222222',font=largefont)
                sheet.paste(Image.fromarray(im),(11,50+j*(h+56)+56))
                f=out/'background_removed_upper'/t;f.mkdir(parents=True,exist_ok=True)
                Image.fromarray(im).save(f/(model+'_'+style+'.png'))
            sheet.save(out/(t+'_full_upper_background_removed_'+style+'.png'),dpi=(300,300))
    # Full native atlas: rows=T1..T4, columns=human plus three models.
    w,h=2541,1108;header=68;left=70
    for style in ['RGB','black_root']:
        sheet=Image.new('RGB',(left+4*w,header+4*h),'white');dd=ImageDraw.Draw(sheet)
        for j,label in enumerate(['Reference annotation','U-Net','DeepLabV3+','SegFormer-B0']):dd.text((left+j*w+20,12),label,fill='#222222',font=largefont)
        for i,t in enumerate(c.PERIODS):
            dd.text((6,header+i*h+12),t,fill='#222222',font=largefont)
            for j,model in enumerate(['Reference']+c.MODELS):
                with Image.open(out/'background_removed_upper'/t/(model+'_'+style+'.png')) as im:sheet.paste(im,(left+j*w,header+i*h))
        sheet.save(out/('Four_period_full_upper_background_removed_'+style+'.png'),dpi=(300,300))
    c.js(out/'display_protocol.json',dict(background='Original RGB values unchanged outside painted root pixels',
        root_color_RGB=[0,158,115],alpha=.75,mask_dilation=False,all_dates_same_coordinates=True,
        test_tiles=sorted({r['tile_id'] for r in data.rows if r['split']=='test'}),all_24_test_date_tiles_shown=True,
        native_local_size=[256,256],period_order=c.PERIODS,column_order=['RGB','Reference','U-Net','DeepLabV3+','SegFormer-B0'],
        error_overlay_colors=dict(TP=[0,158,115],FP=[204,121,167],FN=[230,159,0]),
        background_removed_styles=['Original root RGB on white','Black root mask on white'],
        upper_comparisons_native_dimensions=[2541,1108],upper_reference_overlay='Full upper image contains fitting pixels and is qualitative only'))
    print('RGB_COMPARISONS_DONE',flush=True)

if __name__=='__main__':main()

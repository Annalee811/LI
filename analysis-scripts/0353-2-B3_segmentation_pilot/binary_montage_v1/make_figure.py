"""One-page binary root/background atlas; four periods and both spatial scales.

No inference, label revision, color correction, or mask dilation is performed.
The prior final binary model outputs and their corresponding revised reference
are used, rather than the later temporal-classification references.
"""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
P=['T1','T2','T3','T4'];M=['unet','deeplab','segformer'];NAMES=['Revised annotation','U-Net','DeepLabV3+','SegFormer-B0'];GREEN=np.array([0,158,115],np.float32);DARK=(27,34,41);DPI=600

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for data in iter(lambda:f.read(1048576),b''):h.update(data)
 return h.hexdigest()
def font(n,bold=False):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold' if bold else '')+'.ttf',n)
def overlay(rgb,mask):
 out=rgb.copy();out[mask]=np.rint(.25*rgb[mask]+.75*GREEN).astype('uint8');assert np.array_equal(out[~mask],rgb[~mask]);return out
def isolated(rgb,mask):
 out=np.empty_like(rgb);out[:]=DARK;out[mask]=rgb[mask];assert np.array_equal(out[mask],rgb[mask]);return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();run=a.run;view=run/'revised_report';a.output.mkdir(parents=True,exist_ok=True)
 with (view/'data/manifest.csv').open(encoding='utf-8-sig') as f:manifest=list(csv.DictReader(f))
 rgb={t:np.array(Image.open(view/'data/upper'/f'{t}_RGB.png')) for t in P};ref={t:np.array(Image.open(view/'data/upper'/f'{t}_mask.png'))==255 for t in P};pred={m:{t:np.array(Image.open(view/'native_predictions'/m/'upper'/f'{t}_root_mask.png'))>0 for t in P} for m in M}
 for t in P:
  assert rgb[t].shape==(1108,2541,3) and rgb[t].dtype==np.uint8
  for m in M:assert np.array_equal(pred[m][t],np.array(Image.open(run/m/'upper'/f'{t}_root_mask.png'))>0),'Wrong model version'
 candidates=[]
 for tile in sorted({r['tile_id'] for r in manifest if r['split']=='test'}):
  rr=[r for r in manifest if r['tile_id']==tile];x,y=int(rr[0]['x']),int(rr[0]['y']);assert {r['period'] for r in rr}==set(P);counts=[int(ref[t][y:y+256,x:x+256].sum()) for t in P];candidates.append(dict(tile_id=tile,x=x,y=y,root_pixels_by_period=counts,median_root_pixels=float(np.median(counts))))
 # A content-defined detail window, selected without any model quality metric.
 chosen=max(candidates,key=lambda z:(z['median_root_pixels'],z['tile_id']));rx,ry=chosen['x'],chosen['y'];roi=(rx,ry,rx+256,ry+256);sl=np.s_[ry:ry+256,rx:rx+256]
 W,H=9924,7014;canvas=Image.new('RGB',(W,H),'white');draw=ImageDraw.Draw(canvas);TEXT='#1F2937';MUTED='#536273';BORDER='#D7DEE5';GOLD='#E6AB02';mx=150;bw=(W-2*mx-180)//2;top=235;bh=(H-top-170-155)//2;raw_w=820;label_w=190;gap=30;mw=(bw-raw_w-label_w-3*gap)//4;fh=round(mw*1108/2541);row_y=[160,160+fh+45,160+2*fh+135,160+2*fh+135+mw+45];heights=[fh,fh,mw,mw]
 def center(txt,cx,y,f,fill=TEXT):draw.text((int(cx),int(y)),txt,font=f,fill=fill,anchor='mt')
 def place(im,x,y,w,h,frame=True):
  im=Image.fromarray(im) if isinstance(im,np.ndarray) else im;canvas.paste(im.resize((int(w),int(h)),Image.Resampling.LANCZOS),(int(x),int(y)))
  if frame:draw.rectangle((x,y,x+w-1,y+h-1),outline=BORDER,width=2)
 def scalebar(x,y,length,text,dark=False):
  co='white' if dark else '#F9FAFB';draw.rectangle((int(x),int(y),int(x+length),int(y+8)),fill=co);draw.text((int(x+length/2),int(y-13)),text,font=font(39),fill=co,anchor='mb',stroke_width=1,stroke_fill='#26313C')
 center('Root–background segmentation across four periods',W/2,38,font(100,True));center('0353-2-B3  |  registered upper RGB field  |  native root colors retained after background removal',W/2,155,font(52),MUTED)
 for i,t in enumerate(P):
  bx=mx+(i%2)*(bw+180);by=top+(i//2)*(bh+155);draw.rounded_rectangle((bx-25,by-10,bx+bw+25,by+bh),radius=28,outline=BORDER,width=3);draw.text((bx+15,by+15),f'({chr(97+i)})  {t}',font=font(83,True),fill=TEXT)
  method_x=bx+raw_w+label_w
  for j,name in enumerate(NAMES):center(name,method_x+j*(mw+gap)+mw/2,by+88,font(51,True))
  # A separate control strip avoids repeating identical originals in every row.
  xraw=bx+8;rw=raw_w-32;g0=by+row_y[0];g1=by+row_y[1]+fh;z0=by+row_y[2];z1=by+row_y[3]+mw
  draw.rounded_rectangle((xraw-12,g0-10,xraw+rw+12,g1+10),radius=18,fill='#F4F6F8');draw.rounded_rectangle((xraw-12,z0-10,xraw+rw+12,z1+10),radius=18,fill='#F4F6F8')
  raw_h=round(rw*1108/2541);gy=g0+(g1-g0-raw_h)//2;center('Original RGB',xraw+rw/2,gy-72,font(55,True));place(rgb[t],xraw,gy,rw,raw_h)
  box=(xraw+rx/2541*rw,gy+ry/1108*raw_h,xraw+(rx+256)/2541*rw,gy+(ry+256)/1108*raw_h);draw.rectangle(tuple(map(round,box)),outline=GOLD,width=7);scalebar(xraw+26,gy+raw_h-25,rw*500/2541,'500 px');center('Complete upper field',xraw+rw/2,gy+raw_h+28,font(43),MUTED)
  zy=z0+(z1-z0-rw)//2;center('Original RGB · local',xraw+rw/2,zy-72,font(52,True));place(rgb[t][sl],xraw,zy,rw,rw);scalebar(xraw+26,zy+rw-28,rw*50/256,'50 px');center('Same 256 × 256 px ROI',xraw+rw/2,zy+rw+30,font(44),MUTED)
  # Root and soil colors have no temporal meaning in this binary comparison.
  for k,(line1,line2) in enumerate([('Full','Overlay'),('Full','Roots'),('Local','Overlay'),('Local','Roots')]):
   cy=by+row_y[k]+heights[k]/2;center(line1,bx+raw_w+label_w/2-7,cy-65,font(52,True));center(line2,bx+raw_w+label_w/2-7,cy+4,font(48),MUTED)
  masks=[ref[t]]+[pred[m][t] for m in M]
  for j,mask in enumerate(masks):
   x=method_x+j*(mw+gap);ims=[overlay(rgb[t],mask),isolated(rgb[t],mask),overlay(rgb[t][sl],mask[sl]),isolated(rgb[t][sl],mask[sl])]
   for k,im in enumerate(ims):place(im,x,by+row_y[k],mw,heights[k])
 # Minimal shared legend. Color is constant across models and periods.
 y=H-115;draw.rectangle((215,y+6,265,y+50),fill=tuple(GREEN.astype(int)));draw.text((290,y),'Green: root foreground',font=font(52),fill=TEXT);draw.rectangle((2020,y+6,2070,y+50),fill=DARK);draw.text((2095,y),'Dark field: background removed; original root RGB retained',font=font(52),fill=TEXT);draw.rectangle((6680,y+6,6730,y+50),outline=GOLD,width=6);draw.text((6755,y),'Box: common local ROI  |  binary classes only',font=font(48),fill=TEXT)
 path=a.output/'0353-2-B3_binary_segmentation_integrated.tif';canvas.save(path,format='TIFF',compression='tiff_adobe_deflate',dpi=(DPI,DPI));canvas.resize((2481,1754),Image.Resampling.LANCZOS).save(a.output/'layout_preview.png')
 with Image.open(path) as im:assert im.mode=='RGB' and im.n_frames==1 and im.size==(W,H);im.load();assert all(abs(float(v)-600)<.01 for v in im.info['dpi'])
 sources=[]
 for t in P:
  for kind,f in [('RGB',view/'data/upper'/f'{t}_RGB.png'),('revised_binary_reference',view/'data/upper'/f'{t}_mask.png')]+[(m,view/'native_predictions'/m/'upper'/f'{t}_root_mask.png') for m in M]:sources.append(dict(period=t,kind=kind,path=str(f),SHA256=sha(f),root_pixels=int((np.array(Image.open(f))>0).sum()) if kind in M else None))
 protocol=dict(position='0353-2-B3',task='Binary root/background comparison, NOT temporal classification',periods=P,reference='Prior annotation_refinement_v1 full-field U-Net-assisted revised binary reference; not independent manual truth',new_model_training=False,new_label_revision=False,root_mask_dilation=False,overlay_color=[0,158,115],overlay_alpha=.75,isolated_background_RGB=list(DARK),isolated_root_original_RGB_verified=True,background_outside_overlay_masks_unchanged_before_display_scaling=True,native_RGB_size_wh=[2541,1108],figure_size_pixels=[W,H],figure_page_mm=[420,297],DPI=DPI,TIF_mode='RGB',TIF_compression='lossless Adobe Deflate',figure_panels=72,period_block_order='T1 top-left, T2 top-right, T3 bottom-left, T4 bottom-right',method_columns=NAMES,rows=['Full overlay','Full isolated original root RGB','Local overlay','Local isolated original root RGB'],ROI_xyxy=list(roi),ROI_selection='Maximum median revised-reference root-pixel count across the four periods among the six existing internal-test spatial windows; no model score used',ROI_candidates=candidates,chosen_tile=chosen['tile_id'],scale_bars='Pixels only; no unprovided physical calibration invented',whole_field_qualitative=True,whole_fields_include_training_pixels=True,independent_external_test_images=0,sources=sources,output=dict(file=path.name,bytes=path.stat().st_size,SHA256=sha(path)),all_native_masks_unchanged=True)
 (a.output/'figure_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(output=str(path),bytes=path.stat().st_size,SHA256=protocol['output']['SHA256'],ROI=roi,page='A3 landscape, single page, 600 dpi',panels=72),ensure_ascii=False),flush=True)
if __name__=='__main__':main()

"""Single-page temporal atlas of native RGB, reference, and three model outputs."""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
P=['T1','T2','T3','T4'];M=['unet','deeplab','segformer'];NAMES=['Manual / U-Net width','U-Net','DeepLabV3+','SegFormer-B0'];GREEN=np.array([0,158,115],np.float32);DARK=(247,249,251);DPI=600

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for data in iter(lambda:f.read(1048576),b''):h.update(data)
 return h.hexdigest()
def font(n,bold=False):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold' if bold else '')+'.ttf',n)
PALETTE=np.array([[255,255,255],[136,86,167],[0,158,115],[230,171,2],[0,114,178]],dtype='uint8')
PALETTE=np.vstack([PALETTE,np.full((251,3),148,dtype='uint8')])
def overlay(rgb,classes):
 mask=classes>0;out=rgb.copy();out[mask]=np.rint(.25*rgb[mask]+.75*PALETTE[classes[mask]]).astype('uint8');assert np.array_equal(out[~mask],rgb[~mask]);return out
def isolated(rgb,classes):
 mask=classes>0;out=np.empty_like(rgb);out[:]=DARK;out[mask]=PALETTE[classes[mask]];return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();run=a.run;view=run;a.output.mkdir(parents=True,exist_ok=True)
 with (view/'data/spatial_manifest.csv').open(encoding='utf-8-sig') as f:manifest=list(csv.DictReader(f))
 rgb={t:np.array(Image.open(view/'data/upper'/f'{t}_RGB.png')) for t in P};ref={t:np.array(Image.open(view/'data/revised'/f'{t}_classes.png')) for t in P};pred={m:{t:np.array(Image.open(view/m/'four_period_prediction'/f'{t}_temporal_classes.png')) for t in P} for m in M}
 for t in P:
  assert rgb[t].shape==(1108,2541,3)
  for m in M:assert pred[m][t].shape==ref[t].shape
 candidates=[]
 for tile in sorted({r['tile_id'] for r in manifest if r['split']=='test'}):
  rr=[r for r in manifest if r['tile_id']==tile];x,y=int(rr[0]['x']),int(rr[0]['y']);counts=[int((ref[t][y:y+256,x:x+256]>0).sum()) for t in P];candidates.append(dict(tile_id=tile,x=x,y=y,root_pixels_by_period=counts,median_root_pixels=float(np.median(counts))))
 # A content-defined detail window, selected without any model quality metric.
 chosen=max(candidates,key=lambda z:(z['median_root_pixels'],z['tile_id']));rx,ry=chosen['x'],chosen['y'];roi=(rx,ry,rx+256,ry+256);sl=np.s_[ry:ry+256,rx:rx+256]
 W,H=9924,7014;canvas=Image.new('RGB',(W,H),'white');draw=ImageDraw.Draw(canvas);TEXT='#1F2937';MUTED='#536273';BORDER='#D7DEE5';GOLD='#E6AB02';mx=150;bw=(W-2*mx-180)//2;top=235;bh=(H-top-170-155)//2;raw_w=820;label_w=190;gap=30;mw=(bw-raw_w-label_w-3*gap)//4;fh=round(mw*1108/2541);row_y=[160,160+fh+45,160+2*fh+135,160+2*fh+135+mw+45];heights=[fh,fh,mw,mw]
 def center(txt,cx,y,f,fill=TEXT):draw.text((int(cx),int(y)),txt,font=f,fill=fill,anchor='mt')
 def place(im,x,y,w,h,frame=True):
  im=Image.fromarray(im) if isinstance(im,np.ndarray) else im;canvas.paste(im.resize((int(w),int(h)),Image.Resampling.LANCZOS),(int(x),int(y)))
  if frame:draw.rectangle((x,y,x+w-1,y+h-1),outline=BORDER,width=2)
 def scalebar(x,y,length,text,dark=False):
  co='white' if dark else '#F9FAFB';draw.rectangle((int(x),int(y),int(x+length),int(y+8)),fill=co);draw.text((int(x+length/2),int(y-13)),text,font=font(39),fill=co,anchor='mb',stroke_width=1,stroke_fill='#26313C')
 center('Temporal root segmentation across four periods',W/2,38,font(100,True));center('0353-2-B3  |  registered upper RGB field  |  V14 warm starts | manual temporal classes | learned junction routes',W/2,155,font(52),MUTED)
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
 # Shared categorical legend; classes are relative to the immediately previous date.
 y=H-115
 for j,(k,label) in enumerate([(4,'T1 baseline'),(1,'Persistent'),(2,'Newly observed'),(3,'Tip growth')]):
  x=230+j*2000;draw.rectangle((x,y+6,x+50,y+50),fill=tuple(PALETTE[k].tolist()));draw.text((x+78,y),label,font=font(52),fill=TEXT)
 draw.rectangle((8100,y+6,8150,y+50),fill=(148,148,148));draw.text((8180,y),'Untyped reference',font=font(48),fill=TEXT)
 path=a.output/'0353-2-B3_temporal_integrated_v17.tif';canvas.save(path,format='TIFF',compression='tiff_adobe_deflate',dpi=(DPI,DPI));canvas.resize((2481,1754),Image.Resampling.LANCZOS).save(a.output/'layout_preview.png')
 with Image.open(path) as im:assert im.mode=='RGB' and im.n_frames==1 and im.size==(W,H);im.load();assert all(abs(float(v)-600)<.01 for v in im.info['dpi'])
 sources=[]
 for t in P:
  for kind,f in [('RGB',run/'data/upper'/f'{t}_RGB.png'),('manual_classes_U_Net_morphology_reference',run/'data/revised'/f'{t}_classes.png')]+[(m,run/m/'four_period_prediction'/f'{t}_temporal_classes.png') for m in M]:sources.append(dict(period=t,kind=kind,path=str(f),SHA256=sha(f)))
 protocol=dict(version='temporal_manual_continuity_v17',position='0353-2-B3',task='Temporal root classes on the four registered upper RGB fields',reference='Manual temporal semantics with U-Net-adjusted morphology; uncertain added roots untyped',RGB_enhanced=False,native_RGB_size_wh=[2541,1108],figure_size_pixels=[W,H],figure_page_mm=[420,297],DPI=DPI,TIF_mode='RGB',TIF_compression='lossless Adobe Deflate',figure_panels=72,period_block_order='T1 top-left, T2 top-right, T3 bottom-left, T4 bottom-right',method_columns=NAMES,rows=['Full overlay','Full temporal roots, background removed','Local overlay','Local temporal roots, background removed'],ROI_xyxy=list(roi),ROI_selection='Maximum median reference root content across dates among the same six internal test windows; no prediction score used',ROI_candidates=candidates,chosen_tile=chosen['tile_id'],scale_bars='Pixels only',whole_fields_include_training_pixels=True,independent_external_test_images=0,sources=sources,output=dict(file=path.name,bytes=path.stat().st_size,SHA256=sha(path)),root_only_panels='Categorical colors on near-white background; native RGB root cutouts additionally retained in model results')
 (a.output/'figure_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(output=str(path),bytes=path.stat().st_size,SHA256=protocol['output']['SHA256'],ROI=roi,page='A3 landscape, single page, 600 dpi',panels=72),ensure_ascii=False),flush=True)
if __name__=='__main__':main()

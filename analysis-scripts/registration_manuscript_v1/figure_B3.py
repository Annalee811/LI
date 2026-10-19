"""Dedicated full original / upper common region / detail plate for B3."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from figure_series import ROOT,W,H,DPI,INK,TEAL,GOLD,GRAY,MUTED,STAGES,px,text,poly,fitted,corners,project,sha,font
parser=argparse.ArgumentParser();parser.add_argument('--catalog',type=Path,required=True);parser.add_argument('--raw-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();ROOT=args.output;ROOT.mkdir(parents=True,exist_ok=True)
catalog=json.loads(args.catalog.read_text());e=next(e for e in catalog if e['position_id']=='0353-2-B3');m=e['metadata'];reg=m['regions'][0];assert reg['id']=='A' and len(m['regions'])==1
raw=args.raw_root/'0353-2-B3';rawfiles={p.stem:p for p in raw.iterdir() if p.is_file()};assert set(rawfiles)=={'T1','T2','T3','T4'}
box=[1219,232,1854,864];b=m['joined_boxes_T1']['A'];native_box=[box[0]+b[0],box[1]+b[1],box[2]+b[0],box[3]+b[1]]
im=Image.new('RGB',(W,H),'white');draw=ImageDraw.Draw(im)
text(draw,(9,7),'0353-2-B3 | Original context, common observation area and local detail',13,bold=True)
text(draw,(9,15),'Four stages / one upper observation area / unchanged RGB colors',8,fill=MUTED)
left=29;gap=2.2;cw=(297-left-8-3*gap)/4
for i in range(4):
 x=left+i*(cw+gap);draw.rounded_rectangle((px(x),px(23),px(x+cw),px(30)),radius=px(.8),fill=['#F0F3F5','#EDF6F4','#FAF6EC','#F4F0F7'][i]);text(draw,(x+cw/2,26.5),f'T{i+1}  |  '+['Seedling','Heading','Maturity','Senescence'][i],9,STAGES[i],True,'mm')
for label,a,b,c in [('a',59,'Original','+ region'),('b',111,'Common','upper area'),('c',160,'Local','detail')]:
 text(draw,(9,a),label,10,bold=True);text(draw,(9,a+6),b,7.6,bold=True);text(draw,(9,a+10),c,7.6,bold=True)
proof=[]
for i in range(4):
 key=f'T{i+1}';x=left+i*(cw+gap);rp=Path(e['RGB'][key]);op=rawfiles[key]
 original_hash=sha(op);expected=next(r['original_SHA256'] for r in m['area_rows'] if r['period']==key);assert original_hash==expected
 rgb_hash=sha(rp);original=Image.open(op).convert('RGB');assert list(original.size)==m['original_wh'][key]
 off,scale,frame=fitted(im,original,(x,36,cw,57));draw.rectangle(frame,outline=GRAY,width=2)
 q=np.array(m['polygons_native_pixel_centres'][key]['A']);poly(draw,q*scale+off,TEAL,5)
 dq=project(corners(native_box),reg['matrices'][key]);poly(draw,dq*scale+off,GOLD,5)
 common=Image.open(rp).convert('RGB');assert list(common.size)==m['stitched_RGB_wh']
 coff,cscale,cframe=fitted(im,common,(x,98,cw,33));draw.rectangle(cframe,outline=TEAL,width=4);poly(draw,corners(box)*cscale+coff,GOLD,5)
 detail=common.crop(box);doff,dscale,dframe=fitted(im,detail,(x,138,cw,52));draw.rectangle(dframe,outline=GOLD,width=4)
 xx=dframe[0]+px(2);yy=dframe[3]-px(2);length=round(100*dscale[0]);draw.line([(xx,yy),(xx+length,yy)],fill='#26363C',width=16);draw.line([(xx,yy),(xx+length,yy)],fill='white',width=9);draw.text((xx,yy-px(1.2)),'100 px',font=font(7),fill='white',stroke_width=2,stroke_fill='#26363C',anchor='lb')
 proof.append(dict(period=key,original_path=str(op),original_SHA256=original_hash,original_size_wh=list(original.size),common_RGB_path=str(rp),common_RGB_SHA256=rgb_hash,common_RGB_size_wh=list(common.size),projected_detail_native_xy=dq.tolist()))
for yy in [95,134]:draw.line([(px(8),px(yy)),(px(289),px(yy))],fill=GRAY,width=2)
draw.line([(px(9),px(196)),(px(14),px(196))],fill=TEAL,width=5);text(draw,(15,196),'Same observation area in each original frame',7,anchor='lm');draw.line([(px(143),px(196)),(px(148),px(196))],fill=GOLD,width=5);text(draw,(149,196),'Detail window, identical T1-frame coordinates',7,anchor='lm')
text(draw,(9,202),'Common RGB and detail are shown at the saved T1 scale. Bars: T1 pixels; no physical calibration supplied.',6.8,MUTED)
out=ROOT/'Fig_0353_2_B3_original_ROI_region_detail_single_page.tif';im.save(out,format='TIFF',compression='tiff_lzw',dpi=(600,600));preview=im.copy();preview.thumbnail((2400,1700));preview.save(ROOT/(out.stem+'_review.png'))
for r in proof:assert sha(r['original_path'])==r['original_SHA256'] and sha(r['common_RGB_path'])==r['common_RGB_SHA256']
with Image.open(out) as t:assert t.n_frames==1 and t.size==(W,H) and t.info['dpi']==(600.,600.) and t.mode=='RGB'
summary=dict(file=out.name,series='0353-2-B3',positions=['0353-2-B3'],observed_frames=4,pixels_wh=[W,H],page_mm=[297,210],TIFF_frames=1,dpi=600,compression='lossless LZW',bytes=out.stat().st_size,SHA256=sha(out),detail_window_T1_xyxy=native_box,detail_window_stitched_xyxy=box,source_integrity=proof,full_originals_shown=True,common_area_scope='Upper observation area only',colors_unchanged=True,new_registration_fitted=False)
(ROOT/'B3_verification.json').write_text(json.dumps(summary,indent=2));print('B3_THREE_VIEW_TIFF_VERIFIED',out.stat().st_size,flush=True)

"""Three single-page manuscript TIFFs from unchanged RGB and stored geometry."""
import argparse,hashlib,json,math,string,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataset_identity import require_active_position
ROOT=Path('paper_figures');DPI=600;MM=DPI/25.4
W,H=round(297*MM),round(210*MM)
INK='#233C48';TEAL='#18898A';GOLD='#D2A34C';GRAY='#E0E6E8';MUTED='#667780'
STAGES=['#455B69','#218D87','#B58B33','#856B99']
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
# One identical T1-frame detail window per position, inside a single saved band.
# Region, horizontal/vertical center fractions, width fraction of that band.
SPECS={
 '0353-1-B3':('A',.58,.52,.27),
 '0353-2-LB2':('C',.65,.50,.30),
 '0353-2-LB3':('A',.58,.28,.27),
 '0353-3-LB2':('A',.28,.48,.27),
 '0353-3-LB3':('B',.30,.52,.27),
 'A017-1-B2':('B',.45,.54,.26),
 'A017-1-Z2':('B',.57,.50,.29),
 'A017-2-2':('B',.46,.60,.28),
 'A017-2-3':('B',.43,.52,.26),
 'A017-3-B3':('B',.51,.66,.29),
 'A017-3-Z2':('A',.50,.77,.29),
 'Q1-1-B2':('A',.57,.32,.35),
 'Q1-1-Z2':('A',.64,.46,.30),
 'Q1-1-Z3':('A',.77,.68,.27),
 'Q1-2-B3':('C',.40,.52,.30),
 'Q1-2-Z2':('B',.72,.27,.29),
 'Q1-2-Z3':('A',.52,.43,.30)}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def px(v):return round(v*MM)
def font(pt,bold=False):return ImageFont.truetype(BOLD if bold else FONT,round(pt*DPI/72))
def text(draw,xy,s,pt=8,fill=INK,bold=False,anchor='lt'):
 draw.text((px(xy[0]),px(xy[1])),s,font=font(pt,bold),fill=fill,anchor=anchor)
def poly(draw,points,color,width,dashed=False):
 pts=[tuple(float(v) for v in p) for p in points];pts.append(pts[0])
 if not dashed:
  draw.line(pts,fill='#203238',width=width+2,joint='curve');draw.line(pts,fill=color,width=width,joint='curve');return
 for a,b in zip(pts,pts[1:]):
  a,b=np.array(a),np.array(b);dist=float(np.linalg.norm(b-a))
  for s in np.arange(0,dist,px(.95)):
   u=a+(b-a)*(s/dist);v=a+(b-a)*(min(s+px(.60),dist)/dist)
   draw.line([tuple(u),tuple(v)],fill='#203238',width=width+2);draw.line([tuple(u),tuple(v)],fill=color,width=width)
def fitted(canvas,im,rect):
 x,y,w,h=rect;scale=min(px(w)/im.width,px(h)/im.height);ww,hh=max(1,round(im.width*scale)),max(1,round(im.height*scale));x0=px(x)+(px(w)-ww)//2;y0=px(y)+(px(h)-hh)//2
 canvas.paste(im.resize((ww,hh),Image.Resampling.LANCZOS),(x0,y0))
 return np.array([x0,y0]),np.array([ww/im.width,hh/im.height]),(x0,y0,x0+ww-1,y0+hh-1)
def corners(box):
 x0,y0,x1,y1=box;return np.array([[x0,y0],[x1-1,y0],[x1-1,y1-1],[x0,y1-1]],dtype=float)
def project(q,M):
 M=np.asarray(M);v=np.c_[q,np.ones(len(q))]@M.T;return v[:,:2]/v[:,2:]
def details(e):
 meta=e['metadata'];rid,cx,cy,f=SPECS[e['position_id']];b=meta['joined_boxes_T1'][rid];x0,y0,x1,y1=b;bw,bh=x1-x0,y1-y0;s=min(round(bw*f),round(bh*.88));assert s>=100
 x=round(x0+cx*bw-s/2);y=round(y0+cy*bh-s/2);x=min(max(x,x0+2),x1-s-2);y=min(max(y,y0+2),y1-s-2);box=[x,y,x+s,y+s]
 yoff=0
 for r in meta['regions']:
  if r['id']==rid:break
  bb=meta['joined_boxes_T1'][r['id']];yoff+=bb[3]-bb[1]+24
 stitched=[x-x0,y-y0+yoff,x+s-x0,y+s-y0+yoff]
 assert 0<=stitched[0]<stitched[2]<=meta['stitched_RGB_wh'][0] and 0<=stitched[1]<stitched[3]<=meta['stitched_RGB_wh'][1]
 return rid,box,stitched

def render(series,entries):
 im=Image.new('RGB',(W,H),'white');draw=ImageDraw.Draw(im);n=len(entries)
 text(draw,(9,7),series+' | Original frames and corresponding observation regions',13,bold=True)
 text(draw,(9,15),f'{n} positions  /  {sum(len(e["periods"]) for e in entries)} observed frames  /  original RGB + common region + local detail',8,fill=MUTED)
 left=29;right=8;gap=2.2;cw=(297-left-right-3*gap)/4
 for j in range(4):
  x=left+j*(cw+gap)
  draw.rounded_rectangle((px(x),px(23),px(x+cw),px(30)),radius=px(.8),fill=['#F0F3F5','#EDF6F4','#FAF6EC','#F4F0F7'][j])
  text(draw,(x+cw/2,26.5),f'T{j+1}  |  '+['Seedling','Heading','Maturity','Senescence'][j],9,STAGES[j],True,'mm')
  text(draw,(x+cw*.155,32),'Original',6.7,MUTED,anchor='mt');text(draw,(x+cw*.495,32),'Region',6.7,MUTED,anchor='mt');text(draw,(x+cw*.845,32),'Detail',6.7,MUTED,anchor='mt')
 ystart=37;bottom=190;rh=(bottom-ystart)/n;windows=[];source_proof=[]
 for i,e in enumerate(entries):
  pos=require_active_position(e['position_id']);assert pos!='0353-2-B3' and pos.startswith(series+'-')
  y=ystart+i*rh;rid,box,stitched=details(e);meta=e['metadata'];reg=next(r for r in meta['regions'] if r['id']==rid)
  windows.append(dict(position_id=pos,region_id=rid,T1_detail_xyxy=box,stitched_detail_xyxy=stitched))
  text(draw,(9,y+rh*.34),string.ascii_lowercase[i],10,bold=True)
  text(draw,(9,y+rh*.55),pos+(' †' if e['review_required'] else ''),7.0,bold=True)
  image_h=min(rh-3.3,26.5);cy=y+rh*.51;image_y=cy-image_h/2
  for j,t in enumerate(range(1,5)):
   x=left+j*(cw+gap)
   if t not in e['periods']:
    x0,y0,x1,y1=px(x+.3),px(image_y),px(x+cw-.3),px(image_y+image_h)
    draw.rounded_rectangle((x0,y0,x1,y1),radius=px(.8),fill='#F4F6F6',outline=GRAY,width=2)
    text(draw,(x+cw/2,cy),'Not acquired',7.6,MUTED,anchor='mm');continue
   key=f'T{t}';op=Path(e['originals'][key]);rp=Path(e['RGB'][key])
   original_hash=sha(op);rgb_hash=sha(rp)
   original=Image.open(op).convert('RGB');offset,scale,frame=fitted(im,original,(x,image_y,cw*.31,image_h));draw.rectangle(frame,outline=GRAY,width=2)
   delivery_offset=np.array(meta['original_delivery_windows_xyxy'][key][:2])
   for rr in meta['regions']:
    q=np.array(meta['polygons_native_pixel_centres'][key][rr['id']])-delivery_offset
    poly(draw,q*scale+offset,TEAL,4,rr['review_required'])
   q=project(corners(box),reg['matrices'][key])-delivery_offset
   assert q[:,0].min()>=-.1 and q[:,1].min()>=-.1 and q[:,0].max()<original.width+.1 and q[:,1].max()<original.height+.1
   poly(draw,q*scale+offset,GOLD,5)
   common=Image.open(rp).convert('RGB')
   common_offset,common_scale,common_frame=fitted(im,common,(x+cw*.34,image_y,cw*.31,image_h))
   draw.rectangle(common_frame,outline=TEAL,width=3)
   poly(draw,corners(stitched)*common_scale+common_offset,GOLD,4)
   detail=common.crop(stitched)
   dest_offset,dest_scale,detail_frame=fitted(im,detail,(x+cw*.69,image_y,cw*.31,image_h))
   draw.rectangle(detail_frame,outline=GOLD,width=4)
   ax=px(x+cw*.657);bx=px(x+cw*.681);ay=px(cy)
   draw.line([(ax,ay),(bx,ay)],fill=GOLD,width=3);draw.polygon([(bx,ay),(bx-px(.65),ay-px(.5)),(bx-px(.65),ay+px(.5))],fill=GOLD)
   bar=50 if detail.width<350 else 100;bl=round(bar*dest_scale[0]);xx=detail_frame[0]+px(1.0);yy=detail_frame[3]-px(1.0)
   draw.line([(xx,yy),(xx+bl,yy)],fill='#26363C',width=13);draw.line([(xx,yy),(xx+bl,yy)],fill='white',width=7)
   draw.text((xx,yy-px(.9)),f'{bar} px',font=font(5.9),fill='white',stroke_width=2,stroke_fill='#26363C',anchor='lb')
   source_proof.append(dict(position_id=pos,period=key,original_path=str(op),original_SHA256=original_hash,RGB_path=str(rp),RGB_SHA256=rgb_hash,projected_detail_native_xy=q.tolist()))
  if i<n-1:draw.line([(px(8),px(y+rh)),(px(289),px(y+rh))],fill=GRAY,width=2)
 # Thin frame samples keep the legend independent of region/root colors.
 draw.line([(px(9),px(196)),(px(14),px(196))],fill=TEAL,width=4)
 text(draw,(15,196),'Observation region(s); dashed = review required',6.8,anchor='lm')
 draw.line([(px(133),px(196)),(px(138),px(196))],fill=GOLD,width=5)
 text(draw,(139,196),'Same T1-frame detail window across available dates',6.8,anchor='lm')
 footer='Bars: T1 pixels; original colors retained. † At least one observation band requires review.'
 if series=='Q1':footer+=' Q1-1-Z2: T2/T3 provisional.'
 text(draw,(9,202),footer,6.5,MUTED)
 out=ROOT/f'Fig_{series}_original_ROI_detail_single_page.tif';im.save(out,format='TIFF',compression='tiff_lzw',dpi=(DPI,DPI))
 preview=im.copy();preview.thumbnail((2400,1700));preview.save(ROOT/(out.stem+'_review.png'))
 for v in source_proof:
  assert sha(v['original_path'])==v['original_SHA256'];assert sha(v['RGB_path'])==v['RGB_SHA256']
 with Image.open(out) as tif:
  assert tif.n_frames==1 and tif.mode=='RGB' and tif.size==(W,H) and tif.info['dpi']==(600.,600.)
 summary=dict(file=out.name,series=series,positions=[e['position_id'] for e in entries],observed_frames=len(source_proof),pixels_wh=[W,H],page_mm=[297,210],TIFF_frames=1,dpi=600,compression='lossless LZW',bytes=out.stat().st_size,SHA256=sha(out),detail_windows=windows,source_integrity=source_proof,colors_unchanged=True,new_registration_fitted=False)
 (ROOT/(series+'_verification.json')).write_text(json.dumps(summary,indent=2))
 print('READY',series,len(entries),len(source_proof),out.stat().st_size,flush=True)
 return summary

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--catalog',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();ROOT=args.output;ROOT.mkdir(parents=True,exist_ok=True)
 catalog=json.loads(args.catalog.read_text())
 active=[e for e in catalog if e['position_id']!='0353-2-B3'];assert len(active)==17 and {e['position_id'] for e in active}==set(SPECS)
 reports=[render(s,[e for e in active if e['position_id'].startswith(s+'-')]) for s in ['0353','A017','Q1']]
 assert [len(r['positions']) for r in reports]==[5,6,6] and sum(r['observed_frames'] for r in reports)==62
 (ROOT/'publication_manifest.json').write_text(json.dumps(reports,indent=2));print('ALL_THREE_SINGLE_PAGE_TIFFS_VERIFIED',flush=True)

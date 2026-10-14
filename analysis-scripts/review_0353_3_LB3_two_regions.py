#!/usr/bin/env python3
"""Render two independently evidenced, disjoint observation bands on original T1 scale.

Run the documented A/B descriptor exports first. No periodic wrap is assumed and
no nonrigid field is applied. RGB crops are sampled once from original sources.
"""
from pathlib import Path
import argparse, json, hashlib, zipfile
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import B2_region_tools as u

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True,help='Review workspace with input and band_A/band_B evidence')
p.add_argument('--threads',type=int,default=2)
args=p.parse_args(); root=args.root.resolve(); cv2.setNumThreads(args.threads)
pos='0353-3-LB3'; out=root/'reviewed_results'; out.mkdir(exist_ok=True)
paths={int(f.stem[1:]):f for f in (root/'input/raw/V-变化'/pos).glob('T*')}
original={t:np.array(Image.open(f).convert('RGB')) for t,f in paths.items()}
wh={t:a.shape[1::-1] for t,a in original.items()}; h,w=original[1].shape[:2]
metadata={b:json.loads((root/f'band_{b}'/(pos+'_region_localization')/'localization_metadata.json').read_text()) for b in ['A','B']}
assert all(m['period_ids']==[1,2,3,4] for m in metadata.values())
assert not metadata['B']['candidate_region_manual_review']
for t in range(1,5):
 digest=hashlib.sha256(paths[t].read_bytes()).hexdigest()
 assert all(m['input_sha256'][f'T{t}']==digest for m in metadata.values())
boxes={b:metadata[b]['common_T1_rectangle_xyxy'].copy() for b in metadata}
assert boxes['A'][3]<boxes['B'][1], 'Keep separated observation regions separate'
x0=max(box[0] for box in boxes.values()); x1=min(box[2] for box in boxes.values())
for box in boxes.values(): box[0],box[2]=x0,x1
native={b:{t:np.array(metadata[b]['matrices_T1_to_native'][f'T{t}']) for t in range(1,5)} for b in boxes}
crops={}; mask=np.zeros((h,w),np.uint8); polygons={}; cropped_polygons={}
for b,box in boxes.items():
 xx0,y0,xx1,y1=box; g=u.grid(y1-y0,xx1-xx0)+[xx0,y0]
 mask[y0:y1,xx0:xx1]=255; crops[b]={}; polygons[b]={}
 for t in range(1,5):
  coords=u.project(g,native[b][t]); assert u.inside(coords,wh[t]).all()
  s=np.linalg.svd(native[b][t][:2,:2],compute_uv=False); np.testing.assert_allclose(s[0],s[1],rtol=1e-10)
  rgb=u.remap(original[t],coords); crops[b][t]=rgb
  name=f'T{t}_region_{b}_RGB.png'; Image.fromarray(rgb).save(out/name,dpi=(600,600))
  np.testing.assert_array_equal(np.array(Image.open(out/name)),rgb)
  polygons[b][f'T{t}']=u.project(np.array([[xx0,y0],[xx1-1,y0],[xx1-1,y1-1],[xx0,y1-1]],np.float32),native[b][t]).tolist()
 np.testing.assert_array_equal(crops[b][1],original[1][y0:y1,xx0:xx1])
Image.fromarray(mask).save(out/'T1_reviewed_geometric_observation_mask.png')
gap=24; joined={t:np.concatenate([crops['A'][t],np.full((gap,x1-x0,3),255,np.uint8),crops['B'][t]],axis=0) for t in range(1,5)}
for t,rgb in joined.items():Image.fromarray(rgb).save(out/f'T{t}_two_region_RGB.png',dpi=(600,600))
joined_mask=np.full(joined[1].shape[:2],255,np.uint8); joined_mask[len(crops['A'][1]):len(crops['A'][1])+gap]=0
Image.fromarray(joined_mask).save(out/'two_region_geometric_observation_mask.png')
font=lambda n:ImageFont.truetype(u.FONT,n)
phases={1:'苗期',2:'抽穗期',3:'成熟期',4:'枯黄期'}
colours={'A':'#91d7e4','B':'#edbe82'}
panelw=1600; panelh=max(round(a.shape[0]*panelw/a.shape[1]) for a in original.values()); header=152; footer=150
canvas=Image.new('RGB',(4*panelw,header+panelh+footer),'white'); d=ImageDraw.Draw(canvas)
for t in range(1,5):
 im=Image.fromarray(original[t]); im=im.resize((panelw,round(im.height*panelw/im.width)),Image.Resampling.LANCZOS)
 ox=(t-1)*panelw; canvas.paste(im,(ox,header)); d.text((ox+20,15),f'T{t} {phases[t]} | 两段共同观察区域',font=font(37),fill='#24343c')
 d.text((ox+20,78),'A：T3/T4下部根束；B：另一段重合区域',font=font(27),fill='#52626d')
 for b in boxes:
  pp=np.array(polygons[b][f'T{t}'])*panelw/wh[t][0]+[ox,header]; pp=[tuple(q) for q in pp]
  d.line(pp+[pp[0]],fill=colours[b],width=3)
  corner=min(pp,key=lambda q:q[1]); yy=min(max(corner[1]+7,header+7),header+im.height-45)
  d.rectangle((ox+20,yy,ox+69,yy+43),fill='white');d.text((ox+28,yy-3),b,font=font(34),fill='#24343c')
 d.text((ox+20,header+panelh+13),'两段分别整体定位；保留原色与局部变化',font=font(27),fill='#52626d')
 d.text((ox+20,header+panelh+61),'图框为几何可观察范围，非逐根身份确认',font=font(27),fill='#52626d')
 d.text((ox+20,header+panelh+107),'A远期路线仍有差异，需要继续目视核对',font=font(25),fill='#52626d')
canvas.save(out/f'{pos}_reviewed_region_localization.png',dpi=(600,600))
canvas.resize((2600,round(canvas.height*2600/canvas.width)),Image.Resampling.LANCZOS).save(out/f'{pos}_reviewed_region_preview.jpg',quality=97,subsampling=0)

# Original-colour, same-scale sequence with a visibly labelled non-data divider.
pw=1000; rh=round(joined[1].shape[0]*pw/(x1-x0)); head=112
seq=Image.new('RGB',(4*pw,rh+head+75),'white'); d=ImageDraw.Draw(seq)
for t in range(1,5):
 ox=(t-1)*pw; rgb=Image.fromarray(joined[t]); rgb=rgb.resize((pw,rh),Image.Resampling.LANCZOS);seq.paste(rgb,(ox,head))
 d.text((ox+15,10),f'T{t} {phases[t]}',font=font(34),fill='#24343c')
 d.text((ox+15,61),'上：区域A；下：区域B；白带不代表数据',font=font(23),fill='#52626d')
 d.text((ox+15,head+rh+18),'T1尺度；根系和土层变化保持可见',font=font(23),fill='#52626d')
seq.save(out/f'{pos}_two_region_sequence.png',dpi=(600,600))

# Fixed T1 observation windows; never trace or assert identities of individual roots.
windows=[('A','A1：左侧斜向根束',[190,470,940,boxes['A'][3]-5]),
         ('A','A2：中部细根与分叉',[880,350,1590,boxes['A'][3]-5]),
         ('B','B1：右侧根系与土层',[1810,1390,min(2510,x1-1),min(1910,boxes['B'][3]-5)])]
pw=900; head=110; gaprow=85; rows=[]
for b,label,roi in windows:
 xx0,yy0,xx1,yy1=roi;assert boxes[b][0]<=xx0<xx1<=boxes[b][2] and boxes[b][1]<=yy0<yy1<=boxes[b][3]
 hh=round((yy1-yy0)*pw/(xx1-xx0)); row=Image.new('RGB',(4*pw,hh+head+gaprow),'white');d=ImageDraw.Draw(row)
 for t in range(1,5):
  bx,by,_,_=boxes[b]; rgb=crops[b][t][yy0-by:yy1-by,xx0-bx:xx1-bx]
  row.paste(Image.fromarray(rgb).resize((pw,hh),Image.Resampling.LANCZOS),((t-1)*pw,head))
  d.text(((t-1)*pw+15,12),f'{label} | T{t}',font=font(27),fill='#24343c')
  d.text(((t-1)*pw+15,60),'同一T1观察窗；保留细根及土层局部位移',font=font(21),fill='#52626d')
  d.text(((t-1)*pw+15,head+hh+17),'原色局部放大；不据此判定新生或死亡',font=font(22),fill='#52626d')
 rows.append(row)
review=Image.new('RGB',(4*pw,sum(im.height for im in rows)),'white');yy=0
for row in rows:review.paste(row,(0,yy));yy+=row.height
review.save(out/f'{pos}_root_correspondence_review.png',dpi=(600,600))
review.resize((2200,round(review.height*2200/review.width)),Image.Resampling.LANCZOS).save(out/f'{pos}_root_review_preview.jpg',quality=97,subsampling=0)

extra=root/pos/'T1_to_T4/sampled_matches.npz'; neural={}
if extra.exists():
 with np.load(extra) as z:pa=z['source_xy'][z['reliable']]*1600/wh[1][0];pb=z['target_xy'][z['reliable']]*1600/wh[4][0]
 E,k=cv2.estimateAffinePartial2D(pa.astype('float32'),pb.astype('float32'),method=cv2.RANSAC,ransacReprojThreshold=10,maxIters=30000,confidence=.999)
 E=np.vstack([E,[0,0,1]]);neural=dict(pair='T1_to_T4',supported_samples=len(pa),consensus=int(k.sum()),matrix_analysis_1600=E.tolist(),interpretation='Additional full-frame RoMa supports region B; a single whole-frame mode is not evidence that region A is absent.',comparisons={})
 N=np.diag([wh[4][0]/1600]*2+[1])@E@np.diag([1600/wh[1][0]]*2+[1])
 for b,box in boxes.items():
  xx0,y0,xx1,y1=box;g=u.grid(y1-y0,xx1-xx0)[::16,::16]+[xx0,y0]
  error=np.linalg.norm(u.project(g,N)-u.project(g,native[b][4]),axis=-1)*wh[1][0]/wh[4][0]
  neural['comparisons'][b]=dict(median_T1_px=float(np.median(error)),p95_T1_px=float(np.percentile(error,95)))
audit=dict(position=pos,reference='Original T1 pixel centres and scale',original_wh=wh,
 regions_T1_xyxy=boxes,crop_width=x1-x0,crop_heights={b:box[3]-box[1] for b,box in boxes.items()},
 observed_source_area_pct_T1=float(100*np.count_nonzero(mask)/mask.size),stitched_RGB_wh=list(joined[1].shape[1::-1]),separator_pixels=gap,
 unobserved_T1_gap_y=[boxes['A'][3],boxes['B'][1]],polygons_native=polygons,
 band_metadata=metadata,root_observation_windows=windows,additional_neural_check=neural,
 finding='Earlier single-rectangle export omitted a second evidenced, disjoint region. A maps T1/T2 upper-middle root structures to T3/T4 lower frames; B maps T1/T2 lower to T3/T4 upper frames. No physical cylindrical wrap mechanism is assumed.',
 limitations='A direct T1/T4 structural route differs from its consecutive chain by about 6.65 T1 px median; retain this warning. These are region-location crops, not exact point registration, validated root identities, or manual TRE. All area percentages are geometric observation, not reliable-point mask coverage. White divider is not image data and roots must not be tracked across it. Interpolation/DPI creates no captured detail.',
 input_sha256=metadata['A']['input_sha256'],validation=dict(original_hashes_checked=True,T1_crops_pixel_exact=True,all_crops_inside_original=True,only_uniform_similarity=True,RGB_colour_unmodified=True))
(root/'two_region_review_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
primary=[out/f'{pos}_reviewed_region_localization.png',out/f'{pos}_root_correspondence_review.png',out/f'{pos}_two_region_sequence.png',out/'T1_reviewed_geometric_observation_mask.png']+[out/f'T{t}_region_{b}_RGB.png' for b in boxes for t in range(1,5)]
secondary=[out/f'T{t}_two_region_RGB.png' for t in range(1,5)]+[out/'two_region_geometric_observation_mask.png']
for name,files in [('reviewed_regions_and_RGB',primary),('four_stitched_RGB',secondary)]:
 path=root/f'{pos}_{name}.zip'
 with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for f in files:z.write(f,f.name)
 with zipfile.ZipFile(path) as z:assert z.testzip() is None;assert all(n.endswith('.png') for n in z.namelist())
 assert path.stat().st_size<100*1024**2
 print('ZIP',path,path.stat().st_size,flush=True)
print('REVIEW_VALIDATED',json.dumps({k:audit[k] for k in ['regions_T1_xyxy','observed_source_area_pct_T1','stitched_RGB_wh','additional_neural_check']},ensure_ascii=False),flush=True)

from pathlib import Path
import base64,csv,hashlib,json,re,shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFont

import argparse
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/B3_two_groups_review');args=p.parse_args()
OUT=args.output;OUT.mkdir(parents=True,exist_ok=True)
KIT=ROOT/'roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation'
data=json.loads(Path(__file__).with_name('two_windows_visual_traces_v3.json').read_text())
rows=[r for r in csv.DictReader((KIT/'split_manifest.csv').open(encoding='utf-8-sig')) if r['window_id'] in ['B3_A0_C0','B3_A0_C1']]
assert len(rows)==8
for folder in ['originals','masks','attributes','overlays','roots_only','metadata','review_panels']:(OUT/folder).mkdir(exist_ok=True)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
def raster(points,width):
    canvas=Image.new('L',(1792,1792));d=ImageDraw.Draw(canvas)
    d.line([(round(x*4),round(y*4)) for x,y in points],fill=255,width=max(1,round(width*4)),joint='curve')
    # Explicit sampled width, no normal-profile optimization or intensity-driven relocation.
    for x,y in [points[0],points[-1]]:
        rad=width*2;d.ellipse((x*4-rad,y*4-rad,x*4+rad,y*4+rad),fill=255)
    return np.asarray(canvas.resize((448,448),Image.Resampling.LANCZOS))>=110
def runs(a):
    b=a.ravel();idx=np.flatnonzero(np.r_[True,b[1:]!=b[:-1]])
    return np.column_stack([b[idx],np.diff(np.r_[idx,len(b)])]).ravel().tolist()
records=[];audit=[];relations=[]
for row in rows:
    name=row['image_id'];win=row['window_id'];t=int(row['period'])
    b=(KIT/row['image_path']).read_bytes();assert hashlib.sha256(b).hexdigest()==row['image_sha256']
    (OUT/'originals'/(name+'.png')).write_bytes(b)
    im=np.array(Image.open(KIT/row['image_path']).convert('RGB'))
    arr={k:np.zeros((448,448),np.uint8) for k in ['mask','color','temporal','viability','cohort','scene','instance']};arr['mask'][:]=64
    visible=[]
    for r in data['traces']:
        if r['window']!=win:continue
        parts=r['periods'].get(str(t),[])
        if not parts:continue
        width=r['width'].get(str(t),1) if isinstance(r['width'],dict) else r['width']
        own=np.zeros((448,448),bool)
        for part in parts:own|=raster(part,width)
        rid=r['id'] if r['linked'] else r['id']+40*(t-1)
        origin=1 if t==1 else r['origin'] if r['linked'] else 0
        prior=any(r['periods'].get(str(s),[]) for s in range(1,t))
        state=3 if t>1 and prior and r['linked'] else 0
        arr['mask'][own]=255;arr['scene'][own]=1;arr['instance'][own]=rid;arr['cohort'][own]=origin;arr['temporal'][own]=state
        visible.append(rid)
        relations.append(dict(window=win,period=t,root_id=rid,trace_index=r['id'],origin=origin,
            correspondence='visual_provisional' if r['linked'] else 'unresolved_period_specific_id',
            visible_segments=len(parts),width_estimate_pixels=width,death='not_assessed',note=r.get('note','')))
    if win=='B3_A0_C0':
        for ext in data['extensions']:
            if ext['period']==t:
                own=raster(ext['points'],ext['width']);arr['mask'][own]=255;arr['scene'][own]=1;arr['instance'][own]=ext['root_id'];arr['cohort'][own]=1;arr['temporal'][own]=2
        for u in data['uncertain']:
            if u['period']==t:
                own=raster(u['points'],.8)&(arr['mask']!=255);arr['mask'][own]=128;arr['scene'][own]=5
    root=arr['mask']==255
    if t==1:arr['cohort'][root]=1;arr['temporal'][root]=0
    Image.fromarray(arr['mask']).save(OUT/'masks'/(name+'.png'))
    for key,a in arr.items():
        if key!='mask':Image.fromarray(a).save(OUT/'attributes'/(name+'_'+key+'.png'))
    colors=np.zeros_like(im);colors[root]=(37,99,235)
    for v,c in [(1,(22,163,74)),(2,(234,179,8)),(3,(147,51,234))]:colors[root&(arr['temporal']==v)]=c
    ov=im.copy();ov[root]=(im[root]*.30+colors[root]*.70).astype(np.uint8)
    Image.fromarray(ov).save(OUT/'overlays'/(name+'.png'))
    # The RGB remains exact; enhanced viewing is a separate image with no altered source bytes.
    Image.fromarray(np.dstack([colors,np.where(root,255,0).astype(np.uint8)])).save(OUT/'roots_only'/(name+'.png'))
    record=dict(name=name+'.png',image_sha256=row['image_sha256'],reviewed=False,evidence='',annotator='',
        prefill_origin='assistant_visual_period_specific_polyline_traces_no_ridge_or_flow',review_status='awaiting_user_review',
        **{k:runs(a) for k,a in arr.items()});records.append(record)
    meta=dict(image_id=name,window=win,period=t,original_image_sha256=row['image_sha256'],original_pixels_unchanged=True,
        label_origin=record['prefill_origin'],reviewed_entire_image=False,not_independent_human_ground_truth=True,
        root_ids_are_provisional=True,unlabeled_pixels='unreviewed, not verified background',occluded_gaps_not_connected=True,
        birth_death_not_proven=True,root_id_display=sorted(visible))
    (OUT/'metadata'/(name+'.json')).write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    audit.append(dict(image=name,root_pixels=int(root.sum()),root_ids=len(visible),unreviewed_pixels=int((arr['mask']==64).sum())))

project=dict(kind='B3_reviewable_annotations',version=2,encoding='rle_value_count',not_training_ground_truth=True,records=records)
(OUT/'review_project.json').write_text(json.dumps(project,separators=(',',':')))
(OUT/'root_relationships.json').write_text(json.dumps(relations,ensure_ascii=False,indent=2)+'\n')
(OUT/'annotation_audit.json').write_text(json.dumps(audit,indent=2)+'\n')

for win in ['B3_A0_C0','B3_A0_C1']:
    # Side by side original and overlaid image at native size; repeat for each period.
    panel=Image.new('RGB',(1880,1000),'white');d=ImageDraw.Draw(panel)
    for t in range(1,5):
        x=8+470*(t-1);d.text((x,8),f'{win} / T{t}',font=font,fill=(30,40,40))
        panel.paste(Image.open(OUT/'originals'/f'{win}_T{t}.png'),(x,45));panel.paste(Image.open(OUT/'overlays'/f'{win}_T{t}.png'),(x,525))
        d.text((x,498),'Original above / editable trace below',font=small,fill=(50,50,50))
    panel.save(OUT/'review_panels'/(win+'_original_and_roots.png'))
print('EIGHT_IMAGES_BUILT',audit)

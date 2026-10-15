#!/usr/bin/env python3
"""Crop RGB and all label layers only after whole-image annotation; spatial group split."""
from pathlib import Path
import argparse,csv,hashlib,json,random
import numpy as np
from PIL import Image

KEYS=['mask','color','temporal','viability','cohort','scene','instance']
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def decode(r,n):
    a=np.asarray(r)
    if a.ndim!=1 or len(a)%2:raise ValueError('Bad RLE')
    a=a.reshape(-1,2)
    if not np.issubdtype(a.dtype,np.integer) or np.any(a[:,0]<0) or np.any(a[:,0]>255) or np.any(a[:,1]<=0) or a[:,1].sum()!=n:raise ValueError('Bad RLE values/length')
    return np.repeat(a[:,0],a[:,1]).astype('uint8')
def export(args):
    project=json.loads(args.project.read_text(encoding='utf-8-sig'))
    if project.get('kind')!='B3_full_canvas_project' or project.get('version')!=1 or len(project.get('sources',[]))!=4:raise ValueError('Need four full-image project')
    reviewed=all(s.get('reviewed') is True and str(s.get('annotator','')).strip() for s in project['sources'])
    if not reviewed and not args.allow_draft:raise ValueError('四期完整图尚未全部人工复核。先完成标注；草稿导出须用 --allow-draft，不能用于训练或评价。')
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Output must be empty; previous results will not be overwritten')
    with args.mapping.open(encoding='utf-8-sig') as f:
        rows=json.load(f)
    if len(rows)!=60:raise ValueError('Need audited 60-window coordinate map')
    arrays={};rgb={};source_records={}
    for s in project['sources']:
        name=s['name'];p=args.images/name
        if digest(p)!=s['image_sha256']:raise ValueError('Source RGB SHA256 mismatch '+name)
        im=np.asarray(Image.open(p).convert('RGB'))
        if im.shape!=(1867,2541,3) or [s['width'],s['height']]!=[2541,1867]:raise ValueError('Wrong full-image dimensions')
        aa={k:decode(s[k],2541*1867).reshape(1867,2541) for k in KEYS}
        if not set(np.unique(aa['mask'])).issubset({0,64,128,255}):raise ValueError('Wrong label codes')
        if reviewed:aa['mask'][aa['mask']==64]=0
        arrays[name]=aa;rgb[name]=im;source_records[name]=s
    cols=list(range(5));random.Random(args.seed).shuffle(cols)
    assignment={c:'train' if c in cols[:3] else 'val' if c==cols[3] else 'test' for c in cols}
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    for d in ['images','labels' if reviewed else 'draft_labels','attributes']:(out/d).mkdir(exist_ok=True)
    manifest=[]
    for row in rows:
        name=row['full_name'];x=int(row['full_x']);y=int(row['full_y']);ident=row['image_id'];region=row['band']
        if not (0<=x and x+448<=2541 and ((region=='A' and 0<=y and y+448<=1108) or (region=='B' and 1132<=y and y+448<=1867))):raise ValueError('Crop crosses separator or bounds')
        file=out/'images'/(ident+'.png');patch=rgb[name][y:y+448,x:x+448];Image.fromarray(patch).save(file,dpi=(600,600))
        assert np.array_equal(np.asarray(Image.open(file)),patch)
        aa={k:a[y:y+448,x:x+448] for k,a in arrays[name].items()}
        labels=out/('labels' if reviewed else 'draft_labels');mask=labels/(ident+'.png');Image.fromarray(aa['mask']).save(mask)
        for k in KEYS[1:]:Image.fromarray(aa[k]).save(out/'attributes'/(ident+'_'+k+'.png'))
        s=source_records[name];review=dict(image_id=ident,image_sha256=digest(file),mask_sha256=digest(mask),annotator=s.get('annotator',''),reviewed_entire_image=reviewed,label_origin='preserved_existing_brush_labels_with_full_image_review' if reviewed else 'unreviewed_draft_not_for_training',source_full_image=name,source_image_sha256=s['image_sha256'],full_xyxy=[x,y,x+448,y+448],unknown64_converted_to_background_only_after_explicit_whole_image_review=reviewed)
        (labels/(ident+'.json')).write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
        c=int(row['spatial_group'][1:]);split=assignment[c] if reviewed else ''
        manifest.append(dict(image_id=ident,image_path='images/'+file.name,image_sha256=digest(file),source_crop=name,source_crop_sha256=s['image_sha256'],period=row['period'],band=region,window_id=row['window_id'],spatial_group=row['spatial_group'],split=split,T1_x=row['T1_x'],T1_y=row['T1_y'],crop_x=x,crop_y=y,width=448,height=448))
    with (out/'split_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(manifest[0]),lineterminator='\n');w.writeheader();w.writerows(manifest)
    summary=dict(source_full_images=4,output_patches=60,ready_for_training=reviewed,seed=args.seed,counts={s:sum(r['split']==s for r in manifest) for s in ['train','val','test']},same_spatial_group_all_periods_same_split=True,guard_T1_pixels=64,source_project_sha256=digest(args.project),limitation='One microtube position; spatial holdout is not independent-site testing. No model trained or accuracy inferred.')
    (out/'export_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'original_project_backup.json').write_bytes(args.project.read_bytes())
    print(json.dumps(summary))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--images',type=Path,required=True);p.add_argument('--mapping',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=20261015);p.add_argument('--allow-draft',action='store_true');export(p.parse_args())

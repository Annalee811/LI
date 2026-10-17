#!/usr/bin/env python3
"""Prepare larger native tiles; preserve every original human annotation layer."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import common as c

def main():
    p=argparse.ArgumentParser();p.add_argument('--progress',type=Path,required=True);p.add_argument('--originals',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);shutil.copyfile(a.progress,a.output/'original_human_progress.json')
    upper=a.output/'dataset/upper';upper.mkdir(parents=True,exist_ok=True);rows=[];audit=[];fingerprints={}
    source=json.loads(a.progress.read_text())
    for item in source['images']:
        t=item['name'][:2];assert t in c.PERIODS
        original=a.originals/item['name'];assert c.sha(original)==item['image_sha256']
        rgb=np.array(Image.open(original).convert('RGB'));assert rgb.shape==(1867,2541,3) and (rgb[1108:1132]==255).all()
        rgb=rgb[:1108].copy();layers={k:np.repeat(np.asarray(v[::2],np.uint8),v[1::2]).reshape(1867,2541)[:1108].copy() for k,v in item['layers'].items()}
        unknown=int((layers['mask']==64).sum());layers['mask'][layers['mask']==64]=0
        arrays={'RGB':rgb,**layers}
        for k,value in arrays.items():
            path=upper/(t+'_'+k+'.png');Image.fromarray(value).save(path)
            assert np.array_equal(np.array(Image.open(path)),value);fingerprints[str(path.relative_to(a.output))]=c.sha(path)
        for col,x0 in enumerate(c.COLS):
            split=['train','train','train','val','test'][col]
            for i,(dx,y) in enumerate(c.COORDS):
                x=x0+dx;stem=t+'_C'+str(col)+'_'+str(i).zfill(2);paths={}
                for k in arrays:
                    path=a.output/'dataset/patches'/split/k/(stem+'.png');path.parent.mkdir(parents=True,exist_ok=True)
                    tile=arrays[k][y:y+c.SIDE,x:x+c.SIDE];assert tile.shape[:2]==(c.SIDE,c.SIDE)
                    Image.fromarray(tile).save(path);assert np.array_equal(np.array(Image.open(path)),tile)
                    paths[k+'_path']=str(path.relative_to(a.output));paths[k+'_sha256']=c.sha(path)
                rows.append(dict(image_id=stem,window_id='C'+str(col)+'_'+str(i),period=t,column=col,split=split,x=x,y=y,width=256,height=256,**paths))
        audit.append(dict(period=t,original_rgb_sha256=c.sha(original),root_pixels=int((layers['mask']==255).sum()),
            unknown64_accepted_as_background_upper_only=unknown,review_checkbox=item['reviewed'],
            authority='User explicitly completed four upper regions; lower region excluded',original_labels_modified=False))
    assert len(rows)==200
    for r in rows:
        peers=[q for q in rows if q['window_id']==r['window_id']]
        assert len(peers)==4 and len({q['split'] for q in peers})==1
        for q in rows:
            if r['split']!=q['split']:
                dx=max(r['x']-(q['x']+256),q['x']-(r['x']+256),0)
                dy=max(r['y']-(q['y']+256),q['y']-(r['y']+256),0)
                assert max(dx,dy)>=64
    c.csvout(a.output/'dataset/manifest.csv',rows)
    fingerprints['dataset/manifest.csv']=c.sha(a.output/'dataset/manifest.csv')
    summary=dict(task='Standalone root/background segmentation; no temporal loss, previous-period inputs, or death class',
        original_images=4,positions=1,upper_roi_xyxy=[0,0,2541,1108],patch_size=[256,256],
        image_patches={s:sum(r['split']==s for r in rows) for s in ['train','val','test']},
        physical_windows={s:len({r['window_id'] for r in rows if r['split']==s}) for s in ['train','val','test']},
        all_four_periods_same_spatial_split=True,cross_split_guard_pixels=64,independent_external_test_images=0,
        labels_dilated=False,RGB_resized=False,RGB_color_changed=False,
        sampling='48 native 256x256 crops per epoch: half fixed-grid, half root-centred, train columns only; augmentation is not independent data',
        test_scope='Previously used same-position C4 spatial holdout, not a fresh independent external test',
        foreground='Union of all human-labelled roots regardless of temporal/color attribute; death not trained',
        background='One non-root class; soil, droplets and debris do not have separate manual subtype labels',
        label_scope='Upper ROI unpainted64 accepted as background under user completeness declaration; fine unmarked roots remain a limitation',
        time_stage='Not started unless validation AND spatial holdout root Dice >=0.95, Precision >=0.90, Recall >=0.95',
        length_scope='Human mask-derived skeleton length in pixels; no physical scale or independent manual length measurement')
    c.js(a.output/'dataset/summary.json',summary);c.js(a.output/'annotation_audit.json',audit)
    c.js(a.output/'dataset/provenance.json',dict(source_progress_sha256=c.sha(a.progress),files=fingerprints,
        fingerprint=c.sha(a.output/'dataset/manifest.csv')))
    image=Image.open(upper/'T1_RGB.png').convert('RGB');draw=ImageDraw.Draw(image)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',28)
    for col,x in enumerate(c.COLS):
        color=['#28875a']*3+['#cf9f00','#8554ad'];draw.rectangle((x,0,x+447,1107),outline=color[col],width=6)
        draw.text((x+8,12),'C'+str(col)+' '+['train','train','train','val','holdout'][col],font=font,fill=color[col])
    image.save(a.output/'spatial_split.png',dpi=(300,300))
    c.js(a.output/'verification.json',dict(lossless_native_tiles=True,all_four_periods_grouped=True,guard_pixels_verified=64,
        original_RGB_hashes_matched=True,all_original_layers_preserved=True,n_tiles=len(rows),passed=True))
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()

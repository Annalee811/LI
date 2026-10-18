"""One fixed random allocation of 36 native tiles; no larger C-regions."""
import argparse, itertools, shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import common as c

def main():
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True); p.add_argument('--run',type=Path,required=True); a=p.parse_args()
    out=a.run/'data'; (out/'upper').mkdir(parents=True,exist_ok=True)
    assert c.sha(a.source/'original_human_progress.json')=='27943e5f9d087bf718668bba0a39058dc9a4871de6c75c5ccc27febad4c06a09'
    shutil.copyfile(a.source/'original_human_progress.json',out/'original_human_progress.json')
    arrays={}; sources={}
    for t in c.PERIODS:
        for kind in ['RGB','mask']:
            src=a.source/'upper'/(t+'_'+kind+'.png'); dst=out/'upper'/src.name
            shutil.copyfile(src,dst); sources[str(dst.relative_to(a.run))]=c.sha(dst)
        arrays[t]=(np.array(Image.open(out/'upper'/(t+'_RGB.png')).convert('RGB')),np.array(Image.open(out/'upper'/(t+'_mask.png')).convert('L')))
        assert arrays[t][0].shape==(1108,2541,3) and arrays[t][1].shape==(1108,2541)
        assert set(np.unique(arrays[t][1])).issubset({0,128,255})
    # Geometry only, specified before predictions. No root-rich/easy-tile filtering.
    xs=np.rint(np.linspace(0,2541-c.SIDE,9)).astype(int); ys=np.rint(np.linspace(0,1108-c.SIDE,4)).astype(int)
    tiles=[dict(tile_id='P%02d'%(i+1),x=int(x),y=int(y),width=c.SIDE,height=c.SIDE) for i,(y,x) in enumerate(itertools.product(ys,xs))]
    order=np.random.default_rng(c.SEED).permutation(len(tiles)).tolist()
    assignments={idx:('train' if rank<25 else 'val' if rank<30 else 'test') for rank,idx in enumerate(order)}
    for i,tile in enumerate(tiles): tile['split']=assignments[i]
    for r,q in itertools.combinations(tiles,2):
        assert r['x']+c.SIDE<=q['x'] or q['x']+c.SIDE<=r['x'] or r['y']+c.SIDE<=q['y'] or q['y']+c.SIDE<=r['y']
    manifest=[]
    for tile in tiles:
        for t in c.PERIODS:
            x,y=tile['x'],tile['y']; rgb,gt=arrays[t]; key=tile['tile_id']+'_'+t; paths={}
            for kind,arr in [('RGB',rgb),('mask',gt)]:
                f=out/'patches'/tile['split']/kind/(key+'.png'); f.parent.mkdir(parents=True,exist_ok=True)
                native=arr[y:y+c.SIDE,x:x+c.SIDE]; Image.fromarray(native).save(f)
                assert np.array_equal(np.array(Image.open(f)),native); paths[kind+'_path']=str(f.relative_to(a.run)); paths[kind+'_sha256']=c.sha(f)
            g=gt[y:y+c.SIDE,x:x+c.SIDE]
            manifest.append(dict(image_id=key,period=t,**tile,**paths,root_pixels=int((g==255).sum()),ignored_pixels=int((g==128).sum())))
    for tile in tiles:
        rr=[r for r in manifest if r['tile_id']==tile['tile_id']]
        assert len(rr)==4 and len({r['split'] for r in rr})==1
    c.csvout(out/'tile_allocation.csv',tiles); c.csvout(out/'manifest.csv',manifest)
    composition=[]
    for split in ['train','val','test']:
        rr=[r for r in manifest if r['split']==split]; roots=sum(r['root_pixels'] for r in rr); valid=sum(256*256-r['ignored_pixels'] for r in rr)
        composition.append(dict(split=split,RGB_patch_count=len(rr),physical_tile_positions=len({r['tile_id'] for r in rr}),
            root_pixels=roots,valid_pixels=valid,root_fraction=roots/valid,rootless_patches=sum(r['root_pixels']==0 for r in rr)))
    c.csvout(out/'split_composition.csv',composition)
    summary=dict(source_images=4,positions=1,source_size_wh=[2541,1108],tile_size_wh=[256,256],seed=c.SEED,
        algorithm='numpy.default_rng(seed).permutation(36); first25 train, next5 validation, last6 test; no seed search',
        spatial_tiles={s:sum(r['split']==s for r in tiles) for s in ['train','val','test']},
        RGB_images={s:sum(r['split']==s for r in manifest) for s in ['train','val','test']},
        spatial_tile_coverage_ratio=36*256*256/(2541*1108),min_horizontal_gap_px=int(np.diff(xs).min()-256),min_vertical_gap_px=int(np.diff(ys).min()-256),
        root_density_filtering=False,difficult_tile_filtering=False,all_periods_same_tile_split=True,cross_split_pixel_overlap=0,
        larger_spatial_regions_used_for_splitting=False,resize=False,color_change=False,mask_edit=False,augmentation_after_split_only=True,
        task='root/background only; temporal labels and death not trained',independent_external_test_images=0,
        evaluation='Same-position random tile internal test. Nearby tiles and dates are correlated; no independent tube generalization claim.',
        old_B3_trained_weights_allowed_for_initialization=False,source_annotation_SHA256=c.sha(out/'original_human_progress.json'),source_files=sources,
        manifest_SHA256=c.sha(out/'manifest.csv'),labels='Human root strokes as supplied; unpainted accepted upper pixels are background;128 ignored',
        limitations=['Human strokes may incompletely cover anatomical root widths.', 'Gaps prevent pixel overlap but do not ensure root/biological independence.',
                     'Only annotated upper area is fitted/evaluated; lower area excluded.', 'No independent manual root-length measurements or physical scale.'])
    c.js(out/'summary.json',summary)
    c.js(out/'verification.json',dict(passed=True,lossless_RGB_and_mask_checks=len(manifest)*2,spatial_tiles=36,
        paired_period_checks=36,pairwise_nonoverlap_checks=36*35//2,labels_changed=False,initialization_policy='No previously B3-trained weights'))
    rgb=Image.fromarray(arrays['T1'][0]); draw=ImageDraw.Draw(rgb); font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24)
    palette={'train':'#0072B2','val':'#E69F00','test':'#009E73'}
    for r in tiles:
        x,y=r['x'],r['y']; color=palette[r['split']]; draw.rectangle([x,y,x+255,y+255],outline=color,width=5)
        draw.rectangle([x+4,y+4,x+224,y+36],fill='white'); draw.text([x+8,y+6],r['tile_id']+' '+r['split'],fill=color,font=font)
    rgb.save(out/'random_tile_allocation.png',dpi=(300,300))
    print('DATA_PREPARED',summary['spatial_tiles'],summary['RGB_images'],flush=True)

if __name__=='__main__': main()

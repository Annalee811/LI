"""Verify actual source identity, RGB sampling, area scaling, masks and confidence."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from PIL import Image
from shapely.geometry import Polygon,box
from shapely.ops import unary_union
from render_results import grid,proj,join,sha,corners

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args();cv2.setNumThreads(1)
    # Catch using a T1-pixel crop area against a differently scaled native canvas.
    p0=Polygon(corners([0,0,100,50],True));p1=Polygon(proj(corners([0,0,100,50],True),np.diag([2,2,1])))
    assert np.isclose(p1.area,4*p0.area) and np.isclose(unary_union([p0,p0]).area,p0.area)
    records=[];root=a.work/'deliverable'
    for folder in sorted(root.iterdir()):
        mf=folder/'geometry_metadata.json'
        if not mf.exists():continue
        meta=json.loads(mf.read_text());pos=meta['position'];regions=meta['regions'];ids=[r['id'] for r in regions];boxes=meta['joined_boxes_T1'];width=(meta['stitched_RGB_wh'] or [0])[0]
        for row in meta['area_rows']:
            t=int(row['period'][1:]);raw=next((a.work/'inputs/raw'/pos).glob('T'+str(t)+'.*'));assert sha(raw)==row['original_SHA256'];rgb=np.array(Image.open(raw).convert('RGB'))
            assert rgb.shape[1::-1]==(row['original_width'],row['original_height'])
            if row['common_area_native_px2'] is not None:
                assert 0<=row['common_area_native_px2']<=row['original_area_px2']+1e-4
                assert np.isclose(row['common_area_percent'],100*row['common_area_native_px2']/row['original_area_px2'])
            if pos!='0353-2-B3':assert sha(folder/'originals'/raw.name)==sha(raw)
            else:
                x0,y0,x1,y1=meta['original_delivery_windows_xyxy'][f'T{t}'];assert np.array_equal(np.array(Image.open(folder/'originals'/f'T{t}_original_upper_region_RGB.png')),rgb[y0:y1,x0:x1])
            parts={};polys=[]
            for r in regions:
                k=r['id'];H=np.array(r['matrices'][f'T{t}']);sv=np.linalg.svd(H[:2,:2],compute_uv=False);assert np.isclose(sv[0],sv[1],rtol=1e-8);q=proj(grid(boxes[k]),H)
                if t==1:x0,y0,x1,y1=boxes[k];expected=rgb[y0:y1,x0:x1]
                else:expected=cv2.remap(rgb,q[...,0],q[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                actual=np.array(Image.open(folder/'RGB'/f'T{t}_region_{k}_RGB.png'));assert np.array_equal(expected,actual),(pos,t,k);parts[k]=actual;polys.append(Polygon(proj(corners(boxes[k],True),H)))
            if regions:
                assert np.array_equal(join(parts,ids,width),np.array(Image.open(folder/'RGB'/f'T{t}_independent_stitched_RGB.png')))
                exact=unary_union(polys).intersection(box(-.5,-.5,rgb.shape[1]-.5,rgb.shape[0]-.5)).area;assert np.isclose(exact,row['common_area_native_px2'])
        if regions:
            geom=np.array(Image.open(folder/'masks/stitched_geometric_overlap_mask.png'))>0;assert int(geom.sum())==sum((b[2]-b[0])*(b[3]-b[1]) for b in boxes.values())
            summary=json.loads((folder/'confidence/confidence_summary.json').read_text())
            for r in summary['rows']:
                t=r['target_period'];stem=f'T1_to_{t}'
                with np.load(folder/'confidence'/(stem+'_cumulative_confidence_float32.npz')) as z:c=z['confidence'];bn=z['bottleneck_confidence']
                encoded=np.array(Image.open(folder/'confidence'/(stem+'_cumulative_confidence_16bit.png')));assert c.shape==geom.shape and np.isfinite(c).all() and c.min()>=0 and c.max()<=1
                assert np.max(np.abs(encoded.astype('float64')/65535-c))<=.5/65535+1e-7;assert np.all(c<=bn+1e-6)
                valid=np.array(Image.open(folder/'masks'/(stem+'_effective_overlap_mask.png')))>0;assert not (valid&~geom).any();assert valid.sum()==r['effective_overlap_pixels_T1'];assert geom.sum()==r['geometric_common_pixels_T1']
                low=np.array(Image.open(folder/'masks'/(stem+'_low_confidence_mask.png')))>0;assert np.array_equal(low,(c<.5)&geom)
        records.append(dict(position=pos,images=len(meta['periods']),regions=len(regions),RGB_source_sampling_exact=True,area_jacobian_and_union_verified=True,confidence_encoding_and_mask_counts_verified=bool(regions)))
        print('VERIFIED',pos,len(regions),flush=True)
    assert len(records)==19 and sum(r['images'] for r in records)==69
    (root/'verification.json').write_text(json.dumps(dict(passed=True,positions=records,original_images=69,synthetic_native_scale_and_union_checks=True,manual_landmark_accuracy_claimed=False),indent=2))
    print('PASS all actual RGBs, original bytes, area unions, masks and confidence')
if __name__=='__main__':main()

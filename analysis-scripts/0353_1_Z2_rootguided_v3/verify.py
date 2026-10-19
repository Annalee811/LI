"""Functional checks for exact RGB sampling, original hashes, fields and uncertainty."""
import argparse,hashlib,json,sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image
support=Path(__file__).parent/'support'
if not support.exists():support=Path(__file__).parent.parent/'all_series_localization_v2'
sys.path.insert(0,str(support));from render_results import grid,proj

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args();r=a.work;pos='0353-1-Z2';out=r/'deliverable'/pos;m=json.loads((out/'geometry_metadata.json').read_text());summary=json.loads((out/'evaluation/delivery_summary.json').read_text());box=m['regions'][0]['box'];q=grid(box);x0,y0,x1,y1=box;area=(x1-x0)*(y1-y0);hashes={}
 for t in [1,2,3,4]:
  p=next((out/'originals').glob(f'T{t}.*'));h=sha(p);assert h==m['input_sha256'][f'T{t}'];hashes[f'T{t}']=h;im=np.array(Image.open(p).convert('RGB'));H=np.array(m['regions'][0]['matrices'][f'T{t}']);s=np.linalg.svd(H[:2,:2],compute_uv=False);np.testing.assert_allclose(s[0],s[1],rtol=1e-10);target=proj(q,H);assert target[...,0].min()>=0 and target[...,1].min()>=0 and target[...,0].max()<=im.shape[1]-1 and target[...,1].max()<=im.shape[0]-1
  expected=im[y0:y1,x0:x1] if t==1 else cv2.remap(im,target[...,0],target[...,1],cv2.INTER_LINEAR);np.testing.assert_array_equal(np.array(Image.open(out/f'RGB/T{t}_independent_stitched_RGB.png')),expected)
  row=next(v for v in m['area_rows'] if v['period']==f'T{t}');np.testing.assert_allclose(row['common_area_native_px2'],area*np.linalg.det(H[:2,:2]),rtol=1e-6);np.testing.assert_allclose(row['common_area_percent'],100*row['common_area_native_px2']/(im.shape[0]*im.shape[1]),rtol=1e-12)
 previous=np.ones((y1-y0,x1-x0),np.float32)
 for t in [2,3,4]:
  cf=np.load(out/f'confidence/T1_to_T{t}_cumulative_confidence_float32.npz')['confidence'];assert cf.dtype==np.float32 and cf.shape==previous.shape and np.isfinite(cf).all() and cf.min()>=0 and cf.max()<=1;assert (cf<=previous+1e-7).all();previous=cf;np.testing.assert_array_equal(np.array(Image.open(out/f'confidence/T1_to_T{t}_cumulative_confidence_16bit.png')),np.uint16(np.rint(cf*65535)));np.testing.assert_array_equal(np.array(Image.open(out/f'masks/T1_to_T{t}_low_confidence_mask.png'))>0,cf<.5)
 for aa,bb in [(1,2),(2,3),(3,4)]:
  e=json.loads((r/f'posed_roma_evidence/{pos}/T{aa}_to_T{bb}/completed.json').read_text());assert e['input_sha256']==[sha(out/f'RGB/T{aa}_independent_stitched_RGB.png'),sha(out/f'RGB/T{bb}_independent_stitched_RGB.png')];z=np.load(out/f'fields/T{aa}_to_T{bb}_dense_fields_float32.npz');assert np.isfinite(z['displacement_common_T1_xy']).all();assert z['confidence'].shape==previous.shape
 assert summary['point_supported_adjacent_pairs']==2 and summary['total_adjacent_pairs']==3 and summary['review_required'] is True;assert summary['manual_validated_pair_fraction'] is None
 proof=dict(original_hashes_verified=hashes,all_four_RGB_exact_saved_sampling=True,T1_native_crop_pixel_exact=True,all_transforms_uniform_similarity=True,all_RGB_original_colors=True,no_dense_RGB_deformation=True,area_matches_native_similarity_determinants=True,cumulative_confidence_monotone_and_finite=True,confidence_16bit_and_low_masks_match_float32=True,posed_inference_input_hashes_verified=True,coarse_T2_T3_not_counted_as_point_success=True,manual_TRE_not_fabricated=True,position=pos)
 (out/'evaluation/verification.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof,indent=2),flush=True)
if __name__=='__main__':main()

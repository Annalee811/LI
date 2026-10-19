"""Verify numeric geometry, native RGB preservation, and explicit missing T1 link."""
import argparse,hashlib,json
from pathlib import Path
import cv2,numpy as np
from PIL import Image

def main():
 p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args();out=a.work/'deliverable';m=json.loads((out/'evaluation/recheck_audit.json').read_text());H=np.array(m['T2_to_T3_root_similarity']['matrix_native']);x0,y0,x1,y1=m['T2_reference_box'];ims={t:np.array(Image.open(next((out/'originals').glob(f'T{t}.*'))).convert('RGB')) for t in [1,2,3]}
 for t in ims:assert hashlib.sha256(next((out/'originals').glob(f'T{t}.*')).read_bytes()).hexdigest()==m['input_sha256'][f'T{t}']
 np.testing.assert_allclose(np.linalg.svd(H[:2,:2],compute_uv=False)[0],np.linalg.svd(H[:2,:2],compute_uv=False)[1],rtol=1e-12)
 y,x=np.mgrid[y0:y1,x0:x1].astype('float32');q=np.stack([x,y],-1);target=(q@H[:2,:2].T+H[:2,2]).astype('float32');assert target.min()>=0;assert target[...,0].max()<=ims[3].shape[1]-1;assert target[...,1].max()<=ims[3].shape[0]-1
 np.testing.assert_array_equal(np.array(Image.open(out/'RGB/T2_confirmed_common_RGB.png')),ims[2][y0:y1,x0:x1]);np.testing.assert_array_equal(np.array(Image.open(out/'RGB/T3_aligned_to_T2_common_RGB.png')),cv2.remap(ims[3],target[...,0],target[...,1],cv2.INTER_LINEAR))
 cf=np.load(out/'confidence/T2_to_T3_confidence_float32.npz')['confidence'];assert cf.dtype==np.float32 and cf.shape==(y1-y0,x1-x0);assert np.isfinite(cf).all() and cf.min()>=0 and cf.max()<=1;np.testing.assert_array_equal(np.array(Image.open(out/'confidence/T2_to_T3_confidence_16bit.png')),np.uint16(np.rint(cf*65535)))
 mask=np.array(Image.open(out/'masks/T2_to_T3_confidence_filtered_correspondence_mask.png'))>0;assert mask.sum()==m['pair_QC']['effective_pixels'];assert np.array(Image.open(out/'masks/T2_T3_geometric_common_mask.png')).min()==255
 unknown=json.loads((out/'confidence/T1_referenced_cumulative_UNAVAILABLE.json').read_text());assert all(unknown[k] is None for k in ['T1_to_T2_transform','T1_to_T3_cumulative_transform','T1_cumulative_confidence','three_period_common_area']);assert m['pair_QC']['conditional_on_saved_similarity'];assert m['periods']==[1,2,3] and m['missing_periods']==[4]
 rows=json.loads((out/'evaluation/area_ratios.json').read_text());assert all(v['T1_T2_T3_common_area_px2'] is None for v in rows);area=(x1-x0)*(y1-y0);assert rows[1]['T2_T3_pair_native_common_area_px2']==area;np.testing.assert_allclose(rows[2]['T2_T3_pair_native_common_area_px2'],area*np.linalg.det(H[:2,:2]))
 proof=dict(original_file_hashes_exact=True,T2_native_crop_pixel_exact=True,T3_single_similarity_pixel_exact=True,no_elastic_RGB_warp=True,confidence_png_matches_float32=True,mask_counts_verified=True,area_uses_native_scale_determinant=True,T1_unknown_not_zero=True,only_Q1_3_Z3_scope=True)
 (out/'evaluation/verification.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
if __name__=='__main__':main()
